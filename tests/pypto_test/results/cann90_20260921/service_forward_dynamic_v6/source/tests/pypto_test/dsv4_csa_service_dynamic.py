"""Switch real request counts through one prepared service layer and JIT artifact."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import traceback

from dsv4_csa_env import activate, load_native_extension, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--batches', default='1,2,3,5,40,1')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(',')]
    repo = activate()
    report = {'status': 'FAIL', 'checkpoint': str(args.checkpoint), 'batches': batches, 'cases': []}
    try:
        import torch
        from dsv4_csa_native_fixture import make_config, make_attention, native_session
        from dsv4_csa_native_layout import load_layer_weights
        from dsv4_csa_native_forward import make_numerical_fixture
        from dsv4_csa_full_compare import compare_tensor, check_untouched
        from dsv4_csa_full_replay import tolerance
        from vllm.forward_context import BatchDescriptor
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.models.deepseek_v4.model import DeepseekV4Attention
        from vllm_ascend.models.pypto_deepseek_v4 import install_csa_forward, prepare_csa_model
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.decode_csa import decode_csa_tp1_attention_test
        from pypto.runtime.kernel.context import get_process_kernel_state

        config = make_config(args.checkpoint)
        config.scheduler_config.max_num_seqs = max(40, max(batches))
        device = torch.device(f'npu:{args.device}')
        with native_session(config, args.device), torch.inference_mode():
            attention = make_attention(config, device)
            load_layer_weights(attention, args.checkpoint)
            install_csa_forward(attention)
            prepare_csa_model(SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])))
            runtime = attention.dsa_attn._pto_csa_runtime
            operator = runtime.operators.attention
            submitted = []
            def count_call(*inputs):
                submitted.append(inputs[0].shape[0] // 6)
                return operator(*inputs)
            runtime.operators = SimpleNamespace(attention=count_call)
            artifact_ids = None
            captured = {}
            hook = attention.indexer.register_forward_hook(
                lambda module, inputs, output: captured.update(topk=output.detach().clone()))

            @contextmanager
            def context(fixture):
                for group in fixture['groups'].values():
                    group['owner'].kv_cache = group['views']
                descriptor = BatchDescriptor(num_tokens=fixture['actual'], num_reqs=fixture['actual']//6, uniform=True)
                fixture['executor'].submit(fixture['tasks'], batch_descriptor=descriptor)
                try:
                    with set_ascend_forward_context(fixture['metadata'], config,
                        num_tokens=fixture['actual'], num_actual_tokens=fixture['actual'],
                        batch_descriptor=descriptor, device_metadata_executor=fixture['executor']):
                        yield
                finally:
                    fixture['executor'].release()

            for batch in batches:
                native = make_numerical_fixture(config, device, attention, batch, 131071, 1024)
                pto = make_numerical_fixture(config, device, attention, batch, 131071, 1024)
                initial = {name: group['allocation'].cpu() for name, group in pto['groups'].items()}
                with context(native):
                    expected = DeepseekV4Attention.forward(attention, native['positions'], native['hidden'], None)
                with context(pto):
                    actual = attention(pto['positions'], pto['hidden'], None)
                torch.npu.synchronize()
                checks = {'output': compare_tensor(actual, expected, 1e-2, 1e-2),
                          'topk': compare_tensor(runtime.topk[:pto['actual']], captured['topk'].reshape(-1, 512), 0, 0)}
                for name, group in pto['groups'].items():
                    for index, value in enumerate(group['views']):
                        checks[f'cache.{name}.{index}'] = compare_tensor(
                            value, native['groups'][name]['views'][index], *tolerance(value.dtype))
                checks.update(check_untouched(pto, initial))
                current_ids = tuple(id(value.artifact) for key, value in
                    get_process_kernel_state()._specializations.items() if key[0] is decode_csa_tp1_attention_test)
                assert len(current_ids) == 1, 'expected one dynamic compiled artifact'
                if artifact_ids is None:
                    artifact_ids = current_ids
                assert current_ids == artifact_ids, 'batch change compiled another artifact'
                assert len(submitted) == len(report['cases']) + 1 and submitted[-1] == batch
                status = 'PASS' if all(check['status'] == 'PASS' for check in checks.values()) else 'FAIL'
                report['cases'].append({'batch': batch, 'status': status, 'checks': checks})
                print(f'dynamic service B{batch}: {status}, compiled artifact unchanged', flush=True)
                del native, pto, initial, expected, actual
            hook.remove()
            report['operator_batches'] = submitted
            report['compiled_artifacts'] = 1
            assert all(case['status'] == 'PASS' for case in report['cases'])
            report['status'] = 'PASS'
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir/'service_dynamic.json', report)


if __name__ == '__main__':
    main()
