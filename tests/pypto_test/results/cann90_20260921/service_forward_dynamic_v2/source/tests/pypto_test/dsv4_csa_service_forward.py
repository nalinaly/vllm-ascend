"""Formal-weight service forward, Native fallback and graph integration smoke."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import traceback

from dsv4_csa_env import activate, load_native_extension, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--dummy-capture', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    repo = activate()
    report = {'status': 'FAIL', 'scope': 'real single-layer service forward, not whole-model/P5 acceptance',
              'checkpoint': str(args.checkpoint), 'batch': args.batch, 'cases': []}
    try:
        import torch
        import torch_npu
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor
        from dsv4_csa_native_fixture import make_config, make_attention, native_session
        from dsv4_csa_native_layout import load_layer_weights
        from dsv4_csa_native_forward import make_numerical_fixture, update_numerical_metadata
        from dsv4_csa_full_compare import compare_tensor, check_untouched
        from dsv4_csa_full_replay import restore, tolerance
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.models.deepseek_v4.model import DeepseekV4Attention
        from vllm_ascend.models.pypto_deepseek_v4 import install_csa_forward, prepare_csa_model
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service_config import validate_configuration

        assert (args.checkpoint/'quant_model_description.json').is_file()
        config = make_config(args.checkpoint, full_decode_graph=True)
        device = torch.device(f'npu:{args.device}')
        with native_session(config, args.device), torch.inference_mode():
            validate_configuration(config)
            attention = make_attention(config, device)
            records, methods = load_layer_weights(attention, args.checkpoint)
            report.update(weights=records, quant_methods=methods)
            native = make_numerical_fixture(config, device, attention, args.batch, 131071, 1024)
            pto = make_numerical_fixture(config, device, attention, args.batch, 131071, 1024)
            snapshots = {name: g['allocation'].cpu() for name,g in native['groups'].items()}
            for name,g in pto['groups'].items():
                torch.testing.assert_close(g['allocation'].cpu(), snapshots[name], atol=0, rtol=0)
            install_csa_forward(attention)
            prepare_csa_model(SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])))
            runtime = attention.dsa_attn._pto_csa_runtime
            original_op = runtime.operators.attention
            calls = []
            def count_call(*inputs):
                calls.append(1)
                return original_op(*inputs)
            runtime.operators = SimpleNamespace(attention=count_call)
            @contextmanager
            def context(fixture, graph=False, metadata=True):
                descriptor = BatchDescriptor(num_tokens=fixture['actual'],
                                             num_reqs=fixture['actual'] // 6, uniform=True)
                for g in fixture['groups'].values():
                    g['owner'].kv_cache = g['views']
                if metadata:
                    fixture['executor'].submit(fixture['tasks'], batch_descriptor=descriptor)
                    assert fixture['executor'].uses_external_events
                    report['native_external_events'] = True
                try:
                    with set_ascend_forward_context(
                        fixture['metadata'] if metadata else None, config,
                        num_tokens=fixture['actual'], num_actual_tokens=fixture['actual'],
                        aclgraph_runtime_mode=CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE,
                        batch_descriptor=descriptor, device_metadata_executor=fixture['executor'] if metadata else None,
                    ):
                        yield
                finally:
                    if metadata:
                        fixture['executor'].release()

            with context(pto, metadata=False):
                profiled = attention(pto['positions'], pto['hidden'], None)
            torch.npu.synchronize()
            assert not calls and torch.count_nonzero(profiled).item() == 0
            report['profiling_native_fallback'] = True
            if args.dummy_capture:
                # Match the MRv1 startup inputs: seq_len=S, positions=127,
                # empty page tables and invalid real-request slots. Replay
                # below restores real requests at these same device addresses.
                from vllm_ascend.attention.attention_v1 import AscendAttentionState
                pto['positions'].fill_(127)
                tasks, shared = [], {}
                for group in pto['groups'].values():
                    common, builder = group['common'], group['builder']
                    common.seq_lens.fill_(6)
                    common._seq_lens_cpu = torch.full((args.batch,), 6, dtype=torch.int32)
                    common.max_seq_len = 6
                    common.attn_state = AscendAttentionState.DecodeOnly
                    common.block_table_tensor.zero_()
                    common.slot_mapping.fill_(-1)
                    pto['metadata'][group['prefix']] = builder.build_for_cudagraph_capture(
                        common, common_ratio_to_sas_metadata=shared, num_actual_reqs=args.batch,
                    )
                    tasks.extend(builder.take_device_metadata_tasks())
                pto['tasks'] = tasks
            with context(pto):
                attention(pto['positions'], pto['hidden'], None)
            torch.npu.synchronize()
            assert len(calls) == 1, 'service forward silently fell back during PTO warmup'
            graph = torch_npu.npu.NPUGraph()
            with context(pto, graph=True), torch_npu.npu.graph(graph):
                graph_output = attention(pto['positions'], pto['hidden'], None)
            torch.npu.synchronize()
            assert len(calls) == 2, 'capture did not enter PTO CSA'
            report['capture_uses_pto'] = True
            report['startup_dummy_capture'] = args.dummy_capture
            topk_capture = {}
            hook = attention.indexer.register_forward_hook(lambda module, inputs, output: topk_capture.update(value=output.detach().clone()))
            for iteration, variant in enumerate((0, 1, 0)):
                starts = torch.full((args.batch,), 131071 + 2 * variant, dtype=torch.int32)
                restore(native, snapshots)
                update_numerical_metadata(native, starts, swap_tables=iteration > 0)
                with context(native):
                    expected = DeepseekV4Attention.forward(attention, native['positions'], native['hidden'], None)
                torch.npu.synchronize()
                checks = {f'native_guard.{k}':v for k,v in check_untouched(native, snapshots).items()}
                restore(pto, snapshots)
                update_numerical_metadata(pto, starts, swap_tables=iteration > 0)
                with context(pto):
                    actual = attention(pto['positions'], pto['hidden'], None)
                torch.npu.synchronize()
                checks['eager_output'] = compare_tensor(actual, expected, 1e-2, 1e-2)
                checks['topk'] = compare_tensor(runtime.topk[:pto['actual']], topk_capture['value'].reshape(pto['actual'],512), 0, 0)
                for name,g in pto['groups'].items():
                    for i,view in enumerate(g['views']):
                        checks[f'cache.{name}.{i}'] = compare_tensor(view, native['groups'][name]['views'][i], *tolerance(view.dtype))
                eager_output = actual.cpu()
                eager_allocations = {name:g['allocation'].cpu() for name,g in pto['groups'].items()}
                checks.update({f'eager_guard.{k}':v for k,v in check_untouched(pto, snapshots).items()})
                restore(pto, snapshots)
                count_before = len(calls)
                with context(pto, graph=True):
                    graph.replay()
                torch.npu.synchronize()
                assert len(calls) == count_before, 'replay re-entered Python dispatch'
                checks['graph_output'] = compare_tensor(graph_output.cpu(), eager_output, 0, 0)
                for name,g in pto['groups'].items():
                    checks[f'graph_allocation.{name}'] = compare_tensor(g['allocation'].cpu(), eager_allocations[name], 0, 0)
                checks.update({f'graph_guard.{k}':v for k,v in check_untouched(pto, snapshots).items()})
                assert all(c['status']=='PASS' for c in checks.values()), checks
                report['cases'].append({'variant':variant,'checks':checks,'status':'PASS'})
                print(f'service B{args.batch} variant {variant}: PASS', flush=True)
            hook.remove()
            # Exercise FakeTensor/schema lowering of the actual installed forward.
            compiled = torch.compile(attention, backend='eager', fullgraph=True)
            restore(pto, snapshots)
            with context(pto):
                compiled_output = compiled(pto['positions'], pto['hidden'], None)
            torch.npu.synchronize()
            report['compiled_forward'] = compare_tensor(compiled_output.cpu(), eager_output, 0, 0)
            assert report['compiled_forward']['status'] == 'PASS'
            # A max-query bound above S6 cannot prove uniformity and must
            # use Native even when these particular requests have six rows.
            small = make_numerical_fixture(config, device, attention, 2, 131071, 1024)
            for item in small['metadata'].values():
                item.max_query_len = 7
            small_initial = {name:g['allocation'].cpu() for name,g in small['groups'].items()}
            with context(small):
                expected_small = DeepseekV4Attention.forward(attention, small['positions'], small['hidden'], None)
            torch.npu.synchronize()
            small_final = {name:g['allocation'].cpu() for name,g in small['groups'].items()}
            restore(small, small_initial)
            count_before = len(calls)
            with context(small):
                actual_small = attention(small['positions'], small['hidden'], None)
            torch.npu.synchronize()
            assert len(calls) == count_before
            report['nonuniform_bound_native_fallback'] = compare_tensor(actual_small, expected_small, 0, 0)
            assert report['nonuniform_bound_native_fallback']['status']=='PASS'
            for name,g in small['groups'].items():
                assert torch.equal(g['allocation'].cpu(), small_final[name]), name
            report['operator_submissions_outside_replay'] = len(calls)
            report['status'] = 'PASS'
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir/'service_forward.json', report)


if __name__ == '__main__':
    main()
