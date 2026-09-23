"""Capture full-stream main-compressor inputs at a selected continuous step."""

class MainCompressorBoundary:
    def __init__(self, attention):
        self.enabled = False
        self.captured = {}
        self.hook = attention.compressor.register_forward_hook(self.capture)

    def capture(self, module, inputs, result):
        if self.enabled:
            self.captured['native.output'] = result[0].detach().clone()
            self.captured['native.slots'] = result[1].detach().clone()

    def begin(self, native, call):
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_storage import physical_pages
        self.captured['native.state_before'] = physical_pages(native['groups']['state']['views'][0]).clone()
        self.captured['pto.state_before'] = call.state_storage['state'].clone()

    def observe(self, native, call, output_dir):
        import torch
        from dsv4_csa_env import write_json
        from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_storage import physical_pages
        a = call.args
        dump = {k: v.cpu() for k, v in self.captured.items()}
        dump['native.state_after'] = physical_pages(native['groups']['state']['views'][0]).cpu()
        dump['pto.state_after'] = call.state_storage['state'].cpu()
        for key in ('x_normed_t', 'cmp_wkv', 'cmp_wgate', 'cmp_ape', 'cmp_norm_w', 'position_ids',
                    'state_block_table', 'state_slot_mapping', 'cmp_query_start_loc', 'cmp_seq_lens',
                    'cmp_slot_mapping', 'cmp_freqs_cos', 'cmp_freqs_sin'):
            dump[key] = a[key].cpu()
        slots = a['cmp_slot_mapping'].long()
        valid = (slots[:, 0] >= 0) & (slots[:, 1] >= 0)
        rows = slots[valid, 0] * 32 + slots[valid, 1]
        dump['pto.cache_written'] = a['cmp_kv'].view(-1, 512)[rows].cpu()
        dump['native.cache_written'] = native['groups']['compressed']['views'][0].view(-1, 512)[rows].cpu()
        dump['valid_compact_rows'] = torch.nonzero(valid).flatten().cpu()
        torch.save(dump, output_dir / 'main_compressor_boundary.pt')
        report = {'status': 'CAPTURED', 'scope': 'main compressor production inputs and Native/PTO states, no numeric acceptance'}
        write_json(output_dir / 'main_compressor_boundary.json', report)
        return report

    def close(self):
        self.hook.remove()
