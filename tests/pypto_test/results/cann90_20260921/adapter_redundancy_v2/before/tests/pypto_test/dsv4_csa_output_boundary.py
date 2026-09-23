"""Replay production Query, sparse attention and O projection at a failing step."""


class OutputBoundary:
    def __init__(self, attention):
        import torch_npu

        self.enabled = False
        self.captured = {}
        self.impl = attention.dsa_attn.dsa_attn.impl
        self.prolog_single = self.impl._mla_prolog_single_stream
        self.prolog_multi = self.impl._mla_prolog_multistream
        self.impl._mla_prolog_single_stream = self.capture_single_prolog
        self.impl._mla_prolog_multistream = self.capture_multi_prolog
        self.attention = self.impl._forward_attention
        self.impl._forward_attention = self.capture_attention
        self.projection = self.impl._forward_o_proj
        self.impl._forward_o_proj = self.capture_projection
        self.wo_b_hook = attention.wo_b.register_forward_pre_hook(self.capture_wo_b_input)
        self.torch_npu = torch_npu
        self.quantize = torch_npu.npu_dynamic_quant
        torch_npu.npu_dynamic_quant = self.capture_quantize

    def capture_prolog_result(self, result):
        if self.enabled:
            for name, value in zip(("query", "qr", "qr_scale"), result[:3]):
                if value is not None:
                    self.captured[name] = value.detach().clone()
        return result

    def capture_single_prolog(self, *args, **kwargs):
        return self.capture_prolog_result(self.prolog_single(*args, **kwargs))

    def capture_multi_prolog(self, *args, **kwargs):
        return self.capture_prolog_result(self.prolog_multi(*args, **kwargs))

    def capture_attention(self, *args, **kwargs):
        result = self.attention(*args, **kwargs)
        if self.enabled:
            self.captured["attention_output"] = result.detach().clone()
        return result

    def capture_projection(self, value, output):
        if self.enabled:
            self.captured["projection_input"] = value.detach().clone()
        result = self.projection(value, output)
        if self.enabled:
            self.captured["output"] = result.detach().clone()
        return result

    def capture_wo_b_input(self, module, inputs):
        if self.enabled:
            self.captured["wo_a"] = inputs[0].detach().clone()

    def capture_quantize(self, value, *args, **kwargs):
        result = self.quantize(value, *args, **kwargs)
        if self.enabled and value.ndim == 2 and value.shape[1] == 8192:
            self.captured["wo_b_quantized"] = result[0].detach().clone()
            self.captured["wo_b_scale"] = result[1].detach().clone()
        return result

    def close(self):
        self.impl._mla_prolog_single_stream = self.prolog_single
        self.impl._mla_prolog_multistream = self.prolog_multi
        self.impl._forward_attention = self.attention
        self.impl._forward_o_proj = self.projection
        self.wo_b_hook.remove()
        self.torch_npu.npu_dynamic_quant = self.quantize

    def observe(self, call, expected, output_dir, *, native):
        import torch
        from dsv4_csa_env import write_json
        from dsv4_csa_full_compare import compare_tensor
        from dsv4_csa_precision_kernels import (
            HEAD_DIM, O_GROUPS, O_GROUP_IN, O_PROJ_T_PAD,
            diagnose_main_query, diagnose_o_projection, diagnose_sparse_attention,
        )

        value = self.captured["projection_input"]
        rows = value.shape[0]
        packed = torch.zeros((O_GROUPS, O_PROJ_T_PAD, O_GROUP_IN), dtype=value.dtype, device=value.device)
        packed[:, :rows].copy_(value.reshape(rows, O_GROUPS, O_GROUP_IN).transpose(0, 1))
        output = torch.empty_like(expected)
        a = call.args
        diagnose_o_projection(packed.view(O_GROUPS * O_PROJ_T_PAD, O_GROUP_IN),
                              a["wo_a"], a["wo_b"], a["wo_b_scale"], output)
        torch.npu.synchronize()
        checks = {
            "captured_native_output": compare_tensor(self.captured["output"], expected, 0, 0),
            "given_native_projection_input": compare_tensor(output, expected, 1e-2, 1e-2),
            "given_native_projection_input_exact": compare_tensor(output, expected, 0, 0),
        }
        dump = {f"native.{name}": tensor.cpu() for name, tensor in self.captured.items()}
        dump.update({"pto.given_native_projection_input": output.cpu(), "pto.full_output": a["attn_out"].cpu(),
                     "wo_a": a["wo_a"].cpu(), "wo_b": a["wo_b"].cpu(), "wo_b_scale": a["wo_b_scale"].cpu()})
        query = torch.empty_like(self.captured["query"])
        qr = torch.empty_like(self.captured["qr"])
        qr_scale = torch.empty((rows, 1), dtype=torch.float32, device=value.device)
        diagnose_main_query(a["x_normed_t"], a["wq_a"], a["wq_b"], a["wq_b_scale"], a["wkv"],
                            a["freqs_cos"], a["freqs_sin"], a["gamma_cq"], a["gamma_ckv"],
                            query, qr, qr_scale)
        torch.npu.synchronize()
        for name, tensor in (("query", query), ("qr", qr), ("qr_scale", qr_scale)):
            checks[name + "_exact"] = compare_tensor(tensor, self.captured[name].reshape(tensor.shape), 0, 0)
            dump["pto." + name] = tensor.cpu()
        for label, query_input, swa, compressed in (
            ("pto_query_pto_cache", query, a["kv_cache"], a["cmp_kv"]),
            ("native_query_pto_cache", self.captured["query"], a["kv_cache"], a["cmp_kv"]),
            ("native_query_native_cache", self.captured["query"],
             native["groups"]["swa"]["views"][0], native["groups"]["compressed"]["views"][0]),
        ):
            diagnose_sparse_attention(query_input, swa, a["window_swa_indices"], compressed,
                                       a["cmp_block_table"], a["idx_topk"], a["position_ids"].view(rows, 1),
                                       a["attn_sink"], a["freqs_cos"], a["freqs_sin"],
                                       packed.view(O_GROUPS * O_PROJ_T_PAD, O_GROUP_IN))
            diagnose_o_projection(packed.view(O_GROUPS * O_PROJ_T_PAD, O_GROUP_IN),
                                  a["wo_a"], a["wo_b"], a["wo_b_scale"], output)
            torch.npu.synchronize()
            projection_input = packed[:, :rows].transpose(0, 1).reshape(value.shape)
            checks[label] = {
                "projection_input": compare_tensor(projection_input, value, 1e-2, 1e-2),
                "projection_input_exact": compare_tensor(projection_input, value, 0, 0),
                "output": compare_tensor(output, expected, 1e-2, 1e-2),
                "output_exact": compare_tensor(output, expected, 0, 0),
            }
            if label == "pto_query_pto_cache":
                checks["replayed_production_output"] = compare_tensor(output, a["attn_out"], 0, 0)
            dump["pto." + label + ".projection_input"] = projection_input.cpu()
            dump["pto." + label + ".output"] = output.cpu()
        # Save only the selected key/value rows for CPU numerical analysis.
        window = a["window_swa_indices"].long()
        topk = a["idx_topk"].long()
        positions = a["position_ids"].long().view(rows, 1)
        valid = torch.cat((window >= 0, (topk >= 0) & (topk < (positions + 1) // 4)), dim=1)
        tables = a["cmp_block_table"][torch.arange(rows, device=value.device) // 6]
        compressed_pages = tables.gather(1, topk.clamp_min(0) // 32).long()
        compressed_slots = compressed_pages * 32 + topk.clamp_min(0) % 32
        for label, swa, compressed in (
            ("pto", a["kv_cache"], a["cmp_kv"]),
            ("native", native["groups"]["swa"]["views"][0], native["groups"]["compressed"]["views"][0]),
        ):
            selected = torch.cat((swa.view(-1, HEAD_DIM)[window.clamp_min(0)],
                                  compressed.view(-1, HEAD_DIM)[compressed_slots]), dim=1)
            dump[label + ".selected_kv"] = selected.cpu()
        dump.update({"selected_valid": valid.cpu(), "sink": a["attn_sink"].cpu(),
                     "cos": a["freqs_cos"].cpu(), "sin": a["freqs_sin"].cpu(),
                     "rope_layout": "native_interleaved_fp32"})
        torch.save(dump, output_dir / "output_boundary.pt")
        write_json(output_dir / "output_boundary.json", checks)
        return checks
