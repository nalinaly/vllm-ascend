"""Capture a failing step and reuse production QA/QR/Indexer query helpers."""


class QueryBoundary:
    def __init__(self, attention):
        from vllm_ascend.models.deepseek_v4 import indexer

        self.enabled = False
        self.captured = {}
        self.indexer_module = indexer
        self.ops = attention.indexer.ops
        self.qa = attention.dsa_attn.dsa_attn.impl.cv_wq_a
        self.select = self.ops.select_topk
        self.matmul = self.qa.matmul
        self.hadamard = indexer.hadamard_linear
        self.ops.select_topk = self.capture_select
        self.qa.matmul = self.capture_matmul
        indexer.hadamard_linear = self.capture_hadamard
        self.hook = attention.indexer.register_forward_pre_hook(self.capture_inputs, with_kwargs=True)
        self.weights_hook = attention.indexer.weights_proj.register_forward_hook(self.capture_weights)

    def capture_weights(self, module, inputs, result):
        if self.enabled:
            self.captured["raw_weights"] = result.detach().clone()

    def capture_select(self, query, weights, scale, *arguments):
        if self.enabled:
            self.captured.update(
                query=query.detach().clone(), weights=weights.detach().clone(), query_scale=scale.detach().clone()
            )
        return self.select(query, weights, scale, *arguments)

    def capture_matmul(self, *arguments, **kwargs):
        result = self.matmul(*arguments, **kwargs)
        if self.enabled:
            self.captured["qa"] = result.detach().clone()
        return result

    def capture_hadamard(self, x, matrix):
        if self.enabled and x.ndim == 3 and x.shape[1] == 64:
            self.captured["before_hadamard"] = x.detach().clone()
        return self.hadamard(x, matrix)

    def capture_inputs(self, module, inputs, kwargs):
        if self.enabled:
            self.captured["qr"] = kwargs["qr"].detach().clone()
            self.captured["qr_scale"] = kwargs["qr_pertoken_scale"].detach().clone()

    def close(self):
        self.ops.select_topk = self.select
        self.qa.matmul = self.matmul
        self.indexer_module.hadamard_linear = self.hadamard
        self.hook.remove()
        self.weights_hook.remove()

    def observe(self, call, native_topk, output_dir):
        import torch
        from dsv4_csa_env import write_json
        from dsv4_csa_full_compare import compare_tensor
        from dsv4_csa_precision_kernels import (
            QPROJ_T_PAD,
            T_PAD,
            diagnose_indexer_query,
            diagnose_indexer_weights,
            diagnose_qa,
            diagnose_qr,
        )

        a = call.args
        x = a["x_normed_t"]
        rows = x.shape[0]
        qa = torch.empty((QPROJ_T_PAD, 1024), dtype=torch.float32, device=x.device)
        qr = torch.empty((rows, 1024), dtype=torch.int8, device=x.device)
        scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
        diagnose_qa(x, a["wq_a"], qa)
        diagnose_qr(x, a["wq_a"], a["gamma_cq"], qr, scale)
        before_hadamard = torch.empty((T_PAD * 64, 128), dtype=torch.bfloat16, device=x.device)
        query = torch.empty((T_PAD * 64, 128), dtype=torch.int8, device=x.device)
        query_scale = torch.empty((T_PAD * 64, 1), dtype=torch.float32, device=x.device)
        sign = (torch.arange(64, device=x.device) % 2 * 2 - 1).float()
        diagnose_indexer_query(
            x,
            qr,
            scale,
            a["idx_wq_b"],
            a["idx_wq_b_scale"],
            call.native_cos,
            call.native_sin * sign,
            a["hadamard_idx"],
            before_hadamard,
            query,
            query_scale,
        )
        weights = torch.empty((T_PAD, 64), dtype=torch.float32, device=x.device)
        coefficients = torch.empty((T_PAD, 64), dtype=torch.float16, device=x.device)
        diagnose_indexer_weights(x, a["weights_proj"], a["position_ids"], query_scale, weights, coefficients)
        torch.npu.synchronize()
        actual = {
            "qa": qa[:rows].bfloat16(),
            "qr": qr,
            "qr_scale": scale,
            "before_hadamard": before_hadamard[: rows * 64],
            "query": query[: rows * 64],
            "query_scale": query_scale[: rows * 64],
        }
        checks = {
            name: compare_tensor(value, self.captured[name].reshape(value.shape), 0, 0)
            for name, value in actual.items()
        }
        checks["weights"] = compare_tensor(weights[:rows], self.captured["weights"].half().float(), 0, 0)
        cpu_coefficients = (
            self.captured["weights"].cpu().half().float() * self.captured["query_scale"].cpu().float()
        ).half()
        checks["coefficients_vs_cpu_rounding"] = compare_tensor(coefficients[:rows].cpu(), cpu_coefficients, 0, 0)
        actual.update(weights=weights[:rows], coefficients=coefficients[:rows])
        dump = {f"native.{name}": value.cpu() for name, value in self.captured.items()}
        dump.update({f"pto.{name}": value.cpu() for name, value in actual.items()})
        dump.update(
            hidden=x.cpu(),
            positions=a["position_ids"].cpu(),
            native_topk=native_topk.cpu(),
            pto_topk=a["idx_topk"].cpu(),
            pto_scores=a["idx_topk_scores"].cpu(),
            key_cache=call.views["indexer"][0].cpu(),
            scale_cache=call.views["indexer"][1].cpu(),
            block_table=call.req["indexer"].block_table.cpu(),
            weights_proj=a["weights_proj"].cpu(),
        )
        torch.save(dump, output_dir / "query_boundary.pt")
        write_json(output_dir / "query_boundary.json", checks)
        return checks
