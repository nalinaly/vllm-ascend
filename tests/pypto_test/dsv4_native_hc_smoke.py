"""Execute release Native HcPre/HcPost with formal checkpoint weights."""

import argparse
import json
from pathlib import Path
import traceback

from dsv4_csa_env import activate, load_native_extension, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = activate()
    report = {"status": "FAIL", "scope": "Native HC execution smoke, not full-model accuracy"}
    try:
        import torch
        import torch_npu  # noqa: F401
        from safetensors import safe_open

        torch.npu.set_device(args.device)
        load_native_extension(repo)
        checkpoint = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
        cfg = json.loads((checkpoint / "config.json").read_text())
        index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
        device = f"npu:{args.device}"
        weights = []
        for suffix in ("fn", "scale", "base"):
            name = f"layers.2.hc_attn_{suffix}"
            with safe_open(checkpoint / index[name], framework="pt", device="cpu") as shard:
                weights.append(shard.get_tensor(name).to(device))
        torch.manual_seed(1024)
        hidden = torch.randn((6, cfg["hc_mult"], cfg["hidden_size"]), dtype=torch.bfloat16).to(device)
        pre, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
            hidden, *weights, cfg["hc_mult"], cfg["hc_sinkhorn_iters"], cfg["rms_norm_eps"], cfg["hc_eps"]
        )
        result = torch.ops._C_ascend.npu_hc_post(
            pre.unsqueeze(0), hidden.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
        ).squeeze(0)
        torch.npu.synchronize()
        assert pre.shape == (6, cfg["hidden_size"]) and result.shape == hidden.shape
        assert all(torch.isfinite(x).all().item() for x in (pre, post, comb, result))
        report.update(status="PASS", checkpoint=str(checkpoint),
                      outputs={name: {"shape": list(x.shape), "dtype": str(x.dtype)}
                               for name, x in zip(("pre", "post", "comb", "result"), (pre, post, comb, result))})
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "native_hc.json", report)


if __name__ == "__main__":
    main()
