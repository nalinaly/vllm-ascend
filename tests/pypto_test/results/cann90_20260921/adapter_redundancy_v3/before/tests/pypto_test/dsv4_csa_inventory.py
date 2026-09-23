"""Record local P0 evidence without loading an entire DeepSeek model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import struct
import subprocess
import sys
from pathlib import Path

from dsv4_csa_env import activate, write_json


def command(*args: str, cwd: Path | None = None) -> dict:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def source_info(path: Path) -> dict:
    diff = command("git", "diff", "HEAD", cwd=path)
    return {
        "path": str(path),
        "head": command("git", "rev-parse", "HEAD", cwd=path),
        "status": command("git", "status", "--short", cwd=path),
        "tracked_diff_sha256": hashlib.sha256(diff["stdout"].encode()).hexdigest(),
    }


def checkpoint_inventory(checkpoint: Path, layer: int) -> dict:
    config = json.loads((checkpoint / "config.json").read_text())
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{layer}.attn."
    selected = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
    if not selected or config["compress_ratios"][layer] != 4:
        raise ValueError(f"layer {layer} is not a CSA layer with checkpoint tensors")
    tensors = {}
    for shard in sorted(set(selected.values())):
        path = checkpoint / shard
        with path.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        for name, mapped_shard in selected.items():
            if mapped_shard != shard:
                continue
            descriptor = header[name]
            if descriptor["data_offsets"][1] + header_length + 8 > path.stat().st_size:
                raise ValueError(f"truncated checkpoint tensor: {name}")
            tensors[name] = {"shard": shard, **descriptor}
    return {
        "checkpoint": str(checkpoint.resolve()),
        "role": "local_reference_checkpoint_not_confirmed_final_16card_target",
        "layer": layer,
        "compress_ratio": 4,
        "config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        "dimensions": {
            key: config[key]
            for key in (
                "hidden_size",
                "head_dim",
                "qk_rope_head_dim",
                "num_attention_heads",
                "q_lora_rank",
                "o_groups",
                "o_lora_rank",
                "index_head_dim",
                "index_n_heads",
                "index_topk",
            )
        },
        "quantization_config": config.get("quantization_config"),
        "selected_tensor_count": len(tensors),
        "selected_storage_bytes": sum(t["data_offsets"][1] - t["data_offsets"][0] for t in tensors.values()),
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = activate()
    import torch
    import torch_npu

    report = {
        "phase": "P0",
        "status": "PENDING",
        "environment_scope": "local_2card",
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "npu_count": torch.npu.device_count(),
        "npu_names": [torch.npu.get_device_name(i) for i in range(torch.npu.device_count())],
        "npu_smi": command("npu-smi", "info"),
        "sources": {},
        "imports": {},
        "layout_target": {"weight_nz_mode": 0, "weight_layout": "ND", "kv_layout": "PA_ND"},
        "unverified": ["native_loaded_weight_layout", "native_custom_ops", "numerical_tolerances", "P1-P4"],
    }
    for label, path in {
        "vllm_ascend": repo,
        "pypto_lib": repo.parent / "pypto-lib",
        "pypto": repo.parent / "pto_eager/pypto",
        "simpler": repo.parent / "pto_eager/simpler",
        "vllm": repo / ".cache/csa/vllm-84030bbe",
    }.items():
        report["sources"][label] = source_info(path)
    for name in ("pypto", "simpler_setup", "vllm", "vllm_ascend", "vllm_ascend.attention.dsa_v1"):
        try:
            module = importlib.import_module(name)
            report["imports"][name] = {"status": "PASS", "path": module.__file__}
        except Exception as exc:
            report["imports"][name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    inventory = checkpoint_inventory(args.checkpoint, args.layer)
    write_json(args.output_dir / "checkpoint_inventory.json", inventory)
    write_json(args.output_dir / "manifest.json", report)
    print(
        json.dumps(
            {
                "npu_count": report["npu_count"],
                "imports": report["imports"],
                "reference_tensor_count": inventory["selected_tensor_count"],
                "reference_storage_bytes": inventory["selected_storage_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
