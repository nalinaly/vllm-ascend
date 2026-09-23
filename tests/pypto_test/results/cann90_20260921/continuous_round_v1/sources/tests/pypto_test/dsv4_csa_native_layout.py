"""Load one real W8A8 CSA layer and inspect all five native cache layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

from dsv4_csa_env import activate, load_native_extension, write_json


def load_layer_weights(attention, checkpoint: Path):
    import torch
    from safetensors import safe_open
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    prefix = "layers.2.attn."
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
    parameters = dict(attention.named_parameters())
    records = []
    loaded = set()
    with torch.no_grad():
        for shard in sorted(set(selected.values())):
            with safe_open(checkpoint / shard, framework="pt", device="cpu") as reader:
                for name, mapped_shard in selected.items():
                    if mapped_shard != shard:
                        continue
                    suffix = name.removeprefix(prefix)
                    # Match DeepseekV4ForCausalLM.load_weights for checkpoint
                    # releases that store channel scales as `<linear>.scale`.
                    if suffix.endswith(".scale"):
                        suffix = suffix.removesuffix(".scale") + ".weight_scale"
                    value = reader.get_tensor(name)
                    parameter = parameters[suffix]
                    load_value = value
                    # This reference export stores channel scales as [out],
                    # while Ascend allocates [out, 1] before post-load flattening.
                    # Canonicalize only that singleton axis, never weight data.
                    if (
                        suffix.endswith(".weight_scale")
                        and value.ndim == 1
                        and tuple(parameter.shape) == (value.numel(), 1)
                    ):
                        load_value = value[:, None]
                    getattr(parameter, "weight_loader", default_weight_loader)(parameter, load_value)
                    torch.testing.assert_close(parameter.detach().cpu(), load_value.to(parameter.dtype), rtol=0, atol=0)
                    loaded.add(suffix)
                    records.append(
                        {
                            "name": name,
                            "parameter_name": suffix,
                            "shape": list(value.shape),
                            "loaded_shape": list(parameter.shape),
                            "checkpoint_dtype": str(value.dtype),
                            "loaded_dtype": str(parameter.dtype),
                            "sha256": hashlib.sha256(
                                value.contiguous().view(torch.uint8).numpy().tobytes()
                            ).hexdigest(),
                            "loaded_exact": True,
                        }
                    )
        missing = set(parameters) - loaded
        # Ascend allocates offsets even for a symmetric compressed-tensors
        # checkpoint, which stores only weight and scale. Derive zeros only
        # after checking the checkpoint declares symmetric integer weights.
        description = json.loads((checkpoint / "config.json").read_text())["quantization_config"]
        symmetric = all(scheme["weights"]["symmetric"] for scheme in description["config_groups"].values())
        for name in tuple(missing):
            if symmetric and name.endswith(".weight_offset") and name.removesuffix("_offset") in loaded:
                parameters[name].zero_()
                records.append({"name": name, "source": "derived_zero_symmetric_weight_offset"})
                missing.remove(name)
        if missing:
            raise ValueError(f"unloaded attention parameters: {sorted(missing)}")
        methods = {}
        for name, module in attention.named_modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                methods[name] = type(method).__module__ + "." + type(method).__name__
                method.process_weights_after_loading(module)
    return records, methods


def allocate_native_cache(group, pages, device):
    """Call the real runner's page-layout routine on guarded raw allocations."""
    import torch

    from vllm_ascend.core.kv_cache_interface import get_storage_block_size
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    spec = group["spec"]
    backend = group["owner"].get_attn_backend()
    page_size = get_storage_block_size(spec)
    shape = backend.get_kv_cache_shape(pages, page_size, spec.num_kv_heads, spec.head_size)
    shapes, dtypes = [shape], [spec.dtype]
    if getattr(spec, "scale_dim", 0):
        shapes.append(backend.get_kv_cache_shape(pages, page_size, spec.num_kv_heads, spec.scale_dim))
        dtypes.append(spec.scale_dtype)
    guard_bytes = 128
    allocation = torch.full((pages * spec.page_size_bytes + 2 * guard_bytes,), 37, dtype=torch.int8, device=device)
    raw = allocation[guard_bytes:-guard_bytes]
    views = NPUModelRunner._adjust_kv_layout(None, raw, shapes, dtypes, spec.page_size_bytes)
    group.update(allocation=allocation, raw=raw, views=views, page_size=page_size, pages=pages)
    return {
        "spec": repr(spec),
        "logical_block_size": spec.block_size,
        "storage_block_size": page_size,
        "page_bytes": spec.page_size_bytes,
        "raw_offset": raw.storage_offset(),
        "views": [
            {
                "shape": list(view.shape),
                "dtype": str(view.dtype),
                "stride": list(view.stride()),
                "storage_offset_elements": view.storage_offset(),
                "storage_offset_bytes": view.storage_offset() * view.element_size(),
                "contiguous": view.is_contiguous(),
                "storage_ptr": view.untyped_storage().data_ptr(),
            }
            for view in views
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = activate()
    report = {"case_id": "P0_REAL_LAYER_ND_AND_NATIVE_CACHE_LAYOUT", "status": "FAIL", "python": sys.executable}
    try:
        import torch
        import torch_npu
        from dsv4_csa_native_fixture import make_attention, make_cache_groups, make_config, native_session

        load_native_extension(repo)
        import vllm_ascend.ops  # noqa: F401

        config = make_config(args.checkpoint)
        device = torch.device(f"npu:{args.device}")
        with native_session(config, args.device):
            attention = make_attention(config, device)
            records, methods = load_layer_weights(attention, args.checkpoint)
            report.update(weights=records, quant_methods=methods)
            groups = make_cache_groups(config, device, attention)
            report["groups"] = {kind: allocate_native_cache(group, 7, device) for kind, group in groups.items()}
            formats = {
                name: {
                    "format": torch_npu.get_npu_format(parameter),
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "stride": list(parameter.stride()),
                }
                for name, parameter in attention.named_parameters()
            }
            assert all(value["format"] == 2 for value in formats.values()), formats
            report.update(
                status="PASS",
                loaded_formats=formats,
                quantization="local_reference_compressed_tensors_W8A8",
                scope="one_real_layer_loading_and_layout_not_forward_or_final_target_quantization",
            )
            print("native CSA single-layer weights and five cache layouts: PASS", flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / f"native_layer_layout_device{args.device}.json", report)


if __name__ == "__main__":
    main()
