"""CPU复现及回归：Native共享缓存的描述符别名检查。"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from dsv4_csa_env import activate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-accepted", action="store_true", help="验证上游修复后的行为")
    parser.add_argument("--layout-dir", type=Path, help="复查已采集的真实D16缓存描述符")
    args = parser.parse_args()
    activate()
    import torch
    from pypto.pypto_core import DataType
    from pypto.pypto_core.ir import ParamDirection
    from pypto.torch.interop import TensorArgument, TensorMetadata, _validate_aliases

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.native_storage import indexer_storage, physical_pages

    pages, page_bytes = 4, 4160
    storage = torch.empty(pages * page_bytes, dtype=torch.int8)
    state = storage.view(torch.float32).as_strided((pages, 2, 1, 512), (1040, 512, 512, 1))
    key = storage.as_strided((pages, 32, 1, 128), (4160, 128, 128, 1))
    scale = storage.view(torch.float16).as_strided((pages, 32, 1, 1), (2080, 1, 1, 1), 2048)
    views = [("inner_compress_state", physical_pages(state), DataType.FP32),
             ("idx_kv_cache", indexer_storage(key, scale), DataType.INT8)]
    tensors, descriptors = [], []
    for index, (name, tensor, dtype) in enumerate(views):
        owner = tensor.untyped_storage()
        meta = TensorMetadata(name, index, ParamDirection.InOut, dtype,
                              tuple(tensor.shape), tuple(tensor.stride()), 0, 2,
                              tensor.data_ptr(), owner.data_ptr(), tensor.storage_offset(),
                              owner.nbytes(), tensor.numel() * tensor.element_size())
        tensors.append(TensorArgument(meta, tensor, owner))
        descriptors.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "pointer": meta.data_ptr, "bytes": meta.nbytes})
    assert tensors[0].metadata.data_ptr == tensors[1].metadata.data_ptr
    assert tensors[0].metadata.nbytes == tensors[1].metadata.nbytes
    error = None
    try:
        _validate_aliases(tensors)
    except ValueError as exc:
        error = str(exc)
        assert "partially overlaps" in error
    assert (error is None) == args.expect_accepted, f"别名检查结果不符：{error}"

    # 移动起始地址、缩短范围，构造真正的部分重叠；修复后仍必须拒绝。
    partial = replace(tensors[1], metadata=replace(
        tensors[1].metadata, data_ptr=tensors[1].metadata.data_ptr + 4,
        nbytes=tensors[1].metadata.nbytes - 4))
    try:
        _validate_aliases([tensors[0], partial])
    except ValueError as exc:
        assert "partially overlaps" in str(exc)
    else:
        raise AssertionError("真正的可写部分重叠未被拒绝")

    ranks, layers = 0, 0
    if args.layout_dir:
        assert args.expect_accepted, "真实描述符回归要求修复后的版本"
        dtype_map = {"torch.float32": DataType.FP32, "torch.bfloat16": DataType.BF16,
                     "torch.int8": DataType.INT8}
        files = sorted(args.layout_dir.glob("rank*.cache_layout.json"))
        assert files, "没有找到真实缓存描述符"
        for path in files:
            for worker in json.loads(path.read_text()):
                ranks += 1
                for layer in worker["layers"].values():
                    captured = []
                    for index, (name, view) in enumerate(layer["views"].items()):
                        meta = TensorMetadata(
                            name, index, ParamDirection.InOut, dtype_map[view["dtype"]],
                            tuple(view["shape"]), tuple(view["stride"]), 0, 2,
                            view["pointer"], view["storage_pointer"], view["storage_offset"],
                            view["storage_bytes"], view["view_bytes"])
                        # 检查只读取metadata；CPU占位对象不访问已结束任务的设备地址。
                        captured.append(TensorArgument(meta, storage, storage.untyped_storage()))
                    _validate_aliases(captured)
                    layers += 1
    write_json(args.output, {
        "status": "PASS" if args.expect_accepted else "REPRODUCED",
        "scope": "CPU描述符检查，无设备执行或精度结论",
        "same_byte_range": True, "views": descriptors, "error": error,
        "partial_overlap_rejected": True, "captured_ranks": ranks, "captured_layers": layers,
    })


if __name__ == "__main__":
    main()
