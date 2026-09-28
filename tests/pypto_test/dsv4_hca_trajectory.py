# SPDX-License-Identifier: Apache-2.0
"""HCA 单卡连续 decode 状态轨迹：同一初态下逐步推进，比较 Native 与 PTO 的状态是否发散。

范围与边界（先说清楚它测什么、不测什么）：

- 测的是**固定归约下 cache/state 轨迹的一致性**。每一步给两侧完全相同的输入
  （同一 seed 生成的 hidden、同一套推进后的 metadata），各自在自己的 cache 上累积，
  逐步记录两侧输出与三份 allocation 的偏差随步数的走势。判据是**偏差不随步数增长**。
- **不测** token 轨迹：真实轨迹里第 k 步的输入取决于第 k−1 步的输出，那需要整模型，
  属交接文档剩余事项第 1 项。本文件给同输入下的状态一致性，不替代整机 token 验收。
- 每步都会做写保护检查，确保该步只写它自己的 slot。

推进一步的含义（S6，每请求每步 6 个 token）：positions += 6、seq_lens += 6，
用 fixture 保存的 BlockTable 重算 slot_mapping，再用 Native builder 原地重建 metadata。
页表按 per_request 个物理页循环映射，因此滑窗与滚动 state 页都会在足够步数后回绕——
这正是本测试要覆盖的部分。
"""

from dsv4_csa_single_layer import guard_checks, writable_bytes
from dsv4_csa_validation import compare_tensor


def _advance(fixture, step, base_lengths, base_positions):
    """把 metadata 推进到第 step 步（step=0 即 fixture 的初始状态）。"""
    import torch

    groups = fixture["groups"]
    positions = fixture["positions"]
    positions.copy_(base_positions + 6 * step)
    metadata = {}
    common_cache, prefill_cache, decode_cache = {}, {}, {}
    for name, group in groups.items():
        common, spec = group["common"], group["spec"]
        ratio = getattr(spec, "compress_ratio", 1)
        common._seq_lens_cpu.copy_(base_lengths + 6 * step)
        common.seq_lens.copy_(common._seq_lens_cpu)
        common.max_seq_len = int(common._seq_lens_cpu.max())
        batch = common.num_reqs
        group["table"].compute_slot_mapping(batch, common.query_start_loc, positions // ratio)
        current = group["builder"].build(
            0, common, num_reqs_actual=batch, block_size=spec.block_size,
            common_ratio_to_sas_metadata=common_cache,
            prefill_ratio_to_sas_metadata=prefill_cache,
            decode_ratio_to_sas_metadata=decode_cache,
        )
        old = fixture["metadata"][group["prefix"]].decode
        for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping", "start_pos"):
            before, after = getattr(old, field), getattr(current.decode, field)
            if before is not None and (after is None or before.data_ptr() != after.data_ptr()):
                raise ValueError(f"Native builder 替换了输入地址：{name}.{field}")
        metadata[name] = current
    return metadata


def check_trajectory(fixture, output, native_call, pto_call, impl, steps, seed, report):
    """两侧各自从同一初态跑 steps 步；每步输入相同，比较输出与累积状态的偏差走势。"""
    import torch

    groups = fixture["groups"]
    base_lengths = groups[next(iter(groups))]["common"]._seq_lens_cpu.clone()
    base_positions = fixture["positions"].clone()
    initial = {name: group["initial"].clone() for name, group in groups.items()}
    original_allowed = {name: group["allowed"] for name, group in groups.items()}
    original_readonly = fixture["readonly"]
    result = {
        "status": "RUNNING", "steps": steps,
        "scope": "单卡同输入连续轨迹；每步两侧输入相同，比较输出与累积 cache/state 偏差随步数走势；"
                 "不是 token 轨迹验收，也不替代整机",
        "criterion": "偏差不随步数增长；每步写保护只落在该步 slot 内",
        "per_step": [],
    }
    report["trajectory"] = result
    try:
        traces = {}
        for side, call in (("native", native_call), ("pto", pto_call)):
            # 回到初态：cache/state 重置，positions 与 seq_lens 由 _advance 每步重设。
            for name, group in groups.items():
                group["allocation"].copy_(initial[name].to(group["allocation"].device))
            per_step = []
            for step in range(steps):
                metadata = _advance(fixture, step, base_lengths, base_positions)
                # 每步的 hidden 由 step 决定，两侧看到完全相同的输入。
                torch.manual_seed(seed + step)
                fixture["hidden"].normal_(0, 1)
                # 推进后重取只读快照：本测试每步都故意改 positions/seq_lens/slot_mapping，
                # 只读检查要验证的是"本步内算子没有改动 metadata"，而不是"metadata 等于第 0 步"。
                fixture["readonly"] = {name: (value, value.cpu())
                                       for name, (value, _) in original_readonly.items()}
                slots = impl._compute_compressor_metadata(metadata["compressed"].decode)[2]
                before = {name: group["allocation"].cpu() for name, group in groups.items()}
                for name, group in groups.items():
                    reference = slots if name == "compressed" else metadata[name].decode.slot_mapping
                    group["allowed"] = writable_bytes(group["allocation"], group["views"], reference)
                    group["initial"] = before[name]
                output.fill_(float("nan"))
                call()
                torch.npu.synchronize()
                snapshot = {"output": output.cpu()}
                snapshot.update({name: group["allocation"].cpu() for name, group in groups.items()})
                guards = guard_checks(fixture)
                bad = [name for name, value in guards.items() if value["status"] != "PASS"]
                if bad:
                    raise ValueError(f"{side} 第 {step} 步写保护失败：{bad}")
                per_step.append({
                    "nonfinite": int((~torch.isfinite(snapshot["output"])).sum()),
                    "snapshot": snapshot,
                    "changed_bytes": {name: guards[name]["changed_bytes"] for name in groups},
                })
            traces[side] = per_step
        for step in range(steps):
            native_step, pto_step = traces["native"][step], traces["pto"][step]
            entry = {"step": step, "nonfinite": {"native": native_step["nonfinite"], "pto": pto_step["nonfinite"]},
                     "changed_bytes": pto_step["changed_bytes"], "divergence": {}}
            for key, value in pto_step["snapshot"].items():
                stats = compare_tensor(value, native_step["snapshot"][key], 0, 0)
                entry["divergence"][key] = {
                    "max_abs": stats.get("max_abs"), "rmse": stats.get("rmse"),
                    "mismatches": stats.get("mismatches"),
                }
            result["per_step"].append(entry)
            if native_step["nonfinite"] or pto_step["nonfinite"]:
                raise ValueError(f"第 {step} 步出现非有限值")
        # 走势判据：输出偏差的最大值不得随步数单调放大到首步的数倍。
        series = [entry["divergence"]["output"]["max_abs"] or 0.0 for entry in result["per_step"]]
        result["output_max_abs_series"] = series
        first = next((value for value in series if value), None)
        result["growth_ratio"] = (max(series) / first) if first else None
        result["status"] = "MEASURED"
    except BaseException:
        result["status"] = "FAIL"
        raise
    finally:
        for name, group in groups.items():
            group["initial"] = initial[name]
            group["allowed"] = original_allowed[name]
            group["allocation"].copy_(initial[name].to(group["allocation"].device))
        fixture["readonly"] = original_readonly
        _advance(fixture, 0, base_lengths, base_positions)
