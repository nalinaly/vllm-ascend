# SPDX-License-Identifier: Apache-2.0
"""HCA 的请求生命周期（页复用）与 prefix 共享检查。

两个阶段各有一个精确判据，都不需要第二份 fixture：

一、生命周期：把请求 0 重置为 history 0，但**保留它原先的物理页**，并把这些页里超出新有效
   范围的行填 NaN。若算子读到任何陈旧行，输出立刻出现非有限值——NaN 自身就是 oracle。
   同时要求其余请求的输出与未污染时逐 bit 相同（页复用不得影响别人）。

二、prefix 共享：先把请求 0 的页**内容**复制进请求 1 的页（两者内容相同、各自独占），
   跑一次记下输出；再把请求 1 的页表改成**指向请求 0 的同一批物理页**，再跑一次，
   要求输出逐 bit 相同。共享只读页不得改变任何结果。

边界：这是单层同输入下的行为检查，不含调度器的真实分配/回收时序，也不替代整机验收。
"""

from dsv4_csa_single_layer import guard_checks, writable_bytes
from dsv4_csa_validation import compare_tensor


def _rebuild(fixture, impl):
    """按当前 common 重建 metadata，并按新 slot 重算写保护与初态。"""
    groups = fixture["groups"]
    common_cache, prefill_cache, decode_cache, metadata = {}, {}, {}, {}
    for name, group in groups.items():
        current = group["builder"].build(
            0, group["common"], num_reqs_actual=group["common"].num_reqs,
            block_size=group["spec"].block_size,
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
    slots = impl._compute_compressor_metadata(metadata["compressed"].decode)[2]
    for name, group in groups.items():
        reference = slots if name == "compressed" else metadata[name].decode.slot_mapping
        group["allowed"] = writable_bytes(group["allocation"], group["views"], reference)
        group["initial"] = group["allocation"].cpu()
    fixture["readonly"] = {name: (value, value.cpu()) for name, (value, _) in fixture["readonly"].items()}
    return metadata


def check_lifecycle(fixture, output, call, native_call, impl, report):
    import torch

    groups = fixture["groups"]
    batch = fixture["tokens"] // 6
    if batch < 2:
        raise ValueError("生命周期与 prefix 共享检查需要至少 2 个请求")
    base_positions = fixture["positions"].clone()
    base_lengths = {name: group["common"]._seq_lens_cpu.clone() for name, group in groups.items()}
    base_tables = {name: group["common"].block_table_tensor.clone() for name, group in groups.items()}
    base_alloc = {name: group["allocation"].cpu().clone() for name, group in groups.items()}
    result = {
        "status": "RUNNING",
        "scope": "单卡同输入下的页复用与 prefix 共享行为；不含调度器真实分配/回收时序，不替代整机",
        "phases": {},
    }
    report["lifecycle"] = result

    def reset():
        fixture["positions"].copy_(base_positions)
        for name, group in groups.items():
            group["common"]._seq_lens_cpu.copy_(base_lengths[name])
            group["common"].seq_lens.copy_(group["common"]._seq_lens_cpu)
            group["common"].block_table_tensor.copy_(base_tables[name])
            group["allocation"].copy_(base_alloc[name].to(group["allocation"].device))

    def run_once_with(fn):
        output.fill_(float("nan"))
        fn()
        torch.npu.synchronize()
        return output.cpu()

    def run_once():
        return run_once_with(call)

    try:
        # 基准：原始 fixture 的一次调用，作为"其余请求不受影响"的参照。
        reset()
        _rebuild(fixture, impl)
        baseline = run_once()

        # ---- 阶段一：请求 0 退出并把页原样交给新请求（history 归零，页不清理） ----
        reset()
        positions = fixture["positions"]
        positions[0:6] = torch.arange(6, device=positions.device, dtype=positions.dtype)
        for name, group in groups.items():
            lengths = group["common"]._seq_lens_cpu
            lengths[0] = 6
            group["common"].seq_lens.copy_(lengths)
            group["common"].max_seq_len = int(lengths.max())
        metadata = _rebuild(fixture, impl)
        # 把请求 0 的页里超出新有效范围的行填 NaN：算子若读陈旧行，输出必然非有限。
        poisoned = 0
        for name, group in groups.items():
            table = metadata[name].decode.block_table.cpu()[0].tolist()
            ratio = getattr(group["spec"], "compress_ratio", 1)
            valid_rows = max(0, (6 + ratio - 1) // ratio)
            for view in group["views"]:
                rows = view.shape[1]
                for column, page in enumerate(table):
                    if not (0 <= page < view.shape[0]):
                        continue
                    first = max(0, valid_rows - column * rows)
                    if first < rows:
                        view[page, first:].fill_(float("nan"))
                        poisoned += rows - first
            group["initial"] = group["allocation"].cpu()
        if not poisoned:
            raise ValueError("没有污染任何行，阶段一无效")
        poisoned_alloc = {name: group["allocation"].cpu().clone() for name, group in groups.items()}
        # 先用 Native 在同一份污染上跑一次：若 Native 也出非有限值，说明生产路径依赖
        # "分配新请求时清页"，那是本测试的前提不成立，不能算 PTO 的缺陷。
        native_actual = run_once_with(native_call)
        native_nonfinite = int((~torch.isfinite(native_actual)).sum())
        for name, group in groups.items():
            group["allocation"].copy_(poisoned_alloc[name].to(group["allocation"].device))
            group["initial"] = group["allocation"].cpu()
        actual = run_once()
        guards = guard_checks(fixture)
        others = compare_tensor(actual[6:], baseline[6:], 0, 0)
        result["phases"]["reuse"] = {
            "poisoned_rows": poisoned,
            "native_output_nonfinite": native_nonfinite,
            "premise": "Native 在同一污染下也非有限则说明生产依赖分配时清页，本阶段判定为前提不成立",
            "reference": "请求 0 的页保留旧内容并把超出有效范围的行填 NaN；"
                         "算子读陈旧行则输出非有限。其余请求要求与基准逐 bit 相同",
            "output_nonfinite": int((~torch.isfinite(actual)).sum()),
            "other_requests": others,
            "guards": guards,
        }
        if int((~torch.isfinite(actual)).sum()) and not native_nonfinite:
            raise ValueError("页复用后 PTO 输出非有限而 Native 有限：PTO 读到了陈旧行")
        if native_nonfinite:
            result["phases"]["reuse"]["verdict"] = (
                "前提不成立：Native 在同一污染下同样非有限，生产路径依赖分配新请求时清页；"
                "本阶段不作为 PTO 判据")
        if others["status"] != "PASS":
            raise ValueError("请求 0 的页复用影响了其他请求的输出")
        if any(value["status"] != "PASS" for value in guards.values()):
            raise ValueError("页复用阶段写保护失败")

        # ---- 阶段二：prefix 共享 ----
        reset()
        for name, group in groups.items():
            table = group["common"].block_table_tensor
            own, shared = table[1].clone(), table[0].clone()
            # 先让两者内容相同、各自独占：把请求 0 的页内容复制进请求 1 的页。
            for view in group["views"]:
                for src, dst in zip(shared.tolist(), own.tolist()):
                    if 0 <= src < view.shape[0] and 0 <= dst < view.shape[0]:
                        view[dst].copy_(view[src])
            group["initial"] = group["allocation"].cpu()
        metadata_private = _rebuild(fixture, impl)
        compact_slots = impl._compute_compressor_metadata(metadata_private["compressed"].decode)[2]
        private = run_once()
        private_guards = guard_checks(fixture)
        # 只共享只读前缀列：当前写入位置所在的那一列必须各自独占，否则两个请求的 slot
        # 会落到同一物理行互相覆盖——那是测试设定错误，不是算子缺陷。
        # 用 slot_mapping 反查"本步实际写入了哪些物理页"，页表里命中这些页的列一律不可共享。
        # 不按几何推算列号：state 的 view 是 [pages, page_elements]，shape[1] 是页内元素数
        # 而不是行数，按它算会把整行都当成可共享，正好复现两请求写同一行的冲突。
        slots_by_group = {
            name: (compact_slots if name == "compressed" else metadata_private[name].decode.slot_mapping)
            for name in groups
        }
        shared_columns = {}
        for name, group in groups.items():
            table = group["common"].block_table_tensor
            written = {int(page) for page, row in slots_by_group[name].cpu().tolist() if page >= 0}
            row0, row1 = table[0].cpu().tolist(), table[1].cpu().tolist()
            blocked = [c for c in range(len(row0)) if row0[c] in written or row1[c] in written]
            shared = min(blocked) if blocked else len(row0)
            shared_columns[name] = shared
            if shared:
                table[1, :shared].copy_(table[0, :shared])
        result["phases"].setdefault("prefix_sharing", {})["shared_columns"] = shared_columns
        if not any(shared_columns.values()):
            raise ValueError("没有可共享的只读前缀列，本阶段无效（需要更长的历史）")
        metadata = _rebuild(fixture, impl)
        sharing = run_once()
        shared_guards = guard_checks(fixture)
        same = compare_tensor(sharing, private, 0, 0)
        result["phases"]["prefix_sharing"].update({
            "reference": "内容相同但各自独占页的一次调用；改为共享同一批物理页后输出须逐 bit 相同",
            "output_nonfinite": int((~torch.isfinite(sharing)).sum()),
            "vs_private_pages": same,
            "private_guards": private_guards,
            "shared_guards": shared_guards,
        })
        if int((~torch.isfinite(sharing)).sum()):
            raise ValueError("prefix 共享后输出出现非有限值")
        if same["status"] != "PASS":
            raise ValueError("prefix 共享改变了输出：存在假定物理页独占的读写路径")
        for label, values in (("private", private_guards), ("shared", shared_guards)):
            if any(value["status"] != "PASS" for value in values.values()):
                raise ValueError(f"prefix 共享阶段 {label} 写保护失败")
        result["status"] = "PASS"
    except BaseException:
        result["status"] = "FAIL"
        raise
    finally:
        reset()
        _rebuild(fixture, impl)
