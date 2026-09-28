# SPDX-License-Identifier: Apache-2.0
"""HCA 整层的同一张图"满档 → 补位 → 满档"检查，覆盖动态 BS 切换与 padding/dummy。

与 CSA 的 `dsv4_csa_single_layer.check_padding_graph` 同构，按 HCA 改写三处：
组为 swa / compressed / state（HCA 没有 Indexer），输出只有一个 `output`，
compact metadata 只有 compressed 一组。

判据：有效请求的输出必须与**满档 eager 结果的前 active*6 行逐 bit 相同**——
补位请求的存在不允许改变真实请求的任何一位；补位行对应的 cache/state 保持初态；
写保护只落在有效 slot 内；捕获的 compact metadata 在重放后与 Native oracle 的有效行一致。
"""

from dsv4_csa_single_layer import guard_checks, restore, writable_bytes
from dsv4_csa_validation import compare_tensor


def check_padding_graph(fixture, output, eager, make_call, impl, report):
    """fixture 为满档；eager 是同一实现在满档下的结果（含 output 与三份 allocation）。

    `make_call(compact)` 要返回一个可调用对象：它用给定的 compact metadata 跑一次整层。
    compact 的生产者必须落在捕获区内，否则重放时用的是捕获时刻的旧值，补位测试不成立。
    """
    import torch

    batch = fixture["tokens"] // 6
    groups = fixture["groups"]
    originals = {
        name: {
            "slots": group["common"].slot_mapping.clone(),
            "table": group["common"].block_table_tensor.clone(),
            "lengths": group["common"]._seq_lens_cpu.clone(),
            "allowed": group["allowed"],
        }
        for name, group in groups.items()
    }
    original_readonly = fixture["readonly"]
    result = {
        "status": "RUNNING", "bucket_batch": batch,
        "scope": "单卡 PTO 同一图的满档→补位→满档；Native builder 在图外更新、"
                 "Native compact producer 在图内执行；不代表整机或空 rank 验收",
        "padding": "seq_lens=0、slot=-1、页表=0；positions 与尾部 RoPE 保留旧值",
        "replays": [],
    }
    report["padding_graph"] = result
    captured = {}

    def run():
        # 与生产图一样：compact metadata 的生产者在捕获区内，重放时按更新后的 metadata 重算。
        compact = impl._compute_compressor_metadata(fixture["metadata"][groups["compressed"]["prefix"]].decode)
        captured["compressed"] = compact
        make_call(compact)()

    def update_metadata(active):
        common_cache, prefill_cache, decode_cache, metadata = {}, {}, {}, {}
        for name, group in groups.items():
            common, original = group["common"], originals[name]
            common._seq_lens_cpu.copy_(original["lengths"])
            common._seq_lens_cpu[active:].zero_()
            common.seq_lens.copy_(common._seq_lens_cpu)
            common.num_actual_tokens = active * 6
            common.slot_mapping.copy_(original["slots"])
            common.slot_mapping[active * 6:].fill_(-1)
            common.block_table_tensor.copy_(original["table"])
            common.block_table_tensor[active:].zero_()
            current = group["builder"].build(
                0, common, num_reqs_actual=active, block_size=group["spec"].block_size,
                common_ratio_to_sas_metadata=common_cache,
                prefill_ratio_to_sas_metadata=prefill_cache,
                decode_ratio_to_sas_metadata=decode_cache,
            )
            old = fixture["metadata"][group["prefix"]].decode
            for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping", "start_pos"):
                before, after = getattr(old, field), getattr(current.decode, field)
                if before is not None and (after is None or before.data_ptr() != after.data_ptr()):
                    raise ValueError(f"Native builder 替换了捕获输入地址：{name}.{field}")
            metadata[name] = current
        return metadata

    try:
        restore(fixture)
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        # 满档 → 满档−1 → 1 → 满档：最后一档回到满档，验证补位过后能恢复。
        counts = [batch, *dict.fromkeys((batch - 1, 1)), batch]
        for active in counts:
            metadata = update_metadata(active)
            oracle = {"compressed": impl._compute_compressor_metadata(metadata["compressed"].decode)}
            expected = {"output": eager["output"][:active * 6]}
            compact_checks = {}
            for name, group in groups.items():
                slots = oracle[name][2] if name in oracle else metadata[name].decode.slot_mapping
                group["allowed"] = writable_bytes(group["allocation"], group["views"], slots)
                # 期望的整份 allocation：初态，只在有效 slot 覆盖的字节上换成满档结果。
                # 这一条同时表达了两件事——补位请求一个字节都不许写，
                # 有效请求写出的内容必须与满档时逐 bit 一致。
                allocation = group["initial"].clone()
                allocation[group["allowed"]] = eager[name][group["allowed"]]
                expected[name] = allocation
            fixture["readonly"] = {name: (value, value.cpu()) for name, (value, _) in original_readonly.items()}
            restore(fixture)
            output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            actual = {"output": output.cpu()[:active * 6]}
            for name, group in groups.items():
                actual[name] = group["allocation"].cpu()
            for name, values in oracle.items():
                valid = (values[2].cpu() >= 0).all(dim=1)
                rows = valid.nonzero().flatten()
                for field, value, reference in zip(("cos", "sin", "slots"), captured[name], values):
                    compact_checks[f"{name}.{field}"] = compare_tensor(
                        value.cpu()[rows], reference.cpu()[rows], 0, 0)
            checks = {key: compare_tensor(actual[key], value, 0, 0) for key, value in expected.items()}
            guards = guard_checks(fixture)
            result["replays"].append({
                "active_batch": active, "reference": "同一实现满档有效请求前缀；补位 cache/state 保持初态",
                "state_comparison": checks, "compact_metadata": compact_checks, "guards": guards,
            })
            all_checks = (*checks.values(), *compact_checks.values(), *guards.values())
            if any(value["status"] != "PASS" for value in all_checks):
                raise ValueError(f"有效请求数 {active}/{batch} 的图重放输出、metadata 或保护区失败")
        result["status"] = "PASS"
    except BaseException:
        result["status"] = "FAIL"
        raise
    finally:
        update_metadata(batch)
        for name, group in groups.items():
            group["allowed"] = originals[name]["allowed"]
        fixture["readonly"] = original_readonly
        restore(fixture)
