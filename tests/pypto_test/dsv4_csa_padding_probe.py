"""S0：确认补位请求的陈旧 position 会让 PTO 的哪些索引越界。

这是 padding 开发计划（DSV4_FLASH_CSA_PADDING_PLAN.md）的第一步，只取证据，
不修改任何生产代码，也不发射 PTO kernel。

PTO 的 device 代码不能抛异常，越界读只会读到无关数据，所以无法靠"改成报错"取证。
改为把 kernel 里的四处索引公式在 CPU 上原样复算，再和张量的真实边界比较：

  A. state_table 列          decode_compressor_ratio4.py:209
  B. cmp_slot_mapping 行     decode_compressor_ratio4.py:390
  C. idx_slot_mapping 行     decode_indexer_compressor.py:489 与 :522
  D. ori_block_table 列      decode_sparse_attn_csa.py:216

两种模式：

  predict  纯 CPU。按 Native padding 协议构造补位批次，复算四处索引并报告量级。
           不需要设备，也不需要队列。
  capture  需要一张 NPU。用 Native 真实 metadata builder 造出补位后的 device
           张量，取回 CPU 后用同一套公式复算，并和张量真实形状比较。

常量一律从生产模块导入，避免和 kernel 漂移。
"""
import argparse
import json
import traceback
from pathlib import Path

from dsv4_csa_env import activate, write_json


def load_constants():
    """从生产模块取常量，不在本文件里重复定义。"""
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark import decode_compressor_ratio4 as compressor
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark import decode_indexer_compressor as indexer_compressor
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark import decode_sparse_attn_csa as sparse

    return {
        "S": compressor.S,
        "COMPRESS_RATIO": compressor.COMPRESS_RATIO,
        "STATE_LEN": compressor.STATE_LEN,
        "COMPRESS_STATE_BLOCK_SIZE": compressor.COMPRESS_STATE_BLOCK_SIZE,
        "BOUNDARY_ROWS_PER_REQUEST": indexer_compressor.BOUNDARY_ROWS_PER_REQUEST,
        "WIN": sparse.WIN,
        "BLOCK_SIZE": sparse.BLOCK_SIZE,
        "SWA_RUNS": sparse.SWA_RUNS,
    }


def compact_row_offsets(bounds, lengths, ratio, *, trunc):
    """复刻 compact_metadata.build_compact_row_offsets。

    PTO 的整数除法对负数的取整方向决定补位请求的 offset，而补位请求的
    ``start`` 必为负（length=0、end-begin=S）。两种取整都算一遍，差异写进报告，
    不假定其中一种。
    """
    def divide(value, divisor):
        if trunc:
            return int(value / divisor) if value < 0 else value // divisor
        return value // divisor

    offsets, prefix = [], 0
    for request in range(len(lengths)):
        begin, end = bounds[request], bounds[request + 1]
        length = lengths[request]
        start = length - (end - begin)
        offsets.append(prefix - divide(start, ratio) - 1)
        prefix = prefix + divide(length, ratio) - divide(start, ratio)
    return offsets


def site_state_table_column(positions, real_reqs, const):
    """A：decode_compressor_ratio4.py:209 读 state_table[c_idx, logical_pos // 2]。"""
    seq, ratio, state_len = const["S"], const["COMPRESS_RATIO"], const["STATE_LEN"]
    block = const["COMPRESS_STATE_BLOCK_SIZE"]
    rows = []
    for request in range(len(positions) // seq):
        first_pos = positions[request * seq]
        for step in range(seq):
            token_pos = positions[request * seq + step]
            if (token_pos + 1) % ratio:
                continue
            window_start = token_pos - state_len + 1
            for state_idx in range(state_len):
                logical_pos = window_start + state_idx
                # kernel 的保护条件，原样照抄
                if not (0 <= logical_pos < first_pos):
                    continue
                rows.append({"request": request, "padded": request >= real_reqs,
                             "column": logical_pos // block})
    return rows


def site_compact_row(positions, offsets, real_reqs, const, *, name):
    """B/C：按 compact_offsets + (position+1)//ratio 定位 compact metadata 行。"""
    seq, ratio = const["S"], const["COMPRESS_RATIO"]
    rows = []
    for request in range(len(positions) // seq):
        for step in range(seq):
            token = request * seq + step
            token_pos = positions[token]
            if (token_pos + 1) % ratio:
                continue
            rows.append({"request": request, "padded": request >= real_reqs, "site": name,
                         "row": offsets[request] + (token_pos + 1) // ratio})
    return rows


def site_block_table_column(positions, real_reqs, const):
    """D：decode_sparse_attn_csa.py:216 读 ori_block_table[request, (v_start+v_lo)//32]。"""
    seq, win, block, runs = const["S"], const["WIN"], const["BLOCK_SIZE"], const["SWA_RUNS"]
    rows = []
    for token, position in enumerate(positions):
        request = token // seq
        length = min(position + 1, win)
        start = position - length + 1
        head = start % block
        for run in range(runs):
            low = max(run * block - head, 0)
            high = min((run + 1) * block - head, length)
            if high > low:
                rows.append({"request": request, "padded": request >= real_reqs,
                             "column": (start + low) // block})
    return rows


def summarize(rows, key, bound=None):
    """按真实／补位分组给出索引范围，有边界时判定是否越界。"""
    result = {}
    for padded in (False, True):
        values = [row[key] for row in rows if row["padded"] is padded]
        group = {"count": len(values)}
        if values:
            group.update(min=min(values), max=max(values))
            if bound is not None:
                out = [value for value in values if not 0 <= value < bound]
                group.update(bound=bound, out_of_range=len(out),
                             verdict="OUT_OF_RANGE" if out else "IN_RANGE")
        result["padded" if padded else "real"] = group
    return result


def build_positions(real_reqs, padded_reqs, history, stale, const):
    """真实请求用连续 position；补位段按 Native 协议保留上一步的陈旧值。"""
    seq = const["S"]
    positions = []
    for _ in range(real_reqs):
        positions.extend(history + step for step in range(seq))
    for _ in range(padded_reqs - real_reqs):
        # 上一步残留：runner 只写 [:total_num_scheduled_tokens]，补位段不重置。
        positions.extend(stale + step for step in range(seq))
    return positions


def predict(args, const):
    seq, ratio = const["S"], const["COMPRESS_RATIO"]
    real_tokens, padded_tokens = args.real_batch * seq, args.padded_batch * seq
    positions = build_positions(args.real_batch, args.padded_batch, args.history, args.stale_position, const)

    # Native 协议：补位请求 seq_lens=0；query_start_loc 按均匀 S 步长补齐。
    bounds = [index * seq for index in range(args.padded_batch + 1)]
    lengths = [args.history + seq] * args.real_batch + [0] * (args.padded_batch - args.real_batch)

    # 真实 compact 行数，来自 dsa_v1.py:606 _num_compressor_metadata_rows。
    compact_rows = min(padded_tokens, padded_tokens // ratio + args.padded_batch)

    report = {"mode": "predict", "constants": const,
              "scenario": {"real_batch": args.real_batch, "padded_batch": args.padded_batch,
                           "history": args.history, "stale_position": args.stale_position,
                           "real_tokens": real_tokens, "padded_tokens": padded_tokens,
                           "compact_metadata_rows": compact_rows},
              "compact_offsets": {}, "sites": {}}

    for trunc in (False, True):
        label = "trunc" if trunc else "floor"
        offsets = compact_row_offsets(bounds, lengths, ratio, trunc=trunc)
        report["compact_offsets"][label] = offsets
        for name, tensor in (("cmp_slot_mapping", "compressed"), ("idx_slot_mapping", "indexer")):
            rows = site_compact_row(positions, offsets, args.real_batch, const, name=name)
            report["sites"][f"{name}_row.{label}"] = summarize(rows, "row", bound=compact_rows)
        del tensor

    # 页表列数按 fixture 的分配公式推出；真实 runner 按 max_model_len 一次性分配，
    # 通常更宽，所以这里给出的是偏保守的边界。
    capacity = args.capacity_history if args.capacity_history is not None else args.history
    state_columns = (capacity + seq + const["COMPRESS_STATE_BLOCK_SIZE"] - 1) // const["COMPRESS_STATE_BLOCK_SIZE"] + 2
    block_columns = (capacity + seq + const["BLOCK_SIZE"] - 1) // const["BLOCK_SIZE"] + 2
    report["scenario"]["capacity_history"] = capacity
    report["scenario"]["state_table_columns"] = state_columns
    report["scenario"]["ori_block_table_columns"] = block_columns

    report["sites"]["state_table_column"] = summarize(
        site_state_table_column(positions, args.real_batch, const), "column", bound=state_columns)
    report["sites"]["ori_block_table_column"] = summarize(
        site_block_table_column(positions, args.real_batch, const), "column", bound=block_columns)

    # 真实运行里每个 batch 会被补到哪一档，纯 CPU 可算，作为后续用例的输入依据。
    report["padding_map"] = padding_map(args.capture_sizes, seq, args.max_batch)
    return report


def padding_map(capture_sizes, seq, max_batch):
    """每个真实 batch 落到哪个捕获档位，以及补出多少请求。"""
    sizes = sorted(capture_sizes)
    result = []
    for batch in range(1, max_batch + 1):
        tokens = batch * seq
        chosen = next((size for size in sizes if size >= tokens), None)
        if chosen is None:
            continue
        result.append({"batch": batch, "tokens": tokens, "padded_tokens": chosen,
                       "padded_batch": chosen // seq, "padded_reqs": chosen // seq - batch})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["predict"], default="predict",
                        help="predict 为纯 CPU 复算；capture 模式待补，需要一张 NPU")
    parser.add_argument("--real-batch", type=int, default=3, help="真实请求数")
    parser.add_argument("--padded-batch", type=int, default=4, help="补齐后的请求数")
    parser.add_argument("--history", type=int, default=131071, help="真实请求的历史长度")
    parser.add_argument("--stale-position", type=int, default=131071,
                        help="补位 token 的陈旧 position，模拟上一步残留值")
    parser.add_argument("--capacity-history", type=int, default=None,
                        help="页表按这个历史长度分配列数；默认等于 --history")
    parser.add_argument("--max-batch", type=int, default=40, help="padding_map 覆盖到的最大 batch")
    parser.add_argument("--capture-sizes", type=int, nargs="+",
                        default=[6, 12, 18, 24, 36, 42, 48, 60, 66, 72, 84, 90, 96, 108, 114, 120,
                                 132, 138, 144, 156, 162, 168, 180, 186, 192, 204, 210, 216, 228, 234, 240],
                        help="实际捕获档位，默认取 native_dp_v1 记录的 DP2 结果")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    activate()
    report = {"status": "FAIL"}
    try:
        const = load_constants()
        report = predict(args, const)
        report["status"] = "DONE"
        print(json.dumps(report["sites"], indent=2, ensure_ascii=False), flush=True)
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / "padding_probe.json", report)


if __name__ == "__main__":
    main()
