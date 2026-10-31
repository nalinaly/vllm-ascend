# SPDX-License-Identifier: Apache-2.0
"""单卡验证三段双流Event复用；连续入队且中途不做Host同步。"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from offline_pd.event_mode import get_event_work_mode, set_event_work_mode


def exercise(reuse, rounds):
    main = torch.npu.current_stream()
    aux = torch.npu.Stream()
    events = tuple(torch.npu.Event() for _ in range(6)) if reuse else (None,) * 6
    # Both streams update the same buffer, so any missed generation can change
    # the integer answer. Separate output rows retain every round's result.
    values = torch.zeros(4096, dtype=torch.int32, device="npu")
    outputs = torch.empty((rounds, 4096), dtype=torch.int32, device="npu")
    for iteration in range(rounds):
        for stage in range(3):
            values.add_(1)
            begin = main.record_event(events[2 * stage])
            with torch.npu.stream(aux):
                aux.wait_event(begin)
                values.add_(2)
            main.wait_event(aux.record_event(events[2 * stage + 1]))
        outputs[iteration].copy_(values)
    # A single terminal synchronization observes all queued generations.
    torch.npu.synchronize()
    expected = (torch.arange(1, rounds + 1, dtype=torch.int32) * 9).unsqueeze(1).expand(-1, 4096)
    assert torch.equal(outputs.cpu(), expected)
    return {"reuse": reuse, "rounds": rounds, "event_objects": 6 if reuse else 6 * rounds,
            "record_wait_pairs": 6 * rounds, "all_outputs_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    set_event_work_mode(0)
    result = {"device": args.device, "event_work_mode": get_event_work_mode(),
              "checks": [exercise(False, 64), exercise(True, 64)],
              "scope": "仅验证Event代际与流依赖，不代表CSA/模型性能；不增加循环内Host同步。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
