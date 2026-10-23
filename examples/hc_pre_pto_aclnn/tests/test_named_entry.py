# SPDX-License-Identifier: Apache-2.0
"""Check the packaged Python entry, Torch schema and representative ACLNN paths."""
import json

import bytenn_pto_ops
import torch
import torch_npu  # noqa: F401
from check_outputs import reference
from test_torch import compare


def main():
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.manual_seed(20261019)
    results = []
    for tokens in (1, 48, 96, 168):
        host = (
            torch.randn(tokens, 4, 4096, dtype=torch.bfloat16),
            torch.randn(24, 16384) * 0.01,
            torch.tensor([0.8, 1.0, 1.2]),
            torch.randn(24) * 0.1,
        )
        inputs = tuple(t.npu() for t in host)
        actual = bytenn_pto_ops.hc_pre_pto_bnn(*inputs)
        torch.npu.synchronize()
        checked = compare(actual, reference(*host))
        meta = torch.ops.bytenn_pto_ops.hc_pre_pto_bnn(*(t.to("meta") for t in host))
        assert all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(actual, meta))
        results.append({"tokens": tokens, "outputs": checked, "meta_passed": True})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
