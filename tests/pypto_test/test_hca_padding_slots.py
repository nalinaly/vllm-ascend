"""只在CPU回归空压缩slot的不同容量与越界有效行检查，不依赖PyPTO。"""

import torch
from dsv4_hca_padding import compare_empty_compressed_slots


def test_empty_slots_allow_different_capacities():
    actual = torch.tensor([[-1, 127]], dtype=torch.int32).repeat(16, 1)
    reference = actual[:6].clone()
    checks = compare_empty_compressed_slots(actual, reference, 128)
    assert all(item["status"] == "PASS" for item in checks.values())


def test_empty_slots_reject_valid_entry_beyond_oracle_capacity():
    actual = torch.tensor([[-1, 127]], dtype=torch.int32).repeat(16, 1)
    reference = actual[:6].clone()
    actual[15] = torch.tensor([3, 0], dtype=torch.int32)
    checks = compare_empty_compressed_slots(actual, reference, 128)
    assert checks["slots"]["status"] == "FAIL"
    assert checks["oracle_slots"]["status"] == "PASS"


def test_empty_slots_reject_wrong_invalid_offset():
    actual = torch.tensor([[-1, 127]], dtype=torch.int32).repeat(16, 1)
    reference = actual[:6].clone()
    reference[0, 1] = 0
    checks = compare_empty_compressed_slots(actual, reference, 128)
    assert checks["slots"]["status"] == "PASS"
    assert checks["oracle_slots"]["status"] == "FAIL"
