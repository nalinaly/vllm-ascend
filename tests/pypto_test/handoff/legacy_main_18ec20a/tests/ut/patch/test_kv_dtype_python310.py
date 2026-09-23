# SPDX-License-Identifier: Apache-2.0
"""KV dtype patches must support the declared Python 3.10 minimum."""

import ast
import typing
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "relative_path",
    ["platform/patch_indexer_kv_dtype.py", "worker/patch_kv_cache_dtype.py"],
)
def test_dtype_patch_parses_on_python310(relative_path):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/patch" / relative_path
    ast.parse(source.read_text(), filename=str(source), feature_version=(3, 10))


def test_worker_dtype_widening_preserves_members_and_is_idempotent():
    from vllm.config import attention, cache

    from vllm_ascend.patch.worker import patch_kv_cache_dtype as patch

    for module, name, config, field, widen in (
        (cache, "CacheDType", cache.CacheConfig, "cache_dtype", patch._widen_cache_dtype_literal_for_worker),
        (
            attention,
            "IndexerKVDType",
            attention.AttentionConfig,
            "indexer_kv_dtype",
            patch._widen_indexer_kv_dtype_literal_for_worker,
        ),
    ):
        before = typing.get_args(getattr(module, name))
        widen()
        widen()
        after = typing.get_args(getattr(module, name))
        assert set(before) <= set(after)
        assert after.count("int8") == 1
        assert getattr(config(**{field: "int8"}), field) == "int8"
