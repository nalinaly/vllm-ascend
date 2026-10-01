# SPDX-License-Identifier: Apache-2.0
"""同步文件 cache 不应走异步 admission；使用真实 CPU scheduler 复现容量死锁。

运行时设置 VLLM_PLUGINS=''，不加载模型权重、NPU 或 PyPTO。
"""

import json

import pytest
import torch
import vllm.platforms
from offline_pd.connector import OfflineDSV4Connector
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig
from vllm.platforms.cpu import CpuPlatform
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


@pytest.fixture
def scheduler_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(vllm.platforms, "_current_platform", CpuPlatform())
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 32,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "intermediate_size": 64,
                "vocab_size": 64,
                "max_position_embeddings": 128,
            }
        )
    )
    bank = tmp_path / "bank"
    (bank / "q/tp0").mkdir(parents=True)
    (bank / "q/tp0/manifest.json").write_text(json.dumps({"history": 31}))
    (bank / "q.json").write_text(json.dumps([1] * 32))

    def create(legacy_async=False):
        config = VllmConfig(
            model_config=ModelConfig(
                model=str(model_path), skip_tokenizer_init=True, dtype="float32", max_model_len=128
            ),
            cache_config=CacheConfig(block_size=32, enable_prefix_caching=False),
            parallel_config=ParallelConfig(),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2,
                max_num_batched_tokens=32,
                max_model_len=128,
                async_scheduling=True,
                is_encoder_decoder=False,
            ),
            speculative_config=SpeculativeConfig(method="ngram_gpu", num_speculative_tokens=5),
        )
        # 两个可用块：足够放两个31-token前缀，却不够两个请求继续草稿生成。
        config.cache_config.num_gpu_blocks = 3  # 另有一个 null block。
        cache = KVCacheConfig(
            num_blocks=3,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"], FullAttentionSpec(block_size=32, num_kv_heads=1, head_size=1, dtype=torch.float32)
                ),
            ],
        )
        register_all_kvcache_specs(config)
        scheduler = AsyncScheduler(
            vllm_config=config,
            kv_cache_config=cache,
            block_size=32,
            structured_output_manager=StructuredOutputManager(config),
            log_stats=False,
        )
        # 只测试 admission；不构建 DSpark 模型，采用其实际的5个lookahead槽位。
        scheduler.num_lookahead_tokens = 5
        connector = object.__new__(OfflineDSV4Connector)
        connector.root, connector.tp = bank, 0
        connector.producer, connector.save_root = False, None
        connector.cases = {"q": {"history": 31, "tokens": "q.json"}}
        connector.requests, connector.request_blocks = {}, {}
        connector.pending_loads, connector.saved = [], set()
        if legacy_async:
            original = connector.get_num_new_matched_tokens
            connector.get_num_new_matched_tokens = lambda *args: (original(*args)[0], True)
        scheduler.connector = connector
        for index in range(2):
            scheduler.add_request(
                Request(
                    request_id=str(index),
                    prompt_token_ids=[1] * 32,
                    pooling_params=None,
                    sampling_params=SamplingParams(
                        max_tokens=8, ignore_eos=True, extra_args={"kv_transfer_params": {"offline_key": "q"}}
                    ),
                )
            )
        return scheduler

    return create


def test_legacy_async_admission_stalls_after_successful_load(scheduler_factory):
    scheduler = scheduler_factory(legacy_async=True)
    assert not scheduler.schedule().num_scheduled_tokens
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    # 完成通知没有丢失，仍不能为第一个decode分配lookahead。
    scheduler._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={"0", "1"}))
    for _ in range(3):
        assert not scheduler.schedule().num_scheduled_tokens
    assert not scheduler.running
    assert all(not request.output_token_ids for request in scheduler.requests.values())


def test_sync_admission_keeps_decode_space_and_next_request_progresses(scheduler_factory):
    scheduler = scheduler_factory()
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"0": 1}
    assert scheduler.requests["1"].status == RequestStatus.WAITING
    assert scheduler.schedule().num_scheduled_tokens == {"0": 6}
    assert scheduler.connector.get_finished(set()) == (set(), set())
    # 模拟第一条执行完毕；释放后第二条也能入场，不能以丢请求换取不挂住。
    scheduler.finish_requests("0", RequestStatus.FINISHED_LENGTH_CAPPED)
    assert scheduler.schedule().num_scheduled_tokens == {"1": 1}
