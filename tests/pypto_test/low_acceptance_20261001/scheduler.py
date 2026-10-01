# SPDX-License-Identifier: Apache-2.0
"""仅测试：观察原生异步调度器的真实接受事件，不改变调度或采样。"""
import json
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class AcceptanceScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acceptance_events = {}
        config = args[0] if args else kwargs['vllm_config']
        root = Path(config.additional_config['offline_acceptance_output'])
        self.acceptance_path = root / f'request_stats_rank{self.parallel_config.data_parallel_rank}.jsonl'

    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens,
                                num_accepted_tokens, num_invalid_spec_tokens, request_id):
        result = super().make_spec_decoding_stats(
            spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
            num_invalid_spec_tokens, request_id)
        if self.log_stats and num_draft_tokens:
            proposed = num_draft_tokens - (num_invalid_spec_tokens or {}).get(request_id, 0)
            self.acceptance_events.setdefault(request_id, []).append([proposed, num_accepted_tokens])
        return result

    def _update_request_with_output(self, request, new_token_ids):
        result = super()._update_request_with_output(request, new_token_ids)
        if result[1]:
            events = self.acceptance_events.pop(request.request_id, [])
            record = {'request_id': request.request_id,
                      'key': (request.kv_transfer_params or {}).get('offline_key'),
                      'events': events, 'output_tokens': len(request.output_token_ids),
                      'preemptions': request.num_preemptions}
            # 每个请求结束才写一次CPU记录；不读设备张量，不增加设备同步。
            with self.acceptance_path.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        return result
