// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"

template <bool Multiply>
class HcPreElementwise {
 public:
  __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t elements) {
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
    zGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(z));
    pipe.InitBuffer(inputs, 2, 2 * TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(outputs, 2, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(scratch, 3 * TILE * sizeof(float));
    for (uint32_t offset = AscendC::GetBlockIdx() * TILE; offset < elements; offset += AscendC::GetBlockNum() * TILE) {
      const uint32_t count = elements - offset < TILE ? elements - offset : TILE;
      auto in = inputs.AllocTensor<bfloat16_t>();
      AscendC::DataCopy(in, xGm[offset], count);
      AscendC::DataCopy(in[TILE], yGm[offset], count);
      inputs.EnQue(in);
      in = inputs.DeQue<bfloat16_t>();
      auto a = scratch.Get<float>();
      auto b = a[TILE];
      auto result = a[2 * TILE];
      auto out = outputs.AllocTensor<bfloat16_t>();
      AscendC::Cast(a, in, AscendC::RoundMode::CAST_NONE, count);
      AscendC::Cast(b, in[TILE], AscendC::RoundMode::CAST_NONE, count);
      AscendC::PipeBarrier<PIPE_V>();
      if constexpr (Multiply)
        AscendC::Mul(result, a, b, count);
      else
        AscendC::Add(result, a, b, count);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(out, result, AscendC::RoundMode::CAST_RINT, count);
      outputs.EnQue(out);
      inputs.FreeTensor(in);
      out = outputs.DeQue<bfloat16_t>();
      AscendC::DataCopy(zGm[offset], out, count);
      outputs.FreeTensor(out);
      // Reuse the FP32 scratch only after this tile's vector work ends.
      AscendC::PipeBarrier<PIPE_V>();
    }
    // Once on every participating core, including cores with no data tile.
    // Keep TPipe cleanup intact; this first adaptation does not remove drains.
    AscendC::SetNextTaskStart();
  }

 private:
  static constexpr uint32_t TILE = 2048;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputs;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outputs;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> scratch;
  AscendC::GlobalTensor<bfloat16_t> xGm, yGm, zGm;
};
