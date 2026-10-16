// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"

// PTO's generated MIX code uses a group index shared by one AIC and two AIVs.
// AscendC exposes a linear AIV index; SuperKernel may remap that logical index.
__aicore__ inline uint32_t HcPreGroupIdx() { return AscendC::GetBlockIdx() / AscendC::GetTaskRatio(); }
__aicore__ inline uint32_t HcPreGroupCount() { return AscendC::GetBlockNum(); }

// The pinned PTO umbrella header also contains unused helpers with raw topology
// intrinsics. Adapt them at the include boundary, without changing PTO itself.
#define get_block_idx HcPreGroupIdx
#define get_block_num HcPreGroupCount
#include "pto/pto-inst.hpp"
#undef get_block_num
#undef get_block_idx
