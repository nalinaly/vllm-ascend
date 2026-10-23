// SPDX-License-Identifier: Apache-2.0
#include "../include/ascendc_elementwise.h"
extern "C" __global__ __aicore__ void hc_pre_mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(data, tiling);
  // TPipe construction initializes shared device state; keep it after the wait.
  AscendC::WaitPreTaskEnd();
  HcPreElementwise<true> op;
  op.Run(x, y, z, data.elements);
}
