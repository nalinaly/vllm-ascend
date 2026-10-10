// SPDX-License-Identifier: Apache-2.0
// 只注册上游两个 HC_pre 接口；NPU 和 Meta 实现直接编译上游源文件。
#include <torch/library.h>
TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def("npu_hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, *, int hc_mult=4, int hc_sinkhorn_iters=20, float norm_eps=1e-6, float hc_eps=1e-6) -> (Tensor, Tensor, Tensor)");
    m.def("npu_hc_pre_v2(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, Tensor? pre_mix=None, *, int hc_mult=4, int hc_sinkhorn_iters=20, float norm_eps=1e-6, float hc_eps=1e-6) -> (Tensor, Tensor, Tensor, Tensor)");
}
