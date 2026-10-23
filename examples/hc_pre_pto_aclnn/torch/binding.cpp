// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "op_api_common.h"

namespace {
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
void Check(const at::Tensor& t, at::IntArrayRef shape, at::ScalarType dtype, const at::Device& device,
           const char* name) {
  TORCH_CHECK(t.device() == device, name, ": inputs must share a device");
  TORCH_CHECK(t.sizes() == shape, name, ": unsupported shape");
  TORCH_CHECK(t.scalar_type() == dtype, name, ": unsupported dtype");
  TORCH_CHECK(t.is_contiguous(), name, ": contiguous tensor required");
}
Outputs Allocate(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scale, const at::Tensor& base) {
  TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(0) <= 1048576, "unsupported tokens");
  const auto t = x.size(0);
  Check(x, {t, 4, 4096}, at::kBFloat16, x.device(), "x");
  Check(w, {24, 16384}, at::kFloat, x.device(), "weight");
  Check(scale, {3}, at::kFloat, x.device(), "scale");
  Check(base, {24}, at::kFloat, x.device(), "base");
  return {at::empty({t, 4096}, x.options()), at::empty({t, 4}, w.options()), at::empty({t, 4, 4}, w.options())};
}
Outputs Run(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scale, const at::Tensor& base) {
  const c10_npu::OptionalNPUGuard guard(x.device());
  auto [mixed, post, comb] = Allocate(x, w, scale, base);
  EXEC_NPU_CMD(aclnnHcPrePtoBnn, x, w, scale, base, mixed, post, comb);
  return {mixed, post, comb};
}
at::Tensor AllocateElementwise(const at::Tensor& x, const at::Tensor& y) {
  Check(x, x.sizes(), at::kBFloat16, x.device(), "x");
  Check(y, x.sizes(), at::kBFloat16, x.device(), "y");
  TORCH_CHECK(x.numel() > 0 && x.numel() % 16 == 0 && x.numel() <= 0x7fffffff,
              "elementwise example requires a positive multiple of 16 elements");
  return at::empty(x.sizes(), x.options());
}
at::Tensor RunAdd(const at::Tensor& x, const at::Tensor& y) {
  const c10_npu::OptionalNPUGuard guard(x.device());
  auto z = AllocateElementwise(x, y);
  EXEC_NPU_CMD(aclnnHcPreAdd, x, y, z);
  return z;
}
at::Tensor RunMul(const at::Tensor& x, const at::Tensor& y) {
  const c10_npu::OptionalNPUGuard guard(x.device());
  auto z = AllocateElementwise(x, y);
  EXEC_NPU_CMD(aclnnHcPreMul, x, y, z);
  return z;
}
}  // namespace
TORCH_LIBRARY(bytenn_pto_ops, m) {
  m.def("hc_pre_pto_bnn(Tensor x, Tensor weight, Tensor scale, Tensor base) -> (Tensor, Tensor, Tensor)");
  m.def("ascendc_add(Tensor x, Tensor y) -> Tensor");
  m.def("ascendc_mul(Tensor x, Tensor y) -> Tensor");
}
TORCH_LIBRARY_IMPL(bytenn_pto_ops, PrivateUse1, m) {
  m.impl("hc_pre_pto_bnn", &Run);
  m.impl("ascendc_add", &RunAdd);
  m.impl("ascendc_mul", &RunMul);
}
TORCH_LIBRARY_IMPL(bytenn_pto_ops, Meta, m) {
  m.impl("hc_pre_pto_bnn", &Allocate);
  m.impl("ascendc_add", &AllocateElementwise);
  m.impl("ascendc_mul", &AllocateElementwise);
}
