// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HcPreElementwiseTilingData)
TILING_DATA_FIELD_DEF(uint32_t, elements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HcPreAdd, HcPreElementwiseTilingData)
REGISTER_TILING_DATA_CLASS(HcPreMul, HcPreElementwiseTilingData)
}  // namespace optiling

namespace {
ge::graphStatus TilingElementwise(gert::TilingContext* c) {
  const auto& x = c->GetInputShape(0)->GetStorageShape();
  const auto& y = c->GetInputShape(1)->GetStorageShape();
  const auto elements = x.GetShapeSize();
  // This example needs same-shape BF16 operands with aligned, bounded sizes.
  if (x != y || elements <= 0 || elements % 16 != 0 || elements > 0x7fffffff) return ge::GRAPH_FAILED;
  platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
  c->SetBlockDim(platform.GetCoreNumAiv());
  c->SetTilingKey(0);
  c->GetWorkspaceSizes(1)[0] = 0;
  optiling::HcPreElementwiseTilingData data;
  data.set_elements(elements);
  data.SaveToBuffer(c->GetRawTilingData()->GetData(), c->GetRawTilingData()->GetCapacity());
  c->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferElementwise(gert::InferShapeContext* c) {
  *c->GetOutputShape(0) = *c->GetInputShape(0);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferElementwiseType(gert::InferDataTypeContext* c) {
  c->SetOutputDataType(0, ge::DT_BF16);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

namespace ops {
// Separate op names make the two native AscendC neighbours visible in SK logs.
#define HCPRE_ELEMENTWISE_OP(OpName)                                          \
  class OpName : public OpDef {                                               \
   public:                                                                    \
    explicit OpName(const char* name) : OpDef(name) {                         \
      Input("x")                                                              \
          .ParamType(REQUIRED)                                                \
          .DataType({ge::DT_BF16})                                            \
          .Format({ge::FORMAT_ND})                                            \
          .UnknownShapeFormat({ge::FORMAT_ND});                               \
      Input("y")                                                              \
          .ParamType(REQUIRED)                                                \
          .DataType({ge::DT_BF16})                                            \
          .Format({ge::FORMAT_ND})                                            \
          .UnknownShapeFormat({ge::FORMAT_ND});                               \
      Output("z")                                                             \
          .ParamType(REQUIRED)                                                \
          .DataType({ge::DT_BF16})                                            \
          .Format({ge::FORMAT_ND})                                            \
          .UnknownShapeFormat({ge::FORMAT_ND});                               \
      SetInferShape(InferElementwise).SetInferDataType(InferElementwiseType); \
      AICore().SetTiling(TilingElementwise).AddConfig("ascend910_93");        \
    }                                                                         \
  };                                                                          \
  OP_ADD(OpName);
HCPRE_ELEMENTWISE_OP(HcPreAdd)
HCPRE_ELEMENTWISE_OP(HcPreMul)
#undef HCPRE_ELEMENTWISE_OP
}  // namespace ops
