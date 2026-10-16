// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "../include/contract.h"
#include <initializer_list>

namespace optiling {
BEGIN_TILING_DATA_DEF(HcPrePtoTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tokens);
TILING_DATA_FIELD_DEF(uint32_t, coreGroups);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HcPrePto, HcPrePtoTilingData)
}  // namespace optiling

namespace {
bool ShapeIs(const gert::StorageShape* shape, std::initializer_list<int64_t> dims) {
  if (shape == nullptr || shape->GetStorageShape().GetDimNum() != dims.size()) return false;
  size_t i = 0;
  for (auto d : dims)
    if (shape->GetStorageShape().GetDim(i++) != d) return false;
  return true;
}
ge::graphStatus Tiling(gert::TilingContext* c) {
  const auto* xs = c->GetInputShape(0);
  if (!xs || xs->GetStorageShape().GetDimNum() != 3) return ge::GRAPH_FAILED;
  const auto t = xs->GetStorageShape().GetDim(0);
  if (t <= 0 || t > 1048576) return ge::GRAPH_FAILED;
  if (!ShapeIs(xs, {t, 4, 4096}) || !ShapeIs(c->GetInputShape(1), {24, 16384}) || !ShapeIs(c->GetInputShape(2), {3}) ||
      !ShapeIs(c->GetInputShape(3), {24}) || !ShapeIs(c->GetOutputShape(0), {t, 4096}) ||
      !ShapeIs(c->GetOutputShape(1), {t, 4}) || !ShapeIs(c->GetOutputShape(2), {t, 4, 4}) ||
      c->GetPlatformInfo() == nullptr)
    return ge::GRAPH_FAILED;
  platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
  const auto coreGroups = platform.GetCoreNumAic();
  if (coreGroups == 0 || platform.GetCoreNumAiv() != 2 * coreGroups) return ge::GRAPH_FAILED;
  const auto cfg = hcpre::MakeConfig(t, coreGroups);
  c->SetBlockDim(coreGroups);
  c->SetTilingKey(cfg.key);
  c->GetWorkspaceSizes(1)[0] = cfg.workspaceBytes + platform.GetLibApiWorkSpaceSize();
  optiling::HcPrePtoTilingData data;
  data.set_tokens(t);
  data.set_coreGroups(coreGroups);
  data.SaveToBuffer(c->GetRawTilingData()->GetData(), c->GetRawTilingData()->GetCapacity());
  c->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferShape(gert::InferShapeContext* c) {
  const auto* xs = c->GetInputShape(0);
  if (!xs || xs->GetDimNum() != 3) return ge::GRAPH_FAILED;
  const auto t = xs->GetDim(0);
  *c->GetOutputShape(0) = gert::Shape({t, 4096});
  *c->GetOutputShape(1) = gert::Shape({t, 4});
  *c->GetOutputShape(2) = gert::Shape({t, 4, 4});
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferType(gert::InferDataTypeContext* c) {
  c->SetOutputDataType(0, ge::DT_BF16);
  c->SetOutputDataType(1, ge::DT_FLOAT);
  c->SetOutputDataType(2, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
}  // namespace
namespace ops {
class HcPrePto : public OpDef {
 public:
  explicit HcPrePto(const char* name) : OpDef(name) {
    Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    for (const char* key : {"weight", "scale", "base"})
      Input(key)
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND});
    Output("mixed")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    for (const char* key : {"post", "comb"})
      Output(key)
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND});
    SetInferShape(InferShape).SetInferDataType(InferType);
    AICore().SetTiling(Tiling).AddConfig("ascend910_93");
  }
};
OP_ADD(HcPrePto);
}  // namespace ops
