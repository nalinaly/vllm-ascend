// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include "log/log.h"

// CANN 9.0 predates these diagnostic helpers. opdev/op_log.h also redefines
// OP_LOGE with an error-code argument, so do not defer to that macro at the
// call site. Use the op-common logging and error-reporting APIs directly.
namespace vllm_ascend::cann_compat {
inline void LogInvalidValue(const std::string& opName, const std::string& argument,
                            const std::string& value, const std::string& reason)
{
    if (CheckLogLevel(static_cast<int>(OP), DLOG_ERROR) == 1) {
        DlogRecord(static_cast<int>(OP), DLOG_ERROR, "OpName:[%s] Invalid %s=%s: %s",
                   opName.c_str(), argument.c_str(), value.c_str(), reason.c_str());
    }
    REPORT_INNER_ERR_MSG("EZ9999", "OpName:[%s] Invalid %s=%s: %s",
                         opName.c_str(), argument.c_str(), value.c_str(), reason.c_str());
}
} // namespace vllm_ascend::cann_compat

// Newer SDKs keep their own implementations.
#ifndef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, argument, reason) \
    ::vllm_ascend::cann_compat::LogInvalidValue( \
        Ops::Base::GetOpInfo(opName), argument, "", reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, argument, value, reason) \
    ::vllm_ascend::cann_compat::LogInvalidValue( \
        Ops::Base::GetOpInfo(opName), argument, value, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE
#define OP_LOGE_FOR_INVALID_VALUE(opName, argument, value, expected) \
    ::vllm_ascend::cann_compat::LogInvalidValue( \
        Ops::Base::GetOpInfo(opName), argument, value, "expected " + std::string(expected))
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#endif
#ifndef OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#endif
#ifndef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#endif
#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM
#define OP_LOGE_FOR_INVALID_SHAPEDIM OP_LOGE_FOR_INVALID_VALUE
#endif
