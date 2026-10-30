# SPDX-License-Identifier: Apache-2.0
"""CSA专用ACLNN描述符：连续ND以当前数据地址和零offset表达，不搬动设备数据。"""


def normalize_nd_descriptors(source: str) -> str:
    # CSA的连续ND合同已在准备阶段核对。Native各层cache/state共享大storage，
    # 若保留storage offset，CANN会为每层重复生成同一份94策略静态二进制。
    # 将(base, offset, storage_size)等价表示为(data_ptr, 0, numel)；NZ保持原样。
    replacements = [
        ("  if (!IsOpInputBaseFormat(at_tensor)) {",
         "  const bool base_format = IsOpInputBaseFormat(at_tensor);\n  if (!base_format) {"),
        ("storageDims.push_back(at_tensor.storage().nbytes() / itemsize);",
         "storageDims.push_back(at_tensor.numel());"),
        ("      at_tensor.strides().data(), at_tensor.storage_offset(), format,\n"
         "      storageDims.data(), storageDims.size(),\n"
         "      const_cast<void *>(at_tensor.storage().data()));",
         "      at_tensor.strides().data(), base_format ? 0 : at_tensor.storage_offset(), format,\n"
         "      storageDims.data(), storageDims.size(),\n"
         "      base_format ? at_tensor.data_ptr() : const_cast<void *>(at_tensor.storage().data()));"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("CSA私有ACLNN描述符模板已变化，需要核对当前ConvertType实现")
        source = source.replace(before, after)
    return source
