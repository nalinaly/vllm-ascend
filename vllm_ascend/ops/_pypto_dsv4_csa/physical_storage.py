# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Zero-copy physical-storage views for page-strided CSA cache tensors.

PyPTO's outlined AICore child receives a canonical tensor view.  Runtime
strides from a Torch ``as_strided`` cache cannot therefore be relied on for
addressing inside that child.  The adapter prepares one canonical 2-D alias of
the tensor's *physical* storage and passes the original strides as static
specialization metadata.  Kernels can then calculate physical offsets
explicitly without making a contiguous cache mirror.

Preparing an alias creates Torch tensor metadata but does not allocate or copy
device storage.  Callers must prepare and retain this object before ACLGraph
capture; :meth:`PreparedPhysicalStorageAlias.for_launch` only returns the
already-created tensor object.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


class PhysicalStorageLayoutError(ValueError):
    """Raised when a tensor cannot be represented by the page-layout ABI."""


@dataclass(frozen=True, slots=True)
class PhysicalStorageLayout:
    """Immutable address-layout snapshot for one strided tensor binding.

    ``physical_span`` counts elements from the tensor's first logical element
    through its highest-addressed logical element, including any padding
    between rows or pages.  It intentionally excludes storage before
    ``storage_offset`` and unused tail padding after the final logical element.
    """

    name: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    storage_offset: int
    physical_span: int
    storage_numel: int
    logical_numel: int
    element_size: int

    @property
    def end_storage_offset_exclusive(self) -> int:
        """First underlying-storage element not covered by the alias."""
        return self.storage_offset + self.physical_span

    @property
    def padding_elements(self) -> int:
        """Number of inter-row/page elements retained in the physical alias."""
        return self.physical_span - self.logical_numel

    def physical_offset(self, indices: Sequence[int]) -> int:
        """Map logical indices to an offset relative to the alias data pointer."""
        if len(indices) != len(self.shape):
            raise IndexError(f"{self.name}: expected {len(self.shape)} indices, got {len(indices)}")

        offset = 0
        for axis, (index, extent, stride) in enumerate(zip(indices, self.shape, self.strides, strict=True)):
            if not isinstance(index, int):
                raise TypeError(f"{self.name}: index for axis {axis} must be int, got {type(index).__name__}")
            if index < 0 or index >= extent:
                raise IndexError(f"{self.name}: index {index} is outside axis {axis} extent {extent}")
            offset += index * stride
        return offset


@dataclass(frozen=True, slots=True)
class PreparedPhysicalStorageAlias:
    """Capture-stable owner of a source tensor and its zero-copy launch alias.

    Keeping both tensor objects here is deliberate.  The adapter owns this
    object for at least as long as any eager task or captured graph can refer to
    ``launch_tensor``.  No tensor metadata is rebuilt by :meth:`for_launch`.
    """

    source: torch.Tensor
    launch_tensor: torch.Tensor
    layout: PhysicalStorageLayout

    def for_launch(self) -> torch.Tensor:
        """Return the stable ``[1, physical_span]`` tensor prepared for launch."""
        return self.launch_tensor


def _layout_from_tensor(tensor: torch.Tensor, name: str) -> PhysicalStorageLayout:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(tensor).__name__}")
    if tensor.layout is not torch.strided:
        raise PhysicalStorageLayoutError(f"{name}: only torch.strided tensors are supported, got {tensor.layout}")
    if tensor.device.type == "meta":
        raise PhysicalStorageLayoutError(f"{name}: meta tensors have no physical storage")
    if tensor.ndim == 0:
        raise PhysicalStorageLayoutError(f"{name}: scalar tensors have no page layout")

    shape = tuple(int(extent) for extent in tensor.shape)
    strides = tuple(int(stride) for stride in tensor.stride())
    if any(extent <= 0 for extent in shape):
        raise PhysicalStorageLayoutError(f"{name}: every logical extent must be positive, got shape={shape}")
    if any(stride <= 0 for stride in strides):
        raise PhysicalStorageLayoutError(f"{name}: zero or negative strides are unsupported, got strides={strides}")

    # Page layouts are monotonically row-major, but may contain padding at any
    # dimension.  Reject overlapping or transposed views: their offset set
    # cannot be treated as a simple page family by the static CSA kernels.
    inner_span = 1
    for axis in range(len(shape) - 1, -1, -1):
        extent = shape[axis]
        stride = strides[axis]
        if extent == 1:
            continue
        if stride < inner_span:
            raise PhysicalStorageLayoutError(
                f"{name}: strides must describe a non-overlapping row-major page layout; "
                f"axis {axis} stride {stride} is smaller than inner span {inner_span}, "
                f"shape={shape}, strides={strides}"
            )
        inner_span += (extent - 1) * stride

    physical_span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides, strict=True))
    logical_numel = math.prod(shape)
    storage_offset = int(tensor.storage_offset())
    if storage_offset < 0:
        raise PhysicalStorageLayoutError(f"{name}: storage_offset must be non-negative, got {storage_offset}")

    element_size = int(tensor.element_size())
    storage_nbytes = int(tensor.untyped_storage().nbytes())
    if element_size <= 0 or storage_nbytes % element_size != 0:
        raise PhysicalStorageLayoutError(
            f"{name}: storage byte size {storage_nbytes} is incompatible with element size {element_size}"
        )
    storage_numel = storage_nbytes // element_size
    end_offset = storage_offset + physical_span
    if end_offset > storage_numel:
        raise PhysicalStorageLayoutError(
            f"{name}: physical span [{storage_offset}, {end_offset}) exceeds storage capacity {storage_numel} elements"
        )

    return PhysicalStorageLayout(
        name=name,
        shape=shape,
        strides=strides,
        storage_offset=storage_offset,
        physical_span=physical_span,
        storage_numel=storage_numel,
        logical_numel=logical_numel,
        element_size=element_size,
    )


def prepare_physical_storage_alias(
    tensor: torch.Tensor,
    *,
    name: str = "tensor",
) -> PreparedPhysicalStorageAlias:
    """Prepare a zero-copy canonical alias for explicit physical addressing.

    This function belongs in adapter initialization/warmup, never in an L1
    launch or ACLGraph capture/replay path.  The returned alias has shape
    ``[1, physical_span]`` and starts at the same logical element as ``tensor``.
    Inter-page padding remains present at its original offsets.
    """
    layout = _layout_from_tensor(tensor, name)
    alias = torch.as_strided(
        tensor,
        size=(1, layout.physical_span),
        stride=(layout.physical_span, 1),
        storage_offset=layout.storage_offset,
    )

    # These checks are cheap initialization-time assertions.  They guard
    # against accidentally replacing this helper with a copying reshape.
    if alias.data_ptr() != tensor.data_ptr():
        raise RuntimeError(f"{name}: physical alias changed the tensor data pointer")
    if alias.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr():
        raise RuntimeError(f"{name}: physical alias does not share the source storage")
    if alias.storage_offset() != layout.storage_offset:
        raise RuntimeError(f"{name}: physical alias changed storage_offset")

    return PreparedPhysicalStorageAlias(
        source=tensor,
        launch_tensor=alias,
        layout=layout,
    )


__all__ = [
    "PhysicalStorageLayout",
    "PhysicalStorageLayoutError",
    "PreparedPhysicalStorageAlias",
    "prepare_physical_storage_alias",
]
