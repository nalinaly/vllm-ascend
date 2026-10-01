# SPDX-License-Identifier: Apache-2.0
"""唯一CSA入口；权重准备和Native存储/metadata绑定复用公共基础模块。"""

from ..deepseek_v4_flash_dspark.native_adapter import CSAOperators as _CSAOperators
from ..deepseek_v4_flash_dspark.native_adapter import NativeCSACall as _NativeCSACall
from ..deepseek_v4_flash_dspark.native_adapter import prepare_weights as _prepare_weights
from ..deepseek_v4_flash_dspark.native_storage import indexer_storage
from .decode_csa import _decode_csa_tp1_layer, decode_csa_tp1_layer_hbg, decode_csa_tp1_layer_test
from .host_metadata import CSAHostMetadata


class CSAOperators(_CSAOperators):
    @classmethod
    def register(cls):
        return super().register(decode_csa_tp1_layer_test)


def prepare_weights(attention, hadamard, layer=None):
    return _prepare_weights(attention, hadamard, layer, root_function=_decode_csa_tp1_layer)


class NativeCSACall(_NativeCSACall):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kernel=decode_csa_tp1_layer_test, **kwargs)

    def _indexer_cache_arguments(self):
        native_cache = indexer_storage(*self.views["indexer"])
        return {"idx_native_kv_cache": native_cache}


class HBGCSAOperators(_CSAOperators):
    @classmethod
    def register(cls):
        import pypto.torch

        from ..deepseek_v4_flash_dspark.reduction import validate_reduction_mode

        validate_reduction_mode()
        return cls(pypto.torch.register(decode_csa_tp1_layer_hbg, "dsv4_csa::attention_hbg"))


class HBGNativeCSACall(NativeCSACall):
    """Separate scalar ABI, sharing all Native tensor/cache/weight bindings.

    ``host_metadata`` is required. The caller obtains it from this step's Native
    CPU metadata and must select the matching topology before NPUGraph replay.
    """

    def __init__(self, *args, host_metadata: CSAHostMetadata, **kwargs):
        if not isinstance(host_metadata, CSAHostMetadata):
            raise TypeError("CSA HBG requires explicit CSAHostMetadata")
        super().__init__(*args, host_args={"host_max_seq_len": host_metadata.max_seq_len}, **kwargs)
        if "host_max_seq_len" not in self.param_names:
            raise TypeError("HBGNativeCSACall requires HBGCSAOperators and its Host scalar ABI")
        self.update_host_metadata(host_metadata)

    def update_host_metadata(self, host_metadata: CSAHostMetadata):
        if not isinstance(host_metadata, CSAHostMetadata):
            raise TypeError("CSA HBG requires explicit CSAHostMetadata")
        self.host_metadata = host_metadata
        self.args["host_max_seq_len"] = host_metadata.max_seq_len
        self.core_args = tuple(self.args[name] for name in self.param_names)
