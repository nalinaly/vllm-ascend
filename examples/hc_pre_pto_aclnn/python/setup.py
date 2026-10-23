# SPDX-License-Identifier: Apache-2.0
"""Package the extension produced by scripts/build_torch.py."""
from pathlib import Path

from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


if not (Path(__file__).parent / "bytenn_pto_ops/bytenn_pto_ops_npu.so").is_file():
    raise RuntimeError("Run scripts/build_torch.py before packaging bytenn_pto_ops")

setup(
    name="bytenn_pto_ops",
    version="0.1.0",
    packages=["bytenn_pto_ops"],
    package_data={"bytenn_pto_ops": ["*.so"]},
    distclass=BinaryDistribution,
)
