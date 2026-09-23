"""Run the existing profiler with one frozen CSA package, without editing the checkout."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO / "tests/pypto_test"))

from dsv4_csa_env import activate

activate()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--snapshot", type=Path, required=True)
args, remaining = parser.parse_known_args()
snapshot = args.snapshot.resolve()
prefix = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark"
package_path = snapshot / prefix.replace(".", "/")
assert package_path.is_dir(), package_path
assert not any(name.startswith(prefix + ".") for name in sys.modules)

import vllm_ascend.ops.pypto.deepseek_v4_flash_dspark as package

# The package initializer is empty apart from its license. All algorithm
# modules resolve through this immutable snapshot in this fresh process.
assert (package_path / "__init__.py").read_bytes() == Path(package.__file__).read_bytes()
package.__path__ = [str(package_path)]

import dsv4_csa_profile

sys.argv = [dsv4_csa_profile.__file__, *remaining]
dsv4_csa_profile.main()
loaded = {}
for name, module in sys.modules.items():
    if name.startswith(prefix + "."):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(package_path), (name, path)
        loaded[str(path.relative_to(snapshot))] = hashlib.sha256(path.read_bytes()).hexdigest()
output = Path(remaining[remaining.index("--output-dir") + 1])
(output / "source_snapshot.json").write_text(json.dumps({
    "snapshot": str(snapshot), "loaded_csa_sources": loaded,
    "profiler": dsv4_csa_profile.__file__, "production_checkout_modified": False,
}, indent=2) + "\n")
