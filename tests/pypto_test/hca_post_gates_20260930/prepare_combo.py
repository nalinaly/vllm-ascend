"""叠加已测Q分组与post门控候选，单独测组合增量。"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "source.json"
    manifest = json.loads(path.read_text())
    groups = json.loads((ROOT.parent / "hca_qb_groups_20260930/source.json").read_text())
    baseline = Path(groups["sources"]["ready_v2"])
    gate_source = Path(manifest["sources"]["transpose"])
    candidate = gate_source.with_name("combo")
    assert not candidate.exists()
    shutil.copytree(baseline, candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative = Path("deepseek_v4_flash_hca/o_proj_hc_post.py")
    file = candidate / relative
    file.chmod(0o644)
    file.write_text((gate_source / relative).read_text())
    file.chmod(0o444)
    manifest["sources"].update(combo=str(candidate), combo_base=str(baseline))
    manifest["combo_scope"] = "baseline is ready_v2 Q grouping; candidate adds only post coefficient transpose"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
