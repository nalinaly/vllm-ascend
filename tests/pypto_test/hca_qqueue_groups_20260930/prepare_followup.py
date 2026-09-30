"""对重复占核的实测泳道验证整组同步与上游禁止提前预置，两项分开测。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

from prepare import PREFIX, RELATIVE, ROOT


def main():
    manifest = json.loads((ROOT / "source.json").read_text())
    sources = manifest["sources"]
    changes = {
        "whole_sync": ("cube_groups1", RELATIVE,
                       'name_hint="hca_qb_stream", deps=',
                       'name_hint="hca_qb_stream", sync_start=True, deps='),
        "qr_late": ("base", Path("deepseek_v4_flash_dspark_perf/qkv_proj_rope.py"),
                    'name_hint="qr_rms_norm_quant", allow_early_resolve=True',
                    'name_hint="qr_rms_norm_quant", allow_early_resolve=False'),
    }
    for side, (parent, relative, old, new) in changes.items():
        dest = PREFIX / side
        assert not dest.exists()
        shutil.copytree(sources[parent], dest)
        path = dest / relative
        before = path.read_text()
        assert before.count(old) == 1
        after = before.replace(old, new)
        ast.parse(after)
        path.chmod(0o644)
        path.write_text(after)
        path.chmod(0o444)
        (ROOT / f"{side}.patch").write_text(f"# Parent: {parent}\n" + "".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), n=0,
            fromfile="a/" + str(relative), tofile="b/" + str(relative),
        )))
        sources[side] = str(dest)
    manifest["followup_parents"] = {name: values[0] for name, values in changes.items()}
    manifest["followup_basis"] = "even single-24 Q used some cores twice while others idle; check pending pre-stage"
    (ROOT / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
