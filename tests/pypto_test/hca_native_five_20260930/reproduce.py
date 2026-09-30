"""从指定Git基线导出私有算子包，应用完整补丁；不改工作区、不占NPU。"""

import argparse
import io
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINES = {
    "p1_oa_native": "55852cff",
    "p2_kshift": "55852cff",
    "p2_schedule_only": "55852cff",
    "p2_kshift_schedule": "55852cff",
    "p3_qb_pipeline": "55852cff",
    "p4_qb_activation_reuse": "55852cff",
    "p5_qa_native": "55852cff",
    "integrated_13": "7d07a579",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=BASELINES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=ROOT.parents[2])
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(f"不覆盖已冻结的包：{target}")
    prefix = "vllm_ascend/ops/pypto/"
    archive = subprocess.check_output([
        "git", "-C", str(args.repository), "archive", BASELINES[args.variant], prefix,
    ])
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            relative = Path(member.name[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(relative)
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(bundle.extractfile(member).read())
    subprocess.run(
        ["git", "apply", "--unidiff-zero", str(ROOT / f"{args.variant}.patch")], cwd=target, check=True,
    )
    for path in target.rglob("*.py"):
        path.chmod(0o444)
    print(target)


if __name__ == "__main__":
    main()
