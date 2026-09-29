"""CPU预编译独立候选；失败项留下诊断，不占卡试错。"""

import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def compile_one(item):
    name = item["name"]
    with (ROOT / f"compile_{name}.log").open("w") as output:
        result = subprocess.run(
            [sys.executable, str(ROOT / "compile.py"), name], stdout=output, stderr=subprocess.STDOUT, check=False
        )
    print(name, result.returncode, flush=True)
    return {"name": name, "exit_code": result.returncode}


def main():
    variants = json.loads((ROOT / "source.json").read_text())["variants"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(compile_one, variants))
    (ROOT / "compile_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
