"""Run trusted local workbook self-checks without changing learners' files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check_workbook(path: Path, timeout: float) -> dict:
    """A self-check is feedback, not proof of correctness or a security sandbox."""
    try:
        result = subprocess.run([sys.executable, str(path.resolve())], cwd=path.parent,
                                capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"algorithm": path.parent.name, "status": "timeout"}
    except OSError:
        return {"algorithm": path.parent.name, "status": "error"}
    lines = result.stdout.splitlines()
    if result.returncode != 0 or any(line.startswith("[ERROR]") for line in lines):
        status = "error"
    elif any(line.startswith("[TODO]") for line in lines):
        status = "todo"
    elif (any(line.startswith("[OK]") for line in lines) and
          any(line.startswith("All parts implemented.") for line in lines)):
        status = "self-check-ok"
    else:
        status = "unrecognized"
    return {"algorithm": path.parent.name, "status": status}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check trusted local workbooks. Does not grade solutions or edit progress.")
    parser.add_argument("algorithms", nargs="*", help="Folder names to check; default: all workbooks")
    parser.add_argument("--timeout", type=float, default=60, help="Positive seconds per workbook (default: 60)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary instead of text")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    paths = {p.parent.name: p for p in sorted((ROOT / "algorithms").glob("*/workbook.py"))}
    unknown = set(args.algorithms) - paths.keys()
    if unknown:
        parser.error("Unknown workbook(s): " + ", ".join(sorted(unknown)))
    selected = sorted(set(args.algorithms)) if args.algorithms else list(paths)
    if not selected:
        parser.error("No workbooks found")
    results = [check_workbook(paths[name], args.timeout) for name in selected]
    summary = {"scope": "Local workbook self-checks only; not correctness grading or completion proof.",
               "results": results}
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for item in results:
            print(f"{item['status']:15} {item['algorithm']}")
        print(summary["scope"])
        print("Run the algorithm tests and review your solution before checking off PROGRESS.md.")
    return int(any(item["status"] != "self-check-ok" for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
