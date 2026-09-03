#!/usr/bin/env python
"""Record the software and hardware environment for reproducibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import environment_info, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/environment.json"),
    )
    args = parser.parse_args()
    info = environment_info(PROJECT_ROOT)
    write_json(info, args.output)
    for key, value in info.items():
        print(f"{key}: {value}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
