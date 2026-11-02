#!/usr/bin/env python3
"""Verify the SHA-256 inventory for supplemental TA-01 analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/validation/ta01-review-supplement-manifest.json",
        help="Manifest path relative to the project root",
    )
    args = parser.parse_args()
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            failures.append(f"MISSING {item['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            failures.append(f"HASH MISMATCH {item['path']}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Verified {len(manifest['files'])} files from {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
