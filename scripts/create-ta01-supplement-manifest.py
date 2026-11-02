#!/usr/bin/env python3
"""Create a supplementary hash manifest without rewriting the historical one."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPLICIT = [
    "src/core/fem-2d.js",
    "src/core/study-case-ta01.js",
    "scripts/generate-fem-dataset.js",
    "scripts/train-lstm.py",
    "scripts/train-physics-guided.py",
    "scripts/generate-lstm-oof-predictions.py",
    "scripts/evaluate-ta01-oof-comparison.py",
    "scripts/evaluate-ta01-oof-all-splits.py",
    "scripts/evaluate-ta01-oof-seed-sensitivity.py",
    "scripts/evaluate-ta01-oof-seed-sensitivity-all-splits.py",
    "scripts/evaluate-ta01-oof-target-regimes.py",
    "scripts/evaluate-ta01-oof-corrector-ablation.py",
    "scripts/evaluate-ta01-rounding-sensitivity.py",
    "scripts/evaluate-ta01-solver-tolerance-sensitivity.py",
    "scripts/evaluate-ta01-solver-tolerance-retrained.py",
    "scripts/evaluate-ta01-proportional-rainfall-baseline.py",
    "scripts/evaluate-ta01-primary-influence-and-target-regimes.py",
    "scripts/generate-ta01-review-figures.py",
    "scripts/render-ta01-review-figures.js",
    "scripts/revise-ta01-manuscript-r3.py",
    "scripts/revise-ta01-manuscript-r4.py",
    "scripts/normalize-ta01-fonts-r5.py",
    "artifacts/manuscript-review-2026-10-29/figura1.svg",
    "artifacts/manuscript-review-2026-10-29/figura1.jpg",
    "artifacts/manuscript-review-2026-10-29/figura2.svg",
    "artifacts/manuscript-review-2026-10-29/figura2.jpg",
    "artifacts/manuscript-review-2026-10-29/figura3.svg",
    "artifacts/manuscript-review-2026-10-29/figura3.jpg",
    "artifacts/manuscript-review-2026-10-29/figura4.svg",
    "artifacts/manuscript-review-2026-10-29/figura4.jpg",
    "artifacts/manuscript-review-2026-10-29/figura5.svg",
    "artifacts/manuscript-review-2026-10-29/figura5.jpg",
    "artifacts/manuscript-review-2026-10-29/figura6.svg",
    "artifacts/manuscript-review-2026-10-29/figura6.jpg",
    "Articulo_G25_revision_integrada_DeepSeek_Claude_R4.docx",
    "Articulo_G25_revision_integrada_DeepSeek_Claude_R5_TNR.docx",
    "scripts/create-ta01-supplement-manifest.py",
    "scripts/verify-ta01-supplement-manifest.py",
    "data/validation/ta01-oof-target-regime-summary.json",
    "data/validation/ta01-rounding-sensitivity.json",
    "data/validation/ta01-solver-tolerance-sensitivity.json",
    "data/validation/ta01-solver-tolerance-retrained-comparison.json",
    "data/validation/ta01-proportional-rainfall-baseline.json",
    "data/validation/ta01-proportional-rainfall-baseline-predictions.csv",
    "data/validation/ta01-primary-influence-and-target-regimes.json",
    "scripts/evaluate-ta01-robustness-repeated.py",
    "data/validation/ta01-robustness-repeated.json",
    "data/validation/ta01-robustness-repeated-draws.csv",
]
PATTERNS = {
    "data/generated": (
        "ta01-fem-500-*.csv",
        "ta01-fem-500-*.json",
        "ta01-physics-guided-*-oof*",
        "ta01-backtest-2024-physics-guided-*-oof*",
        "ta01-chronological-physics-guided-*-oof*",
        "ta01-free-corrector-*-predictions.csv",
    ),
    "data/models": (
        "ta01-lstm-*-oof*",
        "ta01-primary-lstm-*-oof*",
        "ta01-backtest-2024-lstm-*-oof*",
        "ta01-chronological-lstm-*-oof*",
        "ta01-physics-guided-*-oof*",
        "ta01-primary-physics-guided-*-oof*",
        "ta01-backtest-2024-physics-guided-*-oof*",
        "ta01-chronological-physics-guided-*-oof*",
        "ta01-free-corrector-*.json",
    ),
    "data/validation": (
        "ta01-*oof*",
        "ta01-free-corrector-*-predictions.csv",
        "ta01-*rounding*",
        "ta01-*solver-tolerance*",
        "ta01-tol1e-6-*",
    ),
}


def main() -> None:
    paths = {ROOT / item for item in EXPLICIT}
    for folder, patterns in PATTERNS.items():
        for pattern in patterns:
            paths.update((ROOT / folder).glob(pattern))
    files = []
    for path in sorted(paths):
        if path.is_file():
            files.append({
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    manifest = {
        "manifestVersion": 1,
        "scope": "Supplementary TA-01 OOF, target-regime, rounding, solver-tolerance, proportional-rainfall baseline, influence, repeated-perturbation robustness, reproducibility analyses, R4/R5 manuscripts, and figure assets. The historical ta01-replication-manifest.json is preserved and is not replaced by this inventory.",
        "files": files,
    }
    output = ROOT / "data/validation/ta01-review-supplement-manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} hashes to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
