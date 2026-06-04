from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.persona_similarity.scripts.common import ensure_parent
from experiments.persona_similarity.scripts.relational_stage1 import RelationalStage1Spec, build_experiment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HGT/RGCN relational Stage1 experiment manifest.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--provider", choices=["hgt", "rgcn"], required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/persona_similarity/artifacts/metrics/relational_stage1_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_experiment_manifest(
        RelationalStage1Spec(
            experiment_id=args.experiment_id,
            provider=args.provider,
            top_k=args.top_k,
        )
    )
    output = ensure_parent(args.output)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
