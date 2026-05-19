from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.persona_similarity.scripts.common import load_config
from experiments.persona_similarity.scripts.evaluation_utils import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate_model(config, "text_all_lambdarank", features_path=config["paths"]["features_with_text"], force=args.force)


if __name__ == "__main__":
    main()
