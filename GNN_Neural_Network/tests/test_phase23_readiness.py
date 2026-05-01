import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISIONS_PATH = ROOT / "artifacts" / "experiment_decisions.json"
SUMMARY_PATH = ROOT / "artifacts" / "experiment_run_summary.md"


def test_phase23_artifacts_exist() -> None:
    assert DECISIONS_PATH.exists(), f"Missing experiment decisions artifact: {DECISIONS_PATH}"
    assert SUMMARY_PATH.exists(), f"Missing experiment summary artifact: {SUMMARY_PATH}"


def test_phase23_decision_file_has_promoted_lightgbm_defaults() -> None:
    raw = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)

    current_system = raw["current_system"]
    assert isinstance(current_system, dict)

    stage1 = current_system["stage1"]
    assert isinstance(stage1, dict)
    stage2 = current_system["stage2"]
    assert isinstance(stage2, dict)
    mmr = current_system["mmr"]
    assert isinstance(mmr, dict)

    assert stage1["status"] == "selected"
    assert stage1["selected_baseline"] == ["popularity", "cooccurrence"]

    assert stage2["status"] == "promoted"
    assert stage2["strategy"] == "LightGBM binary classifier learned ranker"

    assert mmr["status"] == "experimental"
    assert mmr["default_enabled"] is False

    next_priorities = raw["next_priorities"]
    assert isinstance(next_priorities, list)
    assert any(
        isinstance(item, dict)
        and item.get("priority") == 1
        and "LightGBM regularization" in str(item.get("task", ""))
        for item in next_priorities
    )


def test_phase23_summary_documents_status() -> None:
    text = SUMMARY_PATH.read_text(encoding="utf-8")

    assert "Current default path" in text
    assert "Stage 1: `popularity + cooccurrence`" in text
    assert "v2 LightGBM ranker" in text
    assert "Key Lessons" in text
    lowered = text.lower()
    assert "no-go" in lowered
