import json
from pathlib import Path


GOLDEN_SET_PATH = Path("configs/target_persona_golden_set.json")


def test_target_persona_golden_set_shape() -> None:
    data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))

    assert data["version"]
    assert data["quality_gates"]["min_cases"] <= len(data["cases"])
    assert data["quality_gates"]["must_preserve_evidence"] is True
    assert "evidence_uuids" in data["quality_gates"]["required_response_fields"]

    case_ids = [case["id"] for case in data["cases"]]
    assert len(case_ids) == len(set(case_ids))

    for case in data["cases"]:
        assert case["description"]
        assert isinstance(case["params"], dict)
        assert case["expected"]
        assert case["forbidden"]


def test_target_persona_golden_set_has_positive_and_rejection_cases() -> None:
    data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]

    positive_cases = [case for case in cases if "min_sample_size" in case["expected"]]
    rejection_cases = [case for case in cases if case["expected"].get("status_code") == 400]

    assert len(positive_cases) >= 4
    assert len(rejection_cases) >= 1
    assert any("semantic_query" in case["params"] for case in positive_cases)
