from __future__ import annotations

import pandas as pd

from src.data.sampling import normalize_age_group_tokens, sample_age_groups


def _person_row(age_group: str) -> dict[str, object]:
    return {
        "uuid": f"uuid-{age_group}",
        "age_group": age_group,
    }


def test_normalize_age_group_tokens_accepts_numbers_and_decorated_labels() -> None:
    result = normalize_age_group_tokens("10,20대,30")
    assert result == ["10대", "20대", "30대"]


def test_sample_age_groups_balances_target_for_requested_ages() -> None:
    rows: list[dict[str, object]] = []
    for age_group in ("10대", "20대", "30대"):
        for i in range(4000):
            rows.append(_person_row(f"{age_group}-{i}"))

    df = pd.DataFrame(rows)
    # age_group uses string with suffix for this dataset
    df["age_group"] = df["uuid"].str.extract(r"(\d+대)-")[0]

    sampled = sample_age_groups(df, age_groups=["10대", "20대", "30대"], max_rows=10_000, random_seed=123)

    counts = sampled["age_group"].value_counts().to_dict()
    assert counts == {"30대": 3334, "10대": 3333, "20대": 3333}


def test_sample_age_groups_keeps_all_if_group_data_is_small() -> None:
    rows = [
        {"uuid": "uuid-10", "age_group": "10대"},
        {"uuid": "uuid-11", "age_group": "10대"},
        {"uuid": "uuid-20", "age_group": "20대"},
    ]
    df = pd.DataFrame(rows)

    sampled = sample_age_groups(df, age_groups=["10대", "20대", "30대"], max_rows=10_000, random_seed=1)
    assert len(sampled) == 3
    assert set(sampled["age_group"]) == {"10대", "20대"}
