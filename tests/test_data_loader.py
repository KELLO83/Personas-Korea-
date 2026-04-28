from pathlib import Path

import pandas as pd
import pytest

from src.data import loader


def _valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "uuid": "03b4f36a18e6469386d0286dddd513c8",
                "professional_persona": "직업 서술",
                "sports_persona": "운동 서술",
                "arts_persona": "예술 서술",
                "travel_persona": "여행 서술",
                "culinary_persona": "음식 서술",
                "family_persona": "가족 서술",
                "persona": "요약 서술",
                "cultural_background": "문화 배경",
                "skills_and_expertise": "기술 서술",
                "hobbies_and_interests": "취미 서술",
                "career_goals_and_ambitions": "목표 서술",
                "skills_and_expertise_list": "['기술']",
                "hobbies_and_interests_list": "['취미']",
                "sex": "남자",
                "age": 74,
                "marital_status": "배우자있음",
                "military_status": "비현역",
                "family_type": "배우자와 거주",
                "housing_type": "아파트",
                "education_level": "초등학교",
                "bachelors_field": "해당없음",
                "occupation": "하역 및 적재 관련 단순 종사원",
                "district": "광주-서구",
                "province": "광주",
                "country": "대한민국",
            }
        ]
    )


def test_load_dataset_reads_local_csv_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_file = tmp_path / "personas.csv"
    _valid_dataframe().to_csv(data_file, index=False)
    monkeypatch.setattr(loader.settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(loader.settings, "DATA_FILE", "personas.parquet")

    result = loader.load_dataset()

    assert len(result) == 1
    assert result.iloc[0]["uuid"] == "03b4f36a18e6469386d0286dddd513c8"


def test_load_dataset_uses_huggingface_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataset:
        def to_pandas(self) -> pd.DataFrame:
            return _valid_dataframe()

    calls = []

    def fake_load_hf_dataset(dataset_id: str, split: str) -> FakeDataset:
        calls.append((dataset_id, split))
        return FakeDataset()

    monkeypatch.setattr(loader.settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(loader.settings, "DATA_FILE", "personas.parquet")
    monkeypatch.setattr(loader.settings, "HF_DATASET_ID", "nvidia/Nemotron-Personas-Korea")
    monkeypatch.setattr(loader.settings, "HF_DATASET_SPLIT", "train")
    monkeypatch.setattr(loader.settings, "DATA_SAMPLE_SIZE", 10000)
    monkeypatch.setattr(loader, "load_hf_dataset", fake_load_hf_dataset)

    result = loader.load_dataset()

    assert len(result) == 1
    assert calls == [("nvidia/Nemotron-Personas-Korea", "train[:10000]")]


def test_load_dataset_uses_full_huggingface_split_when_sample_size_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataset:
        def to_pandas(self) -> pd.DataFrame:
            return _valid_dataframe()

    calls = []

    def fake_load_hf_dataset(dataset_id: str, split: str) -> FakeDataset:
        calls.append((dataset_id, split))
        return FakeDataset()

    monkeypatch.setattr(loader.settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(loader.settings, "DATA_FILE", "personas.parquet")
    monkeypatch.setattr(loader.settings, "HF_DATASET_ID", "nvidia/Nemotron-Personas-Korea")
    monkeypatch.setattr(loader.settings, "HF_DATASET_SPLIT", "train")
    monkeypatch.setattr(loader, "load_hf_dataset", fake_load_hf_dataset)

    result = loader.load_dataset(sample_size=None)

    assert len(result) == 1
    assert calls == [("nvidia/Nemotron-Personas-Korea", "train")]


def test_load_dataset_defaults_to_full_split_when_sample_size_setting_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataset:
        def to_pandas(self) -> pd.DataFrame:
            return _valid_dataframe()

    calls = []

    def fake_load_hf_dataset(dataset_id: str, split: str) -> FakeDataset:
        calls.append((dataset_id, split))
        return FakeDataset()

    monkeypatch.setattr(loader.settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(loader.settings, "DATA_FILE", "personas.parquet")
    monkeypatch.setattr(loader.settings, "HF_DATASET_ID", "nvidia/Nemotron-Personas-Korea")
    monkeypatch.setattr(loader.settings, "HF_DATASET_SPLIT", "train")
    monkeypatch.setattr(loader.settings, "DATA_SAMPLE_SIZE", None)
    monkeypatch.setattr(loader, "load_hf_dataset", fake_load_hf_dataset)

    result = loader.load_dataset()

    assert len(result) == 1
    assert calls == [("nvidia/Nemotron-Personas-Korea", "train")]


def test_validate_dataframe_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="Dataset is missing required columns"):
        loader._validate_dataframe(pd.DataFrame([{"uuid": "x"}]))
