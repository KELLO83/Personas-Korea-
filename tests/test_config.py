import pytest

from src.config import Settings


def test_data_sample_size_blank_string_becomes_none() -> None:
    assert Settings(DATA_SAMPLE_SIZE="").DATA_SAMPLE_SIZE is None


def test_data_sample_size_string_is_converted_to_int() -> None:
    assert Settings(DATA_SAMPLE_SIZE="12").DATA_SAMPLE_SIZE == 12


def test_data_sample_size_rejects_non_numeric() -> None:
    with pytest.raises(ValueError):
        Settings(DATA_SAMPLE_SIZE="abc")
