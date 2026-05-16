from __future__ import annotations

from pathlib import Path

from GNN_Neural_Network.scripts.train_ranker import _embedding_model_metadata


def test_embedding_model_metadata_records_requested_backbone(tmp_path: Path) -> None:
    metadata = _embedding_model_metadata(
        enabled=True,
        model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        model_revision="rev-a",
        cache_dir=tmp_path,
        batch_size=16,
        device="cuda",
        resource_plan={"effective_batch_size": 16},
    )

    assert metadata["model_name"] == "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    assert metadata["model_revision"] == "rev-a"
    assert metadata["preprocessing_version"] == "domain_tagged_masked_v1"
    assert metadata["cache_key_policy"] == "model_name|model_revision|preprocessing_version|text"
