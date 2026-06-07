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
        attention_implementation="sdpa",
        torch_dtype="float16",
        torch_compile=False,
        torch_compile_mode="reduce-overhead",
        backend_policy={"sdpa_backend_selection": "auto_by_pytorch_dispatcher"},
    )

    assert metadata["model_name"] == "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    assert metadata["model_revision"] == "rev-a"
    assert metadata["preprocessing_version"] == "domain_tagged_masked_v1"
    assert metadata["attention_implementation"] == "sdpa"
    assert metadata["torch_dtype"] == "float16"
    assert metadata["torch_compile"] is False
    assert metadata["backend_policy"]["sdpa_backend_selection"] == "auto_by_pytorch_dispatcher"
    assert metadata["cache_key_policy"] == "model_name|model_revision|preprocessing_version|attention_implementation|torch_dtype|torch_compile|torch_compile_mode|text"
