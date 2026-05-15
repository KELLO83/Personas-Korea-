from __future__ import annotations

import pytest

from GNN_Neural_Network.gnn_recommender.config import (
    ExperimentalFeatureConfig,
    LightGCNConfig,
    TrainConfig,
    validate_experimental_feature_policy,
)
from GNN_Neural_Network.gnn_recommender.data import IndexedEdges
from GNN_Neural_Network.gnn_recommender.model import LightGCN, XSimGCL
from GNN_Neural_Network.gnn_recommender.train import _build_model


def _indexed() -> IndexedEdges:
    return IndexedEdges(edges=[(0, 0)], person_to_id={"p1": 0}, hobby_to_id={"h1": 0})


def test_experimental_features_are_disabled_by_default() -> None:
    config = LightGCNConfig()
    with pytest.raises(ValueError, match="KURE dense MMR"):
        validate_experimental_feature_policy(config, use_kure_mmr=True)
    with pytest.raises(ValueError, match="text embedding feature"):
        validate_experimental_feature_policy(config, include_text_embedding_feature=True)
    with pytest.raises(ValueError, match="XSimGCL"):
        validate_experimental_feature_policy(config, use_xsimgcl=True)
    with pytest.raises(ValueError, match="source one-hot features"):
        validate_experimental_feature_policy(config, include_source_features=True)


def test_default_model_type_builds_lightgcn() -> None:
    model = _build_model(_indexed(), LightGCNConfig())
    assert isinstance(model, LightGCN)
    assert not isinstance(model, XSimGCL)


def test_xsimgcl_requires_explicit_opt_in() -> None:
    config = LightGCNConfig(train=TrainConfig(model_type="xsimgcl"))
    with pytest.raises(ValueError, match="allow_xsimgcl=true"):
        _build_model(_indexed(), config)


def test_xsimgcl_builds_when_explicitly_enabled() -> None:
    config = LightGCNConfig(
        train=TrainConfig(model_type="xsimgcl"),
        experimental=ExperimentalFeatureConfig(allow_xsimgcl=True),
    )
    model = _build_model(_indexed(), config)
    assert isinstance(model, XSimGCL)


def test_text_source_and_kure_mmr_guardrails_pass_when_enabled() -> None:
    config = LightGCNConfig(
        experimental=ExperimentalFeatureConfig(
            allow_kure_mmr=True,
            allow_text_embedding_feature=True,
            allow_source_features=True,
        ),
    )
    validate_experimental_feature_policy(
        config,
        use_kure_mmr=True,
        include_text_embedding_feature=True,
        include_source_features=True,
    )
