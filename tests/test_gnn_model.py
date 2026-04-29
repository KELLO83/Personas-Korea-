import pytest
import torch

from GNN_Neural_Network.gnn_recommender.model import LightGCN, XSimGCL, build_normalized_adjacency, info_nce_loss


def test_lightgcn_forward_matches_score() -> None:
    device = torch.device("cpu")
    model = LightGCN(num_persons=2, num_hobbies=2, embedding_dim=4, num_layers=1)
    adjacency = build_normalized_adjacency(
        num_persons=2,
        num_hobbies=2,
        train_edges=[(0, 0), (1, 1)],
        device=device,
    )
    person_ids = torch.tensor([0, 1], dtype=torch.long)
    hobby_ids = torch.tensor([0, 1], dtype=torch.long)

    assert torch.equal(model(person_ids, hobby_ids, adjacency), model.score(person_ids, hobby_ids, adjacency))


def test_xsimgcl_forward_matches_lightgcn_shape() -> None:
    device = torch.device("cpu")
    model = XSimGCL(num_persons=2, num_hobbies=3, embedding_dim=4, num_layers=2, contrastive_layer=1)
    adjacency = build_normalized_adjacency(
        num_persons=2,
        num_hobbies=3,
        train_edges=[(0, 0), (0, 1), (1, 2)],
        device=device,
    )
    person_embeddings, hobby_embeddings = model.all_embeddings(adjacency)

    assert person_embeddings.shape == (2, 4)
    assert hobby_embeddings.shape == (3, 4)


def test_xsimgcl_contrastive_views_and_loss() -> None:
    device = torch.device("cpu")
    model = XSimGCL(num_persons=2, num_hobbies=2, embedding_dim=4, num_layers=1, contrastive_layer=1, noise_epsilon=0.1)
    adjacency = build_normalized_adjacency(
        num_persons=2,
        num_hobbies=2,
        train_edges=[(0, 0), (1, 1)],
        device=device,
    )
    person_ids = torch.tensor([0, 1], dtype=torch.long)
    hobby_ids = torch.tensor([0, 1], dtype=torch.long)

    users_one, hobbies_one, users_two, hobbies_two = model.contrastive_embeddings(adjacency)
    loss = model.contrastive_loss(adjacency, person_ids, hobby_ids)

    assert users_one.shape == users_two.shape == (2, 4)
    assert hobbies_one.shape == hobbies_two.shape == (2, 4)
    assert loss.item() >= 0.0


def test_xsimgcl_contrastive_loss_deduplicates_batch_ids(monkeypatch) -> None:
    device = torch.device("cpu")
    model = XSimGCL(num_persons=2, num_hobbies=2, embedding_dim=4, num_layers=1, contrastive_layer=1, noise_epsilon=0.0)
    adjacency = build_normalized_adjacency(
        num_persons=2,
        num_hobbies=2,
        train_edges=[(0, 0), (1, 1)],
        device=device,
    )
    seen_shapes: list[tuple[int, ...]] = []

    def _fake_info_nce(view_one: torch.Tensor, view_two: torch.Tensor, temperature: float) -> torch.Tensor:
        seen_shapes.append(tuple(view_one.shape))
        return view_one.sum() * 0.0 + view_two.sum() * 0.0 + temperature * 0.0

    monkeypatch.setattr("GNN_Neural_Network.gnn_recommender.model.info_nce_loss", _fake_info_nce)

    model.contrastive_loss(
        adjacency,
        person_ids=torch.tensor([0, 0, 1], dtype=torch.long),
        hobby_ids=torch.tensor([1, 1, 1], dtype=torch.long),
    )

    assert seen_shapes == [(2, 4), (1, 4)]


def test_xsimgcl_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="contrastive_layer"):
        XSimGCL(num_persons=2, num_hobbies=2, embedding_dim=4, num_layers=1, contrastive_layer=2)
    with pytest.raises(ValueError, match="temperature"):
        info_nce_loss(torch.ones(2, 4), torch.ones(2, 4), 0.0)
