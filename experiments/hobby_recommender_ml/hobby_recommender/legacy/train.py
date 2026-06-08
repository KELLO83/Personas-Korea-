from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .baseline import baseline_ranking_metrics
from .config import LightGCNConfig
from .data import EdgeSplit, IndexedEdges, iter_bpr_batches, save_json
from .metrics import summarize_ranking_metrics
from .model import LightGCN, bpr_loss, build_normalized_adjacency, choose_device
from .recommend import batch_recommend_hobbies


def train_lightgcn(indexed: IndexedEdges, split: EdgeSplit, config: LightGCNConfig) -> dict[str, Any]:
    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)
    device = choose_device(config.train.device)
    model = LightGCN(
        num_persons=len(indexed.person_to_id),
        num_hobbies=len(indexed.hobby_to_id),
        embedding_dim=config.train.embedding_dim,
        num_layers=config.train.num_layers,
    ).to(device)
    adjacency = build_normalized_adjacency(
        num_persons=model.num_persons,
        num_hobbies=model.num_hobbies,
        train_edges=split.train,
        device=device,
    )
    optimizer = _build_optimizer(model, config)
    history: list[dict[str, float]] = []
    best_recall = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        losses: list[float] = []
        for users, positives, negatives in tqdm(
            iter_bpr_batches(
                train_edges=split.train,
                num_hobbies=model.num_hobbies,
                full_known=split.full_known,
                batch_size=config.train.batch_size,
                seed=config.train.seed + epoch,
            ),
            desc=f"epoch {epoch}",
        ):
            user_tensor = torch.tensor(users, dtype=torch.long, device=device)
            positive_tensor = torch.tensor(positives, dtype=torch.long, device=device)
            negative_tensor = torch.tensor(negatives, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            person_embeddings, hobby_embeddings = model.all_embeddings(adjacency)
            batch_person_embeddings = person_embeddings[user_tensor]
            positive_scores = (batch_person_embeddings * hobby_embeddings[positive_tensor]).sum(dim=1)
            negative_scores = (batch_person_embeddings * hobby_embeddings[negative_tensor]).sum(dim=1)
            loss = bpr_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate_model(model, adjacency, split, config, device)
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        epoch_result = {"epoch": float(epoch), "loss": mean_loss, **metrics}
        history.append(epoch_result)
        recall_10 = metrics.get("recall@10", 0.0)
        if recall_10 > best_recall:
            best_recall = recall_10
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _save_checkpoint(config.paths.checkpoint, model, best_state, config, indexed)
    metrics_payload = {
        "history": history,
        "best_recall@10": best_recall,
        "baseline_validation": baseline_ranking_metrics(
            train_edges=split.train,
            target_edges=split.validation,
            known_by_person=split.train_known,
            top_k_values=config.eval.top_k,
        ),
        "num_persons": len(indexed.person_to_id),
        "num_hobbies": len(indexed.hobby_to_id),
        "num_edges": len(indexed.edges),
    }
    save_json(config.paths.metrics, metrics_payload)
    return metrics_payload


def evaluate_model(
    model: LightGCN,
    adjacency: torch.Tensor,
    split: EdgeSplit,
    config: LightGCNConfig,
    device: torch.device,
) -> dict[str, float]:
    truth: dict[int, set[int]] = {}
    for person_id, hobby_id in split.validation:
        truth.setdefault(person_id, set()).add(hobby_id)
    if not truth:
        return {f"{name}@{k}": 0.0 for k in config.eval.top_k for name in ("recall", "ndcg", "hit_rate")}
    max_k = max(config.eval.top_k)
    recommended = batch_recommend_hobbies(
        model=model,
        adjacency=adjacency,
        person_ids=list(truth),
        known_by_person=split.train_known,
        top_k=max_k,
        chunk_size=config.eval.score_chunk_size,
        device=device,
    )
    return summarize_ranking_metrics(truth, recommended, config.eval.top_k)


def _build_optimizer(model: LightGCN, config: LightGCNConfig) -> torch.optim.Optimizer:
    name = config.train.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    raise ValueError("optimizer must be 'adam' or 'adamw'")


def _save_checkpoint(
    path: Path,
    model: LightGCN,
    state_dict: dict[str, torch.Tensor],
    config: LightGCNConfig,
    indexed: IndexedEdges,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "num_persons": model.num_persons,
            "num_hobbies": model.num_hobbies,
            "embedding_dim": config.train.embedding_dim,
            "num_layers": config.train.num_layers,
            "person_to_id": indexed.person_to_id,
            "hobby_to_id": indexed.hobby_to_id,
        },
        path,
    )
