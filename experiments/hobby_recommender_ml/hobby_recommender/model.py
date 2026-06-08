from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LightGCN(nn.Module):
    def __init__(self, num_persons: int, num_hobbies: int, embedding_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_persons <= 0 or num_hobbies <= 0:
            raise ValueError("num_persons and num_hobbies must be positive")
        if embedding_dim <= 0 or num_layers < 0:
            raise ValueError("embedding_dim must be positive and num_layers must be non-negative")
        self.num_persons: int = num_persons
        self.num_hobbies: int = num_hobbies
        self.num_layers: int = num_layers
        self.person_embedding: nn.Embedding = nn.Embedding(num_persons, embedding_dim)
        self.hobby_embedding: nn.Embedding = nn.Embedding(num_hobbies, embedding_dim)
        _ = nn.init.normal_(self.person_embedding.weight, std=0.1)
        _ = nn.init.normal_(self.hobby_embedding.weight, std=0.1)

    def all_embeddings(self, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        combined = torch.cat([self.person_embedding.weight, self.hobby_embedding.weight], dim=0)
        outputs = [combined]
        current = combined
        for _ in range(self.num_layers):
            current = torch.sparse.mm(adjacency, current)
            outputs.append(current)
        averaged = torch.stack(outputs, dim=0).mean(dim=0)
        person_embeddings, hobby_embeddings = torch.split(averaged, [self.num_persons, self.num_hobbies], dim=0)
        return person_embeddings, hobby_embeddings

    def score(self, person_ids: Tensor, hobby_ids: Tensor, adjacency: Tensor) -> Tensor:
        person_embeddings, hobby_embeddings = self.all_embeddings(adjacency)
        return (person_embeddings[person_ids] * hobby_embeddings[hobby_ids]).sum(dim=1)

    def forward(self, person_ids: Tensor, hobby_ids: Tensor, adjacency: Tensor) -> Tensor:
        return self.score(person_ids, hobby_ids, adjacency)


class XSimGCL(LightGCN):
    def __init__(
        self,
        num_persons: int,
        num_hobbies: int,
        embedding_dim: int,
        num_layers: int,
        contrastive_layer: int = 1,
        noise_epsilon: float = 0.1,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(num_persons=num_persons, num_hobbies=num_hobbies, embedding_dim=embedding_dim, num_layers=num_layers)
        if contrastive_layer < 0 or contrastive_layer > num_layers:
            raise ValueError("contrastive_layer must be between 0 and num_layers")
        if noise_epsilon < 0.0:
            raise ValueError("noise_epsilon must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.contrastive_layer: int = contrastive_layer
        self.noise_epsilon: float = noise_epsilon
        self.temperature: float = temperature

    def all_embeddings(self, adjacency: Tensor, perturbed: bool = False) -> tuple[Tensor, Tensor]:
        all_embeddings, _ = self._propagate(adjacency, perturbed=perturbed)
        person_embeddings, hobby_embeddings = torch.split(all_embeddings, [self.num_persons, self.num_hobbies], dim=0)
        return person_embeddings, hobby_embeddings

    def contrastive_embeddings(self, adjacency: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _, view_one = self._propagate(adjacency, perturbed=True)
        _, view_two = self._propagate(adjacency, perturbed=True)
        users_one, hobbies_one = torch.split(view_one, [self.num_persons, self.num_hobbies], dim=0)
        users_two, hobbies_two = torch.split(view_two, [self.num_persons, self.num_hobbies], dim=0)
        return users_one, hobbies_one, users_two, hobbies_two

    def contrastive_loss(self, adjacency: Tensor, person_ids: Tensor, hobby_ids: Tensor) -> Tensor:
        users_one, hobbies_one, users_two, hobbies_two = self.contrastive_embeddings(adjacency)
        unique_person_ids = torch.unique(person_ids)
        unique_hobby_ids = torch.unique(hobby_ids)
        user_loss = info_nce_loss(users_one[unique_person_ids], users_two[unique_person_ids], self.temperature)
        hobby_loss = info_nce_loss(hobbies_one[unique_hobby_ids], hobbies_two[unique_hobby_ids], self.temperature)
        return user_loss + hobby_loss

    def _propagate(self, adjacency: Tensor, *, perturbed: bool) -> tuple[Tensor, Tensor]:
        combined = torch.cat([self.person_embedding.weight, self.hobby_embedding.weight], dim=0)
        outputs = [combined]
        current = combined
        contrastive = combined
        for layer in range(1, self.num_layers + 1):
            current = torch.sparse.mm(adjacency, current)
            if perturbed and self.noise_epsilon > 0.0:
                noise = F.normalize(torch.rand_like(current), dim=-1)
                current = current + torch.sign(current) * noise * self.noise_epsilon
            if layer == self.contrastive_layer:
                contrastive = current
            outputs.append(current)
        averaged = torch.stack(outputs, dim=0).mean(dim=0)
        return averaged, contrastive


def build_normalized_adjacency(
    num_persons: int,
    num_hobbies: int,
    train_edges: list[tuple[int, int]],
    device: torch.device,
) -> Tensor:
    total_nodes = num_persons + num_hobbies
    if not train_edges:
        raise ValueError("train_edges must not be empty")
    row_indices: list[int] = []
    col_indices: list[int] = []
    for person_id, hobby_id in train_edges:
        hobby_node = num_persons + hobby_id
        row_indices.extend([person_id, hobby_node])
        col_indices.extend([hobby_node, person_id])
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    values = torch.ones(len(row_indices), dtype=torch.float32, device=device)
    degree = torch.zeros(total_nodes, dtype=torch.float32, device=device)
    _ = degree.scatter_add_(0, indices[0], values)
    degree_inv_sqrt = torch.pow(degree.clamp_min(1.0), -0.5)
    normalized_values = degree_inv_sqrt[indices[0]] * values * degree_inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(
        indices,
        normalized_values,
        (total_nodes, total_nodes),
        device=device,
        check_invariants=True,
    ).coalesce()


def bpr_loss(positive_scores: Tensor, negative_scores: Tensor) -> Tensor:
    return -F.logsigmoid(positive_scores - negative_scores).mean()


def info_nce_loss(view_one: Tensor, view_two: Tensor, temperature: float) -> Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if view_one.shape != view_two.shape:
        raise ValueError("contrastive views must have the same shape")
    normalized_one = F.normalize(view_one, dim=-1)
    normalized_two = F.normalize(view_two, dim=-1)
    logits = torch.matmul(normalized_one, normalized_two.transpose(0, 1)) / temperature
    labels = torch.arange(view_one.shape[0], dtype=torch.long, device=view_one.device)
    return F.cross_entropy(logits, labels)


def choose_device(configured: str) -> torch.device:
    if configured == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)
