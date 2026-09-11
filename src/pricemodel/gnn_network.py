"""Small dependency-free GraphSAGE price network for the GNN baseline."""
from __future__ import annotations

import torch
import torch.nn as nn


class GraphSAGEConv(nn.Module):
    """Mean-neighbour GraphSAGE layer using native PyTorch scatter operations."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.neighbor_projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.self_projection(node_features)
        source, target = edge_index
        aggregate = torch.zeros_like(node_features)
        aggregate.index_add_(0, target, node_features[source])
        counts = torch.zeros((node_features.size(0), 1), device=node_features.device)
        counts.index_add_(0, target, torch.ones((len(target), 1), device=node_features.device))
        return self.self_projection(node_features) + self.neighbor_projection(
            aggregate / counts.clamp_min(1.0)
        )


class H3GraphPriceModel(nn.Module):
    """Two-hop H3 GraphSAGE encoder plus a property/time/market price head."""

    def __init__(
        self,
        node_feature_dim: int,
        property_dim: int,
        time_dim: int,
        market_dim: int,
        graph_hidden_dim: int = 32,
        head_hidden_dim: int = 128,
        dropout_rate: float = 0.2,
        graph_layer_norm: bool = False,
        graph_residual: bool = False,
    ):
        super().__init__()
        self.graph_layer_one = GraphSAGEConv(node_feature_dim, graph_hidden_dim)
        self.graph_layer_two = GraphSAGEConv(graph_hidden_dim, graph_hidden_dim)
        self.use_graph_layer_norm = bool(graph_layer_norm)
        self.use_graph_residual = bool(graph_residual)
        if self.use_graph_layer_norm:
            self.graph_norm_one = nn.LayerNorm(graph_hidden_dim)
            self.graph_norm_two = nn.LayerNorm(graph_hidden_dim)
        sale_feature_dim = property_dim + time_dim + market_dim
        self.price_head = nn.Sequential(
            nn.Linear(graph_hidden_dim + sale_feature_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(head_hidden_dim // 2, 1),
        )

    def encode_nodes(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return embeddings with two rounds of one-ring message passing."""
        encoded = self.graph_layer_one(node_features, edge_index)
        if self.use_graph_layer_norm:
            encoded = self.graph_norm_one(encoded)
        encoded = torch.relu(encoded)

        updated = self.graph_layer_two(encoded, edge_index)
        if self.use_graph_residual:
            updated = updated + encoded
        if self.use_graph_layer_norm:
            updated = self.graph_norm_two(updated)
        return torch.relu(updated)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        sale_node_index: torch.Tensor,
        property_features: torch.Tensor,
        time_features: torch.Tensor,
        market_features: torch.Tensor,
    ) -> torch.Tensor:
        node_embeddings = self.encode_nodes(node_features, edge_index)
        sale_inputs = torch.cat(
            [node_embeddings[sale_node_index], property_features, time_features, market_features],
            dim=1,
        )
        return self.price_head(sale_inputs)
