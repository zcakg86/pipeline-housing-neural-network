"""PyTorch network architecture for neighborhood-aware price prediction."""
from __future__ import annotations

import torch
import torch.nn as nn

from .h3_community_embedding import H3CommunityEmbedding, LocalNeighborhoodEncoder

class EnhancedEmbeddingModel(nn.Module):
    """
    Attention-based house price model with H3 L8 neighborhood-aware community embeddings.
    Always builds the uncertainty head (uncertainty_layer); use return_uncertainty=True
    in forward() to get log-variance output at inference time.
    """
    def __init__(self, embedding_dim, hidden_dim, property_dim,
                 continuous_time_dim, market_dim,
                 community_embedding_length,
                 community_embedding_dim=16,
                 dropout_rate=0.1,
                 estimate_uncertainty=False,   # kept for API compat, no longer gates uncertainty_layer
                 use_neighborhood_pooling=True,
                 pooling_strategy='mean',
                 local_feature_dim=0,
                 attention_layer_norm=False,
                 attention_residual=False):
        super().__init__()
        
        self.use_neighborhood_pooling = use_neighborhood_pooling
        self.local_feature_dim = int(local_feature_dim)
        self.community_embedding_dim = int(community_embedding_dim)
        self.use_attention_layer_norm = bool(attention_layer_norm)
        self.use_attention_residual = bool(attention_residual)
        
        # --- Embedding Layers (Categorical) ---
        if use_neighborhood_pooling:
            self.community_embedding = H3CommunityEmbedding(
                num_communities=int(community_embedding_length),
                embedding_dim=self.community_embedding_dim,
                pooling_strategy=pooling_strategy
            )
        else:
            self.community_embedding = nn.Embedding(
                int(community_embedding_length), self.community_embedding_dim
            )
        # Community identity is intentionally compact. Project it into the
        # shared attention width only after lookup so the location table cannot
        # use all 128 token dimensions as an unconstrained identity code.
        self.community_projection = nn.Linear(
            self.community_embedding_dim, embedding_dim
        )
        
        # --- Feature Projection Layers ---
        self.property_feature_layer = nn.Linear(property_dim, embedding_dim)
        self.time_feature_layer = nn.Linear(continuous_time_dim, embedding_dim)
        self.market_feature_layer = nn.Linear(market_dim, embedding_dim)
        
        # --- Learnable CLS Token ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # --- Attention Mechanism ---
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        if self.use_attention_layer_norm:
            self.attention_output_norm = nn.LayerNorm(embedding_dim)
        
        # --- Regressor Head ---
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer  = nn.Linear(hidden_dim // 2, 1)
        
        # Uncertainty head — always built, activated via return_uncertainty at inference
        self.uncertainty_layer = nn.Linear(hidden_dim // 2, 1)

        # Fine-scale branch. It sees each H3 cell's market state separately and
        # learns a gated correction to the broad/global prediction.
        if self.local_feature_dim > 0:
            if not use_neighborhood_pooling:
                raise ValueError("Local market features require H3 neighborhood pooling")
            self.local_neighborhood_encoder = LocalNeighborhoodEncoder(
                embedding_dim, self.local_feature_dim, dropout_rate
            )
            self.local_residual_hidden = nn.Linear(embedding_dim * 2, hidden_dim // 2)
            self.local_residual_layer = nn.Linear(hidden_dim // 2, 1)
            self.local_confidence_gate = nn.Linear(embedding_dim * 2, 1)
            self.local_uncertainty_layer = nn.Linear(embedding_dim, 1)
            # Begin with a conservative local correction. The gate can open
            # when the neighborhood history provides repeatable signal.
            nn.init.constant_(self.local_confidence_gate.bias, -2.0)
        
        self.relu = nn.ReLU()
        self.last_attention_weights = None
        self.last_cls_attention = None
        self.last_local_attention = None
        self.last_local_gate = None
        self.last_local_residual = None

    def forward(self, community_indices, property_features,
                time_features, market_features, local_market_features=None,
                return_uncertainty=False, return_components=False,
                need_weights=False):
        """
        Forward pass with continuous time and market features
        
        Args:
            community_indices: If use_neighborhood_pooling=True, shape (batch, 7)
                             Otherwise, shape (batch,)
            property_features, time_features, market_features: Shape (batch, feature_dim)
        """
        # --- Safety: Clamp indices to valid range ---
        if self.use_neighborhood_pooling:
            # Clamp each of the 7 neighbor indices
            community_indices = torch.clamp(
                community_indices, 0, 
                self.community_embedding.vocab_size - 1
            )
            # Embed once, then share the cell representations between broad
            # community pooling and the fine-scale local residual encoder.
            raw_cell_community_embeddings = self.community_embedding.embedding(
                community_indices
            )
            cell_community_embeddings = self.community_projection(
                raw_cell_community_embeddings
            )
            community_embeddings = self.community_embedding.pool_embeddings(
                cell_community_embeddings
            )
        else:
            # Legacy single index
            community_indices = torch.clamp(
                community_indices, 0, 
                self.community_embedding.num_embeddings - 1
            )
            community_embeddings = self.community_projection(
                self.community_embedding(community_indices)
            )
        
        # --- Process Continuous Features ---
        processed_property = self.relu(self.property_feature_layer(property_features))
        processed_time = self.relu(self.time_feature_layer(time_features))
        processed_market = self.relu(self.market_feature_layer(market_features))
        
        # --- Stack Sequence ---
        # [community, property, time, market]
        tokens = torch.stack([
            community_embeddings,
            processed_property,
            processed_time,
            processed_market
        ], dim=1)
        
        # --- Prepend CLS Token ---
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        
        # --- Self-Attention ---
        attention_output, attention_weights = self.attention_layer(
            seq, seq, seq,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        if self.use_attention_residual:
            attention_output = attention_output + seq
        if self.use_attention_layer_norm:
            attention_output = self.attention_output_norm(attention_output)
        
        cls_out = attention_output[:, 0, :]
        
        # Attention capture is disabled during training for efficiency and
        # explicitly enabled by prediction diagnostics / ONNX export.
        if attention_weights is not None:
            self.last_attention_weights = attention_weights.detach()
            self.last_cls_attention = attention_weights[:, :, 0, 1:].detach()
        else:
            self.last_attention_weights = None
            self.last_cls_attention = None
        
        # --- MLP Head with Dropout ---
        h1 = self.dropout(self.relu(self.hidden_layer1(cls_out)))
        h2 = self.dropout(self.relu(self.hidden_layer2(h1)))
        h3 = self.relu(self.hidden_layer3(h2))
        
        # Broad/global prediction.
        global_output = self.output_layer(h3)
        output = global_output

        local_delta = torch.zeros_like(global_output)
        local_gate = torch.zeros_like(global_output)
        local_representation = None
        if self.local_feature_dim > 0:
            if local_market_features is None:
                raise ValueError("local_market_features are required by this checkpoint")
            local_representation, local_attention = self.local_neighborhood_encoder(
                cell_community_embeddings, local_market_features
            )
            residual_context = torch.cat([cls_out, local_representation], dim=1)
            local_hidden = self.dropout(self.relu(self.local_residual_hidden(residual_context)))
            local_delta = self.local_residual_layer(local_hidden)
            local_gate = torch.sigmoid(self.local_confidence_gate(residual_context))
            output = global_output + local_gate * local_delta
            self.last_local_attention = local_attention.detach()
            self.last_local_gate = local_gate.detach()
            self.last_local_residual = local_delta.detach()

        components = {
            "global_output": global_output,
            "local_delta": local_delta,
            "local_gate": local_gate,
        }
        
        if return_uncertainty:
            # Log variance for uncertainty estimation (always available)
            log_var = self.uncertainty_layer(h3)
            if local_representation is not None:
                log_var = log_var + local_gate * self.local_uncertainty_layer(
                    local_representation
                )
            if return_components:
                return output, log_var, components
            return output, log_var

        if return_components:
            return output, components
        
        return output
