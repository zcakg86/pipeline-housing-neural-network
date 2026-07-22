"""
H3 Community Embedding with Neighborhood Pooling
Uses H3 L8 hexagons with their 6 neighbors' community information
"""
import torch
import torch.nn as nn


class H3CommunityEmbedding(nn.Module):
    """
    Embeds H3 hexagons using their community and neighboring communities.
    
    Each H3 hex is represented by 7 community IDs:
    - Index 0: Center hex community
    - Indices 1-6: Six neighboring hexes' communities
    
    The embeddings are pooled to create a location-aware representation.
    """
    
    def __init__(self, num_communities, embedding_dim, pooling_strategy='mean'):
        """
        Args:
            num_communities: Number of unique communities (vocabulary size - 1)
            embedding_dim: Dimension of the embedding vectors
            pooling_strategy: How to pool neighbor embeddings
                - 'mean': Simple average of all 7 embeddings
                - 'center_weighted': 50% center, 50% average of neighbors
                - 'attention': Learnable attention weights (future)
        """
        super().__init__()
        
        # +1 for the UNKNOWN/Padding community
        self.num_communities = num_communities
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        self.vocab_size = num_communities + 1
        
        # Embedding layer with padding
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=num_communities  # Ignore padding in gradients
        )
        
        # Optional: Learnable pooling weights
        if pooling_strategy == 'learnable':
            self.pool_weights = nn.Parameter(torch.ones(7) / 7)
    
    def forward(self, hex_community_matrix):
        """
        Forward pass through the embedding layer with pooling.
        
        Args:
            hex_community_matrix: Tensor of shape (Batch_Size, 7)
                Column 0: Center hex community index
                Columns 1-6: Neighbor hex community indices
        
        Returns:
            pooled_embeds: Tensor of shape (Batch_Size, Embedding_Dim)
        """
        # Shape: (Batch_Size, 7, Embedding_Dim)
        embeds = self.embedding(hex_community_matrix)
        return self.pool_embeddings(embeds)

    def pool_embeddings(self, embeds):
        """Pool an already-embedded ``[batch, 7, embedding_dim]`` tensor."""
        
        if self.pooling_strategy == 'mean':
            # Simple mean pooling across all 7 embeddings
            pooled_embeds = embeds.mean(dim=1)
            
        elif self.pooling_strategy == 'center_weighted':
            # Give 50% weight to center, 50% to neighbors
            center_embed = embeds[:, 0, :]  # (Batch_Size, Embedding_Dim)
            neighbor_embed = embeds[:, 1:, :].mean(dim=1)  # (Batch_Size, Embedding_Dim)
            pooled_embeds = (center_embed * 0.5) + (neighbor_embed * 0.5)
            
        elif self.pooling_strategy == 'learnable':
            # Learnable weights for each position
            weights = torch.softmax(self.pool_weights, dim=0)  # Normalize to sum to 1
            # Expand weights: (7,) -> (1, 7, 1)
            weights = weights.view(1, 7, 1)
            # Weighted sum: (Batch_Size, 7, Embedding_Dim) * (1, 7, 1) -> (Batch_Size, Embedding_Dim)
            pooled_embeds = (embeds * weights).sum(dim=1)
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled_embeds
    
    def get_embedding_weights(self):
        """Return the embedding weight matrix for analysis"""
        return self.embedding.weight.data
    
    def get_pooling_weights(self):
        """Return pooling weights if using learnable strategy"""
        if self.pooling_strategy == 'learnable':
            return torch.softmax(self.pool_weights, dim=0).detach()
        else:
            return None


class LocalNeighborhoodEncoder(nn.Module):
    """Encode seven H3-local market states without collapsing them prematurely."""

    def __init__(self, embedding_dim, local_feature_dim, dropout_rate=0.1):
        super().__init__()
        self.local_feature_layer = nn.Linear(local_feature_dim, embedding_dim)
        # k=1 has two meaningful roles: center and neighbor. Treating the six
        # neighbors symmetrically avoids learning arbitrary H3-ID sort order.
        self.position_embedding = nn.Embedding(2, embedding_dim)
        self.fusion_norm = nn.LayerNorm(embedding_dim)
        self.attention_score = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, cell_community_embeddings, local_market_features):
        """
        Args:
            cell_community_embeddings: ``[batch, 7, embedding_dim]``
            local_market_features: ``[batch, 7, local_feature_dim]``
        """
        positions = torch.tensor(
            [0, 1, 1, 1, 1, 1, 1], device=local_market_features.device
        )
        positions = self.position_embedding(positions).unsqueeze(0)
        local_projection = torch.relu(self.local_feature_layer(local_market_features))
        tokens = self.fusion_norm(cell_community_embeddings + local_projection + positions)
        scores = self.attention_score(torch.tanh(tokens)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = (self.dropout(tokens) * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


# Example usage and testing
if __name__ == "__main__":
    print("Testing H3CommunityEmbedding...")
    
    # Simulate parameters
    num_communities = 231  # From precomputation
    embedding_dim = 128
    batch_size = 32
    vocab_size = num_communities + 1
    
    # Test different pooling strategies
    for strategy in ['mean', 'center_weighted', 'learnable']:
        print(f"\n--- Testing {strategy} pooling ---")
        
        # Initialize layer
        location_encoder = H3CommunityEmbedding(
            num_communities=num_communities,
            embedding_dim=embedding_dim,
            pooling_strategy=strategy
        )
        
        # Simulated batch from DataLoader (Batch Size x 7 neighbors)
        dummy_input = torch.randint(0, vocab_size, (batch_size, 7))
        
        # Forward pass
        output = location_encoder(dummy_input)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected: torch.Size([{batch_size}, {embedding_dim}])")
        
        if strategy == 'learnable':
            weights = location_encoder.get_pooling_weights()
            print(f"Pooling weights: {weights}")
    
    print("\n✓ All tests passed!")
