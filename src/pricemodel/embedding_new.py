import torch
import torch.nn as nn

class EmbeddingModelEnhanced(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, property_dim,
                 community_embedding_length, community_feature_dim,
                 year_length, week_length):
        """Enhanced model with proper attention layer initialization and weight tracking"""
        super().__init__()

        # Layer dims
        self.property_dim = property_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Embedding dim
        self.community_embedding_length = community_embedding_length
        self.community_feature_dim = community_feature_dim
        self.week_length = week_length
        self.year_length = year_length

        # Embedding Layers
        self.community_embedding = nn.Embedding(int(community_embedding_length), embedding_dim)
        self.year_embedding = nn.Embedding(int(year_length), embedding_dim)
        self.week_embedding = nn.Embedding(int(week_length), embedding_dim)
        # Feature Processing Layers
        self.community_feature_layer = nn.Linear(community_feature_dim, hidden_dim)
        self.property_feature_layer = nn.Linear(property_dim, embedding_dim)

        # # Calculate combined embedding dimension
        # self.combined_embedding_dim = 3 * embedding_dim
        # self.embed_dim_attention = 1 * hidden_dim + self.combined_embedding_dim

        # Initialize attention layer
        # Need all stacked layers in attention layer to have equal dims, including feature layer...
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=2,
            batch_first=True)

        # Hidden and Output Layers
        self.hidden_layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

        # Storage for attention weights and intermediate outputs
        self.last_attention_weights = None
        self.intermediate_outputs = {}
        self.save_intermediates = False

    def forward(self, community_indices, community_features, year, week, property_features, targets=None):
        # Clear previous intermediate outputs
        if self.save_intermediates:
            self.intermediate_outputs = {}

        # Embeddings
        community_embeddings = self.community_embedding(community_indices)
        year_embeddings = self.year_embedding(year)
        week_embeddings = self.week_embedding(week)
        combined_embeddings = torch.cat([community_embeddings, year_embeddings, week_embeddings], dim=-1)
                                  
        if self.save_intermediates:
            self.intermediate_outputs['community_embeddings'] = community_embeddings.detach()
            self.intermediate_outputs['year_embeddings'] = year_embeddings.detach()
            self.intermediate_outputs['week_embeddings'] = week_embeddings.detach()

        # Feature Processing
        #processed_community_features = self.relu(self.community_feature_layer(community_features))
        processed_property_features = self.relu(self.property_feature_layer(property_features))

        if self.save_intermediates:
        #    self.intermediate_outputs['processed_community_features'] = processed_community_features.detach()
            self.intermediate_outputs['processed_property_features'] = processed_property_features.detach()
        # Cm out to try stacking for attention weights
        # # Reshape for attention
        # combined_features = combined_features.unsqueeze(1)
        # # print(f'Combined features shape: {combined_features.shape}')
        # # Combine embeddings and features
        # combined_features = torch.cat([combined_embeddings, #processed_community_features,
        #                                processed_property_features],
        #                                dim=-1)
        # attention_output = attention_output.squeeze(1)
        # # Attention Layer with weight extraction
        # attention_output, attention_weights = self.attention_layer(
        #     comings_features, comings_features, comings_features,
        #     need_weights=True, average_attn_weights=False
        # )
        # hidden1 = self.relu(self.hidden_layer1(attention_output))



        # Stack embedding and property layers
        tokens = torch.stack([community_embeddings, year_embeddings, week_embeddings, processed_property_features],
                             dim = 1)

        # Attention Layer with weight extraction
        attention_output, attention_weights = self.attention_layer(
            tokens, tokens, tokens,
            need_weights=True, average_attn_weights=False
        )

        # Store attention weights
        self.last_attention_weights = attention_weights.detach()

        # print(f'self.save_intermediates = {self.save_intermediates}')
        if self.save_intermediates:
            self.intermediate_outputs['attention_output'] = attention_output.detach()
            self.intermediate_outputs['attention_weights'] = attention_weights.detach()
            
        pooled = attention_output.mean(dim=1) 

        # Hidden Layers
    
        hidden1 = self.relu(self.hidden_layer1(pooled))
        hidden2 = self.relu(self.hidden_layer2(hidden1))

        if self.save_intermediates:
            self.intermediate_outputs['hidden1'] = hidden1.detach()
            self.intermediate_outputs['hidden2'] = hidden2.detach()

        # Output Layer
        output = self.output_layer(hidden2)

        return output
