import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=3, dropout=0.1):
        """
        dim: Feature dimension
        num_heads: Number of attention heads
        num_layers: Number of Transformer layers
        dropout: Dropout probability
        """
        super(GNNTransformerBlock, self).__init__()

        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            batch_first=True
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Learnable positional encoding to distinguish hierarchical levels
        self.positional_encoding = nn.Parameter(torch.randn(num_layers, dim))

    def forward(self, gnn_outputs):
        """
        gnn_outputs: Multi-layer GNN outputs in shape (batch_size, num_layers, num_nodes, dim)
        """
        # Reshape input: treat num_nodes and num_layers as sequence length
        num_layers, num_nodes, dim = gnn_outputs.size()
        batch_size = 1
        gnn_outputs = gnn_outputs.view(batch_size * num_nodes, num_layers, dim)

        # Add positional encoding to each GNN layer's output to help Transformer distinguish layers
        gnn_outputs += self.positional_encoding.unsqueeze(0).repeat(batch_size * num_nodes, 1, 1)

        # Split into 3 chunks and process separately
        chunk_num = 10
        chunk_size = gnn_outputs.size(0) // chunk_num
        remainder = gnn_outputs.size(0) % chunk_num

        # Calculate chunk boundaries
        chunks = []
        start_idx = 0

        for i in range(chunk_num):
            # Add remainder to first chunks if needed
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
        
            # Process chunk through transformer
            chunk_input = gnn_outputs[start_idx:end_idx]
            chunk_output = self.transformer_encoder(chunk_input)
            chunk_final = chunk_output.mean(dim=1)
        
            chunks.append(chunk_final)
            start_idx = end_idx

        # Concatenate all chunks back together
        final_output = torch.cat(chunks, dim=0)
        final_output = final_output.view(num_nodes, dim)
        return final_output

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "Embedding dimension must be divisible by number of heads"

        # Linear projections for Query, Key, and Value
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        # Final linear layer
        self.out_proj = nn.Linear(dim, dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Shapes of query, key, and value: [batch_size, feature_size]
        """
        batch_size, feature_size = query.size()

        # Add sequence length dimension: [batch_size, feature_size] → [batch_size, 1, feature_size]
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        # Linear projections into multi-head dimensions
        query = self.query_proj(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32, device=query.device))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of values using attention probabilities
        context = torch.matmul(attention_probs, value)

        # Concatenate multi-head results
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)
        output = self.out_proj(context).squeeze(1)
        return output


