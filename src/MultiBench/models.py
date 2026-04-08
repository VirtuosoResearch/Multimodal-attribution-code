"""Implements common unimodal encoders."""
import torch
import math
from torch import nn

class Linear(torch.nn.Module):
    """Linear Layer with Xavier Initialization, and 0 Bias."""
    
    def __init__(self, indim, outdim, xavier_init=False):
        """Initialize Linear Layer w/ Xavier Init.

        Args:
            indim (int): Input Dimension
            outdim (int): Output Dimension
            xavier_init (bool, optional): Whether to apply Xavier Initialization to Layer. Defaults to False.
        
        """
        super(Linear, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        if xavier_init:
            nn.init.xavier_normal(self.fc.weight)
            self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        """Apply Linear Layer to Input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor
        
        """
        return self.fc(x)



class Transformer(nn.Module):
    """Extends nn.Transformer."""
    
    def __init__(self, n_features, dim, nhead=5, num_layers=5, conv1d=True, out_last=True, pos_embd=False, pos_learnable=False, max_len=128):
        """Initialize Transformer object.

        Args:
            n_features (int): Number of features in the input.
            dim (int): Dimension which to embed upon / Hidden dimension size.
        """
        super().__init__()
        self.embed_dim = dim
        self.conv1d = conv1d
        self.out_last = out_last
        self.pos_embd = pos_embd
        self.pos_learnable = pos_learnable
        self.max_len = max_len
        if self.conv1d:
            self.conv = nn.Conv1d(n_features, self.embed_dim,
                                kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Positional embeddings (optional)
        if self.pos_embd:
            if self.pos_learnable:
                self.pos_embedding = nn.Embedding(self.max_len, self.embed_dim)
            else:
                # fixed sinusoidal table
                position = torch.arange(self.max_len).unsqueeze(1)  # (L,1)
                div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim))
                pe = torch.zeros(self.max_len, self.embed_dim)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pos_table', pe)  # (L, D)

    def forward(self, x):
        """Apply Transformer to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if type(x) is list:
            x = x[0] # (batch_size, seq_len, embed_dim)
        if self.conv1d:
            x = self.conv(x.permute([0, 2, 1])) # (batch_size, embed_dim/channel, seq_len)
            x = x.permute([2, 0, 1]) # (seq_len, batch_size, embed_dim)
        else:
            x = x.permute([1, 0, 2]) # (seq_len, batch_size, embed_dim)
        
        # Add positional embeddings if requested
        if self.pos_embd:
            seq_len = x.size(0)
            if seq_len > self.max_len:
                seq_len = self.max_len
                x = x[:seq_len]
            positions = torch.arange(seq_len, device=x.device)
            if self.pos_learnable:
                pos = self.pos_embedding(positions)  # (seq_len, D)
            else:
                pos = self.pos_table[positions]      # (seq_len, D)
            x = x + pos.unsqueeze(1)  # (seq_len, batch, D)
        
        # Create causal mask for autoregressive modeling
        seq_len = x.size(0)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        if self.out_last:
            return x[-1]
        x = x.permute([1, 0, 2]) # (batch_size, seq_len, embed_dim)
        return x
