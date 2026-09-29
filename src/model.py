import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta

class Block(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.WQ = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.WK = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.WV = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.WO = nn.Parameter(torch.randn(d_model, d_model) * 0.1)

        self.gamma1 = nn.Parameter(torch.ones(d_model))
        self.beta1 = nn.Parameter(torch.zeros(d_model))
        self.gamma2 = nn.Parameter(torch.ones(d_model))
        self.beta2 = nn.Parameter(torch.zeros(d_model))

        self.W1 = nn.Parameter(torch.randn(d_model, d_ff) * 0.1)
        self.bias1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = nn.Parameter(torch.randn(d_ff, d_model) * 0.1)
        self.bias2 = nn.Parameter(torch.zeros(d_model))

    def masked_self_attention(self, x, causal_mask):
        # x is (B, T, d_model)
        B, T, _ = x.shape
        Q = x @ self.WQ
        K = x @ self.WK
        V = x @ self.WV

        # (B, T, n_heads, d_k) -> (B, n_heads, T, d_k)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # scores: (B, n_heads, T, T)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # causal_mask: (T, T)
        mask = causal_mask[:T, :T].to(scores.device)
        scores = scores.masked_fill(mask.view(1, 1, T, T), float('-inf'))
        attn_w = torch.softmax(scores, dim=-1)

        out = attn_w @ V
        # (B, n_heads, T, d_k) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = out @ self.WO
        return out, attn_w

    def forward(self, x, causal_mask):
        att_out, _ = self.masked_self_attention(x, causal_mask)
        x2 = layer_norm(x + att_out, self.gamma1, self.beta1)

        ff_out = torch.relu(x2 @ self.W1 + self.bias1)
        ff_out = ff_out @ self.W2 + self.bias2
        x3 = layer_norm(x2 + ff_out, self.gamma2, self.beta2)
        return x3

class MiniGPT(nn.Module):
    def __init__(self, V, d_model, n_heads, n_layers, block_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.V = V
        d_ff = d_model * 4

        self.E = nn.Parameter(torch.randn(V, d_model) * 0.1)

        pe = torch.zeros(block_size, d_model)
        pos = torch.arange(block_size).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

        causal_mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask)

        self.blocks = nn.ModuleList([Block(d_model, d_ff, n_heads) for _ in range(n_layers)])

        self.Wlm = nn.Parameter(torch.randn(d_model, V) * 0.1)
        self.blm = nn.Parameter(torch.zeros(V))

    def forward(self, idxs, targets=None):
        B, T = idxs.shape
        x = self.E[idxs] + self.pe[:T]

        for block in self.blocks:
            x = block(x, self.causal_mask)

        logits = x @ self.Wlm + self.blm

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.V), targets.view(-1))
            return logits, loss
        return logits
