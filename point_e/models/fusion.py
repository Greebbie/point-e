import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    """
    Standard Cross-Attention layer followed by a Feed-Forward Network.
    Query is derived from input 'x', Key/Value are derived from 'context'.
    Includes Layer Normalization and Residual Connections.
    """
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.head_dim = dim // heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def attention(self, q, k, v):
        batch_size = q.shape[0]
        q = q.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.out_proj(out)

    def forward(self, x, context):
        residual = x
        x_norm = self.norm1(x)
        context_norm = self.context_norm(context)
        q = self.q_proj(x_norm)
        k = self.k_proj(context_norm)
        v = self.v_proj(context_norm)
        x = self.attention(q, k, v) + residual
        residual = x
        x_norm = self.norm2(x)
        x = self.ffn(x_norm) + residual
        return x

class TextImageFusionModule(nn.Module):
    """
    Fuses text and image embeddings using bidirectional cross-attention.
    """
    def __init__(self, text_dim=768, image_dim=768, fusion_dim=768, heads=8):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim) # 1024 ok proj already
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.t2i = CrossAttentionLayer(fusion_dim, heads)
        self.i2t = CrossAttentionLayer(fusion_dim, heads)
        self.final = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, text_emb, img_tokens):
        """
        Args:
            text_emb (torch.Tensor): [B, text_dim]
            img_tokens (torch.Tensor): [B, N, image_dim]
        Returns:
            torch.Tensor: [B, fusion_dim]
        """
        txt = self.text_proj(text_emb).unsqueeze(1)        # [B, 1, F]
        img = self.image_proj(img_tokens)                  # [B, N, F]
        q_txt = self.t2i(txt, img).squeeze(1)              # [B, F]
        img2t = self.i2t(img, txt).mean(dim=1)             # [B, F]
        fused_output = self.final((q_txt + img2t) * 0.5)   # [B, F]
        return fused_output