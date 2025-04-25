
import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.head_dim = dim // heads
        
        # Multi-head attention components
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
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
        x = self.norm1(x)
        context = self.context_norm(context)
        
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        x = self.attention(q, k, v) + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        
        return x

class TextImageFusionModule(nn.Module):
    def __init__(self, 
                 clip_dim=768, 
                 fusion_dim=512, 
                 use_cross_attention=True, 
                 heads=8):
        super().__init__()
        self.text_proj = nn.Linear(clip_dim, fusion_dim)
        self.image_proj = nn.Linear(clip_dim, fusion_dim)  # works on (B,N,clip_dim)
        
        if use_cross_attention:
            self.t2i = CrossAttentionLayer(fusion_dim, heads)
            self.i2t = CrossAttentionLayer(fusion_dim, heads)
            self.final = nn.Linear(fusion_dim, fusion_dim)
        else:
            self.fuse = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, fusion_dim),
            )
    
    def forward(self, text_emb, img_tokens):
        """
        text_emb   : [B, 768]
        img_tokens : [B, N, 768]  (N ≈ 196 for 224px ViT-L/14)
        """
        txt = self.text_proj(text_emb).unsqueeze(1)        # [B,1,F]
        img = self.image_proj(img_tokens)                  # [B,N,F]
        
        if hasattr(self, "t2i"):
            q_txt = self.t2i(txt, img).squeeze(1)          # [B,F]
            img2t = self.i2t(img, txt).mean(dim=1)         # [B,F]
            return self.final((q_txt + img2t) * 0.5)       # [B,F]
        else:
            pooled = img.mean(dim=1)                       # [B,F]
            return self.fuse(torch.cat([txt.squeeze(1), pooled], dim=-1))