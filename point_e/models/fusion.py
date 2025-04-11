# module helps text image fusion
import torch
import torch.nn as nn

class TextImageFusionModule(nn.Module):
    def __init__(self, text_dim=768, image_dim=1024, fusion_dim=1024):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, text_emb, image_emb):
        text_proj = self.text_proj(text_emb)
        image_proj = self.image_proj(image_emb)
        
        fused = torch.cat([text_proj, image_proj], dim=-1)
        return self.fusion(fused)


# attention definition here
class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        # Multi-head attention components
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
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
        context = self.norm1(context)
        
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        x = self.attention(q, k, v) + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        
        return x
    
class TextImageFusionModule(nn.Module):
    def __init__(self, text_dim=768, image_dim=1024, fusion_dim=1024, use_cross_attention=True):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.use_cross_attention = use_cross_attention
        
        if use_cross_attention:
            # Option 2: Cross-attention between text and image
            self.text_to_image_attn = CrossAttentionLayer(fusion_dim)
            self.image_to_text_attn = CrossAttentionLayer(fusion_dim)
            self.final_proj = nn.Linear(fusion_dim * 2, fusion_dim)
        else:
            # Option 1: Simple concatenation
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Linear(fusion_dim, fusion_dim)
            )
    
    def forward(self, text_emb, image_emb):
        text_proj = self.text_proj(text_emb)
        image_proj = self.image_proj(image_emb)
        
        if self.use_cross_attention:
            # Add batch dimension if needed for single embeddings
            if text_proj.dim() == 1:
                text_proj = text_proj.unsqueeze(0)
            if image_proj.dim() == 1:
                image_proj = image_proj.unsqueeze(0)
                
            if text_proj.dim() == 2:
                text_proj = text_proj.unsqueeze(1)
            if image_proj.dim() == 2:
                image_proj = image_proj.unsqueeze(1)
            
            # Cross-attention in both directions
            text_attended = self.text_to_image_attn(text_proj, image_proj)
            image_attended = self.image_to_text_attn(image_proj, text_proj)
            
            # Combine attended features
            batch_size = text_proj.shape[0]
            text_attended = text_attended.view(batch_size, -1)
            image_attended = image_attended.view(batch_size, -1)
            
            combined = torch.cat([text_attended, image_attended], dim=-1)
            return self.final_proj(combined)
        else:
            # Simple concatenation
            fused = torch.cat([text_proj, image_proj], dim=-1)
            return self.fusion(fused)