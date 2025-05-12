import torch
import torch.nn as nn
from typing import Iterable, Optional
from point_e.models.transformer import CLIPImageGridPointDiffusionTransformer, timestep_embedding
from point_e.models.configs import MODEL_CONFIGS
from point_e.diffusion.configs import DIFFUSION_CONFIGS
from point_e.models.pretrained_clip import FrozenImageCLIP
from point_e.models.fusion import TextImageFusionModule
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

class LoRaLinear(nn.Module):
    def __init__(self, linear, rank=8):
        super().__init__()
        self.linear = linear 
        self.rank = rank      # LoRa 
        in_features = linear.in_features
        out_features = linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.normal_(self.lora_A, mean=0, std=1)
        nn.init.zeros_(self.lora_B)
        
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_output = self.linear(x)
        lora_output = x @ self.lora_A @ self.lora_B
        return original_output + lora_output
    
    
    
class Sepe(CLIPImageGridPointDiffusionTransformer):
    """A *much* cleaner Point‑E transformer that fuses CLIP text + image features.

    • One call → CLIP returns both text‑CLS and image patch tokens (plus CLS).
    • Inline fusion (linear concat **or** bidirectional cross‑attention).
    • Forward still matches Point‑E expectations, but we drop every helper / assert.
      At inference just supply `embeddings=` (pre‑cached) or give `images`+`texts`.
    """

#     def __init__(
#         self,
#         *,
#         device: torch.device,
#         dtype: torch.dtype = torch.float32,
#         cache_dir: Optional[str] = None,
#         use_cross_attention: bool = False,
#     ):
#         self.device, self.dtype = device, dtype

#         cfg = MODEL_CONFIGS['base40M'].copy()
#         diff_cfg = DIFFUSION_CONFIGS['base40M']
#         cfg['input_channels'] = 6
#         cfg['width'] = 512
#         cfg.pop('cond_dim', None)
#         cfg['output_channels'] = 6
#         n_ctx, drop_p = cfg.pop('n_ctx'), cfg.pop('cond_drop_prob')
#         cfg.pop('name', None)
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
        use_cross_attention: bool = False,
        use_lora: bool = False,
        lora_rank = 12,
        **cfg,                       # 允许外部覆盖 width / output_channels
    ):
        self.device, self.dtype = device, dtype
        cfg = {**cfg}
        cfg.setdefault("input_channels", 6)
        cfg.setdefault("width", 512)
        cfg.setdefault("output_channels", cfg["input_channels"] * 2)  # learned_range ⇒ C×2
        n_ctx = cfg.pop("n_ctx", 1024)
        drop_p = cfg.pop("cond_drop_prob", 0.1)
        cfg.pop("name", None)
        cfg.pop("cond_dim", None)
        cfg['width'] = 512 
 
        
        super().__init__(
            device=device, dtype=dtype, n_ctx=n_ctx, cond_drop_prob=drop_p,
            frozen_clip=True, cache_dir=cache_dir, **cfg
        )

        # ------- CLIP ------- #
        self.clip = FrozenImageCLIP(device=device, clip_name='ViT-L/14', cache_dir=cache_dir)
        self.clip.model.to(device=device, dtype=torch.float32)
        clip_dim = self.clip.feature_dim  # 768

        # ------- Fusion ------- #
        self.use_cross_attention = use_cross_attention
        fusion_dim = 512 # 512  
        
        self.cond_proj = nn.Sequential(
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )
        #self.dim_fix = (nn.Identity() if fusion_dim == backbone_dim else nn.Linear(fusion_dim, backbone_dim))
        #self.projection = nn.Linear(embeddings.shape[-1], self.backbone.width)
        #self.projection = nn.Linear(256, self.backbone.width).to(device, dtype)
        if use_cross_attention:
            self.fusion = TextImageFusionModule(text_dim=768, image_dim=1024, fusion_dim=512, heads=2)
            
            
            
        else:
            self.fusion = nn.Sequential(
                nn.Linear(1792, fusion_dim),
                nn.GELU(),
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
            )
#         self.ppj = nn.Linear(fusion_dim, 512)
        

        # ------- Train only fusion + last 2 resblocks + output head ------- #
        
        for p in self.parameters():
            p.requires_grad = False

#         for p in self.clip_embed.parameters():
#             p.requires_grad = True

        for p in self.fusion.parameters():
            p.requires_grad = True
#         for blk in self.backbone.resblocks[:]:
#             for p in blk.parameters():
#                 p.requires_grad = False

        for blk in self.backbone.resblocks[-2:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.output_proj.parameters():
            p.requires_grad = True
        for p in self.time_embed.parameters():
            p.requires_grad = False
        for p in self.cond_proj.parameters():
            p.requires_grad = True

            
#         if use_lora:
#             for blk in self.backbone.resblocks:
#                 self.apply_lora_to_layer(blk.attn, rank=lora_rank)  
#                 self.apply_lora_to_layer(blk.mlp, rank=lora_rank)   

#             if use_cross_attention:
#                 self.apply_lora_to_layer(self.fusion, rank=lora_rank)

#             self.apply_lora_to_layer(self.input_proj, rank=lora_rank)
#             self.apply_lora_to_layer(self.output_proj, rank=lora_rank)      
        if use_lora:
            self.apply_lora_to_specific_parts(rank=lora_rank)
            
#         if use_lora and hasattr(self, 'output_proj') and \
#            hasattr(self.output_proj, '0') and isinstance(self.output_proj[0], nn.LayerNorm):
#             for param in self.output_proj[0].parameters():
#                 param.requires_grad = True
##two
#         for param in self.parameters():
#             param.requires_grad = True


#         if hasattr(self, 'clip'):
#             for param in self.clip.model.parameters():
#                 param.requires_grad = False

#         for block in self.backbone.resblocks[:-2]:
#             for param in block.parameters():
#                 param.requires_grad = False

    # ------------------------------------------------------------------ #
    #  Single tiny CLIP helper – returns text‑CLS, image‑CLS, image patches
    # ------------------------------------------------------------------ #
    def _clip(self, images: torch.Tensor, texts: Iterable[str]):
            with torch.no_grad():

                txt_emb = self.clip(batch_size=len(texts), texts=texts).to(self.device, torch.float32)

                clip_model = self.clip.model.clip_model
                image_tensor = images.to(self.device, clip_model.visual.conv1.weight.dtype)
                x = clip_model.visual.conv1(image_tensor)  
                x = x.reshape(x.shape[0], x.shape[1], -1)  
                x = x.permute(0, 2, 1) 
                cls_embedding = clip_model.visual.class_embedding.to(x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
                x = torch.cat([cls_embedding, x], dim=1)  
                x = x + clip_model.visual.positional_embedding.to(x.dtype)
                x = clip_model.visual.ln_pre(x)
                x = x.permute(1, 0, 2) 
                x = clip_model.visual.transformer(x)
                x = clip_model.visual.ln_post(x)
                x = x.permute(1, 0, 2)  # 转换回 [batch, seq_len, dim]

                # 分离 patch tokens 和 class token
                patch_tokens = x[:, 1:, :].to(self.device, torch.float32)
                img_cls = x[:, 0, :].to(self.device, torch.float32)

            return txt_emb, img_cls, patch_tokens
        
    def apply_lora_to_specific_parts(self, rank=8):
        if hasattr(self, 'backbone') and hasattr(self.backbone, 'resblocks'):
            for blk in self.backbone.resblocks:
                if hasattr(blk, 'attn'):
                    self._apply_lora_recursively(blk.attn, rank=rank)
                if hasattr(blk, 'mlp'):
                    self._apply_lora_recursively(blk.mlp, rank=rank)
        
        # 对 input_proj 应用
#         if hasattr(self, 'input_proj'):
#             self._apply_lora_recursively(self.input_proj, rank=rank)
            
#         # 对 output_proj 应用
#         if hasattr(self, 'output_proj'):
#             self._apply_lora_recursively(self.output_proj, rank=rank)



    def _apply_lora_recursively(self, current_module, rank=8):
        for name, child_module in current_module._modules.items():
            if child_module is None:
                continue

            if isinstance(child_module, nn.Linear) and not isinstance(child_module, LoRaLinear):
                lora_version = LoRaLinear(child_module, rank=rank)
                current_module._modules[name] = lora_version
            elif not isinstance(child_module, LoRaLinear): # 重要的：不要进入 LoRaLinear 内部递归
                # 也避免进入其他不应包含可替换线性层的自定义模块，如果适用
                self._apply_lora_recursively(child_module, rank=rank)
                

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        images: Optional[torch.Tensor] = None,
        texts: Optional[Iterable[str]] = None,
        embeddings: Optional[torch.Tensor] = None,
        **__,
    ) -> torch.Tensor:
        # Time embedding
        t_embed = self.time_embed(timestep_embedding(t.to(self.device), self.backbone.width))

        if embeddings is not None:
            cond = [(t_embed, False), (embeddings, True)]
        elif images is not None and texts is not None:
            txt_emb = self.clip.embed_text(texts).to(self.device, self.dtype)
            img_tok = self.clip.embed_images_grid(images).to(self.device, self.dtype)
            img_tok = img_tok.permute(0, 2, 1)
            if self.use_cross_attention:
                fused = self.fusion(txt_emb, img_tok)
            else:
                batch_size = img_tok.shape[0]
                num_patches = img_tok.shape[1]
                text_dim = txt_emb.shape[-1]
                text_emb_expanded = txt_emb.unsqueeze(1).expand(batch_size, num_patches, text_dim)
                concatenated_sequence = torch.cat([text_emb_expanded, img_tok], dim=2)
                fused = self.fusion(concatenated_sequence)
            cond = [(t_embed, False), (fused, True)]
        else:
            raise ValueError("Must provide 'embeddings' or both 'images' and 'texts'")

        return self._forward_with_cond(x.to(self.device, self.dtype), cond)

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.parameters()).device
        texts = model_kwargs["texts"]
        images = model_kwargs["images"]

        with torch.no_grad():

            text_emb = self.clip.embed_text(texts).to(device)
            img_emb = self.clip.embed_images_grid(images).to(device)
            img_emb = img_emb.permute(0, 2, 1)
            if self.use_cross_attention:
                fused_emb = self.fusion(text_emb, img_emb)
            else:
                batch_size = img_emb.shape[0]
                num_patches = img_emb.shape[1]
                text_dim = text_emb.shape[-1]
                text_emb_expanded = text_emb.unsqueeze(1).expand(batch_size, num_patches, text_dim)
                concatenated_sequence = torch.cat([text_emb_expanded, img_emb], dim=2)
                #text_emb = text_emb.unsqueeze(-1)  # [batch_size, text_dim, 1]
                fused_emb = self.fusion(concatenated_sequence)
            print(f"fused_emb: {fused_emb.shape}")
            #fused_emb = fused_emb.mean(dim=2, keepdim=True).expand(-1, -1, self.clip.grid_size**2)

        return dict(embeddings=fused_emb)
