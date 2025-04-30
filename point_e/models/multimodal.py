import torch
import torch.nn as nn
from typing import Optional, Iterable, Dict, Any
from PIL import Image

from point_e.models.configs         import MODEL_CONFIGS
from point_e.models.fusion import TextImageFusionModule
from point_e.models.transformer     import CLIPImageGridPointDiffusionTransformer, timestep_embedding
from point_e.models.pretrained_clip import FrozenImageCLIP, ImageType

class SimpleMultimodalTransformer(CLIPImageGridPointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cache_dir: Optional[str] = None,
        use_cross_attention: bool = False,
    ):
        # 1) 准备父类需要的配置
        cfg     = MODEL_CONFIGS['base40M'].copy()
        cond_dp = cfg.pop('cond_drop_prob')
        n_ctx   = cfg.pop('n_ctx')
        cfg.pop('name', None)

        super().__init__(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            cond_drop_prob=cond_dp,
            frozen_clip=True,
            cache_dir=cache_dir,
            **cfg,    # input_channels, output_channels, layers, heads, init_scale, time_token_cond
        )

        # 2) 冻结 CLIP
        self.clip = FrozenImageCLIP(device=device, clip_name="ViT-L/14", cache_dir=cache_dir)

        # fusion module

        # 3) 简单 concat->linear 融合
        if  use_cross_attention == False:
            fusion_in  = self.clip.feature_dim + self.clip.grid_feature_dim
            fusion_out = self.backbone.width
            self.fusion = nn.Linear(fusion_in, fusion_out)
        else:
            self.fusion = TextImageFusionModule(
            text_dim=self.clip.feature_dim,        # 768
            image_dim=self.clip.grid_feature_dim,   # e.g. 256
            fusion_dim=self.backbone.width,         # 512
            use_cross_attention=use_cross_attention,
            heads=8
        )
        for p in self.parameters():
            p.requires_grad = False
        for p in self.fusion.parameters():
            p.requires_grad = True
        for p in self.clip_embed.parameters():
            p.requires_grad = True

    def cached_model_kwargs(
        self,
        batch_size: int,
        model_kwargs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        # 预先计算一次融合向量，采样每步直接复用
        images = model_kwargs['images']  # List[PIL.Image]
        texts  = model_kwargs['texts']   # List[str]

        grid    = self.clip.embed_images_grid(images)    # [B, C_img, N]
        grid = grid[:, :, 1:]           # 丢掉 CLS（第一 token）
        grid = grid.permute(0,2,1).contiguous()
        
        img_emb = grid.mean(dim=2)                       # [B, C_img]
        txt_emb = self.clip(batch_size=batch_size, texts=texts)  # [B, C_txt]
        fused   = self.fusion(torch.cat([txt_emb, img_emb], dim=1))  # [B, fusion_out]
        return {'embeddings': fused}

    def forward(
        self,
        x: torch.Tensor,                              # [B, C_in, n_ctx]
        t: torch.Tensor,                              # [B]
        images:    Optional[Iterable[ImageType]] = None,
        texts:     Optional[Iterable[str]]      = None,
        embeddings:Optional[torch.Tensor]       = None,
        **kwargs,  # 收下 sampler 其它可能传入的参数
    ) -> torch.Tensor:
        B = x.shape[0]
        # 时间 embed
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        # 直接用 sampler 给的 fused embeddings，跳过重复编码
        if embeddings is not None:
            fused = embeddings
        else:
            assert images is not None and texts is not None
            grid    = self.clip.embed_images_grid(images)
            img_emb = grid.mean(dim=2)
            txt_emb = self.clip(batch_size=B, texts=texts)
            fused   = self.fusion(torch.cat([txt_emb, img_emb], dim=1))

        cond = [
            (t_embed, self.time_token_cond),
            (fused,   True),
        ]
        return self._forward_with_cond(x, cond)
