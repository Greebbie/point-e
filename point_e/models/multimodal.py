import torch
import torch.nn as nn
from typing import Optional, Iterable, Dict, Any, Union
from PIL import Image

# Assuming these imports point to the correct Point-E library structure
# If TextImageFusionModule is in the same directory, use:
# from fusion import TextImageFusionModule
# Otherwise, adjust the import path as needed:
from point_e.models.fusion import TextImageFusionModule
from point_e.models.configs import MODEL_CONFIGS
from point_e.models.transformer import CLIPImageGridPointDiffusionTransformer, timestep_embedding
from point_e.models.pretrained_clip import FrozenImageCLIP, ImageType
class SimpleMultimodalTransformer(CLIPImageGridPointDiffusionTransformer):
    """
    Point-E transformer modified for text and image conditioning using
    either simple linear fusion or cross-attention fusion.
    """
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cache_dir: Optional[str] = None,
        use_cross_attention: bool = False,
    ):
        # **FIX:** Store device and dtype as instance attributes *before* calling super().__init__
        # This ensures self.device is available if needed by methods called during super().__init__
        # or if the parent doesn't store it reliably.
        self.device = device
        self.dtype = dtype
    
        # Get the base config first to check input_channels if needed
        cfg = MODEL_CONFIGS['base40M'].copy()
        cfg['output_channels'] = 6
        self.expected_input_channels = cfg.get('input_channels', 6) # Default to 6 based on common use (XYZ+RGB)
        print(f"Model config expects input_channels: {self.expected_input_channels}")

        cond_dp = cfg.pop('cond_drop_prob')
        n_ctx = cfg.pop('n_ctx')
        cfg.pop('name', None)

        # Initialize parent class, passing the full config including input_channels
        # The parent class *should* also handle device/dtype, but we store it explicitly too.
        super().__init__(
            device=self.device, # Pass stored device
            dtype=self.dtype,   # Pass stored dtype
            n_ctx=n_ctx,
            cond_drop_prob=cond_dp,
            frozen_clip=True,
            cache_dir=cache_dir,
            **cfg,
        )
        self.cond_proj = nn.Linear(1024, self.backbone.width)
        self.text_proj = nn.Linear(768, 1024)  
        # Use try-except for FrozenImageCLIP in case point-e wasn't fully imported
        try:
            # Pass the explicitly stored device/dtype to FrozenImageCLIP as well
            self.clip = FrozenImageCLIP(device=self.device, clip_name="ViT-L/14", cache_dir=cache_dir)
            # Ensure the CLIP model itself is on the correct device/dtype
            self.clip.model.to(device=self.device, dtype=self.dtype)
            self.text_feature_dim = self.clip.feature_dim
            self.image_grid_feature_dim = 1024 # ViT-L/14 patch feature dim
        except NameError:
             print("Warning: FrozenImageCLIP not available. Cannot initialize CLIP.")
             self.clip = None # Set clip to None if it couldn't be initialized
             self.text_feature_dim = 768 # Default value
             self.image_grid_feature_dim = 1024 # Default value


        self.use_cross_attention = use_cross_attention

        if self.use_cross_attention:
            print("Initializing with TextImageFusionModule (Cross-Attention)")
            self.fusion = TextImageFusionModule(
                text_dim=1024, # just hard code it for now
                image_dim=self.image_grid_feature_dim,
                fusion_dim=self.backbone.width, # Use backbone width from parent
                heads=8
            )
        else:
            print("Initializing with simple Linear fusion")
            fusion_in_dim = self.text_feature_dim + self.image_grid_feature_dim
            fusion_out_dim = self.backbone.width # Use backbone width from parent
            self.fusion = nn.Linear(fusion_in_dim, fusion_out_dim)

        # Move fusion module to the correct device/dtype
        self.fusion.to(device=self.device, dtype=self.dtype)

        # Freeze parameters selectively
        for p in self.parameters():
            p.requires_grad = False
        for p in self.fusion.parameters():
            p.requires_grad = True
        # Unfreeze the final projection layer in the backbone if it exists
        if hasattr(self, 'clip_embed'):
             for p in self.clip_embed.parameters():
                 p.requires_grad = True
        else:
             # Check for the output projection layer name used in base40M config
             if hasattr(self, 'output_proj'):
                 print("Unfreezing self.output_proj parameters.")
                 for p in self.output_proj.parameters():
                     p.requires_grad = True
             else:
                 print("Warning: Neither self.clip_embed nor self.output_proj found. Cannot unfreeze output layer.")


        self.eval() # Set to evaluation mode by default


    def _get_clip_embeddings(
        self, batch_size: int, images: Optional[torch.Tensor] = None,
        texts: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        获取文本嵌入和图像嵌入。
        期望 'images' 是一个预处理过的张量批次。
        """
        if self.clip is None:
            raise RuntimeError("CLIP 模型未成功初始化。")
        if not hasattr(self.clip, 'model') or not hasattr(self.clip.model, 'clip_model'):
            raise RuntimeError("CLIP 模型结构不符合预期（缺少 self.clip.model.clip_model）。")

        outputs = {}
        self.clip.model.to(self.device)
        self.clip.model.clip_model.to(self.device)

        with torch.no_grad():
            # 获取文本嵌入
            if texts is not None:
                outputs['text_emb'] = self.clip(batch_size=batch_size, texts=texts).to(self.device, self.dtype)

            # 获取图像嵌入
            if images is not None:
                if not isinstance(images, torch.Tensor):
                    raise TypeError(f"_get_clip_embeddings 期望 'images' 是预处理过的张量，但得到 {type(images)}")

                image_tensor = images.to(self.device, self.dtype)
                visual_output = self.clip.model.clip_model.visual(image_tensor.to(self.clip.model.clip_model.dtype))

                # 根据输出维度处理
                if visual_output.ndim == 2:
                    # 输出为 [B, dim]，直接使用
                    outputs['img_emb'] = visual_output.to(self.device, self.dtype)
                elif visual_output.ndim == 3:
                    # 输出为 [B, num_tokens, dim]，提取 patch tokens
                    img_tokens = visual_output[:, 1:, :]  # 排除 CLS token
                    outputs['img_tokens'] = img_tokens.to(self.device, self.dtype)
                else:
                    raise ValueError(f"视觉模型输出形状异常：{visual_output.shape}")

        return outputs

    # def _get_clip_embeddings(
    #     self, batch_size: int, images: Optional[torch.Tensor] = None,
    #     texts: Optional[Iterable[str]] = None,
    # ) -> Dict[str, torch.Tensor]:
    #     if self.clip is None:
    #         raise RuntimeError("CLIP 模型未成功初始化。")
        
    #     outputs = {}
    #     with torch.no_grad():
    #         # 获取文本嵌入
    #         if texts is not None:
    #             outputs['text_emb'] = self.clip(batch_size=batch_size, texts=texts).to(self.device, self.dtype)

    #         # 获取图像嵌入（使用 patch embeddings）
    #         if images is not None:
    #             # image_tensor = images.to(self.device, self.dtype)
    #             # visual_output = self.clip.model.clip_model.visual(image_tensor.to(self.clip.model.clip_model.dtype))
                
    #             # print(f"Visual output shape: {visual_output.shape}")
    #             # if visual_output.ndim == 3:
    #             #     # 提取 patch tokens，去掉 CLS token，形状为 [B, num_patches, 1024]
    #             #     img_tokens = visual_output[:, 1:, :]
    #             #     outputs['img_tokens'] = img_tokens.to(self.device, self.dtype)
    #             # else:
    #             #     raise ValueError("视觉模型输出形状异常")
    #             grid = self.clip.embed_images_grid(images)        

    #             # 2) 取最后一层、去掉 CLS，只保留 patch token
    #             img_tokens = grid[:, -1, 1:, :]                  # [B, num_patches, 1024]
    #             # 3) 存到 outputs
    #             outputs['img_tokens'] = img_tokens.to(self.device, self.dtype)

    #     return outputs

    def _compute_fused_embedding(
        self, text_emb: Optional[torch.Tensor] = None, img_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.fusion.to(self.device)
        if self.use_cross_attention:
            assert text_emb is not None and img_features is not None
            text_emb = text_emb.to(self.device, self.dtype)
            img_features = img_features.to(self.device, self.dtype)
            # 如果是 [B, dim]，增加一个维度到 [B, 1, dim]
            if img_features.ndim == 2:
                img_features = img_features.unsqueeze(1)
            fused = self.fusion(text_emb=text_emb, img_tokens=img_features)
        else:
            assert text_emb is not None and img_features is not None
            text_emb = text_emb.to(self.device, self.dtype)
            img_features = img_features.to(self.device, self.dtype)
            # 如果是 [B, N, dim]，取平均值；如果是 [B, dim]，直接使用
            if img_features.ndim == 3:
                img_emb_avg = img_features.mean(dim=1)
            else:
                img_emb_avg = img_features
            concat_emb = torch.cat([text_emb, img_emb_avg], dim=1)
            fused = self.fusion(concat_emb)
        return fused.to(self.device, self.dtype)

    def cached_model_kwargs(
        self, batch_size: int, model_kwargs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        images = model_kwargs.get('images')
        texts = model_kwargs.get('texts')
        if images is None or texts is None:
             raise ValueError("Both 'images' and 'texts' must be provided in model_kwargs")

        clip_embeddings = self._get_clip_embeddings(batch_size, images, texts)
        text_emb = clip_embeddings.get('text_emb')
        img_tokens = clip_embeddings.get('img_tokens')
        fused_embedding = self._compute_fused_embedding(text_emb, img_tokens)
        return {'embeddings': fused_embedding}

    def forward(
        self, x: torch.Tensor, t: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None, texts: Optional[Iterable[str]] = None,
        embeddings: Optional[torch.Tensor] = None, **kwargs,
    ) -> torch.Tensor:
        B = x.shape[0]
        x = x.to(self.device)
        t = t.to(self.device)
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width)).to(self.device, self.dtype)

        if embeddings is not None:
            fused = embeddings.to(self.device, self.dtype)
        else:
            assert images is not None and texts is not None
            clip_embeddings = self._get_clip_embeddings(B, images, texts)
            text_emb = clip_embeddings.get('text_emb')
            text_emb_1024 = self.text_proj(text_emb)
            text_emb_proj = self.cond_proj(text_emb_1024)
            # 优先获取 'img_tokens'，没有则使用 'img_emb'
            img_features = clip_embeddings.get('img_tokens', clip_embeddings.get('img_emb'))

            if img_features is None:
                raise ValueError("clip_embeddings 中未找到图像特征")
            fused = self._compute_fused_embedding(text_emb_1024, img_features)

        cond = [(text_emb_proj, self.time_token_cond), (fused, True)]
        output = self._forward_with_cond(x, cond)
        return output