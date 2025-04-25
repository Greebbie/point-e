import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Iterable, Union

from point_e.models.pretrained_clip import ImageType
from point_e.models.transformer import (
    CLIPImageGridPointDiffusionTransformer,
    timestep_embedding
)
from point_e.models.fusion import TextImageFusionModule


class MultimodalPointDiffusionTransformer(CLIPImageGridPointDiffusionTransformer):
    """
    Extension of Point-E to support multimodal conditioning with both text and image.
    """
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        cond_drop_prob: float = 0.1,
        frozen_clip: bool = True,
        frozen_transformer: bool = True,  # Freeze most model parameters
        use_cross_attention: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            cond_drop_prob=cond_drop_prob,
            frozen_clip=frozen_clip,
            cache_dir=cache_dir,
            **kwargs
        )
        
        # Create fusion module
        self.fusion_module = TextImageFusionModule(
            text_dim=self.clip.feature_dim,      # CLIP text embedding dim
            image_dim=self.clip.grid_feature_dim,  # CLIP image grid feature dim
            fusion_dim=self.backbone.width,      # Point-E transformer width
            use_cross_attention=use_cross_attention
        )
        
        # If we're freezing the transformer backbone
        if frozen_transformer:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
            
            # Unfreeze only the fusion module and the conditional projection
            for param in self.fusion_module.parameters():
                param.requires_grad = True
            for param in self.clip_embed.parameters():
                param.requires_grad = True

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Precompute embeddings for the batch to reuse during sampling.
        """
        with torch.no_grad():
            # Get image embeddings
            image_embeddings = self.clip.embed_images_grid(model_kwargs["images"])
            
            # Get text embeddings
            text_embeddings = self.clip(batch_size, texts=model_kwargs["texts"])
            
            return dict(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
    ):
        """Forward pass with both text and image conditioning."""
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        
        # Get image embeddings
        if images is not None:
            # Average pool grid features to get single vectors per image
            image_emb = self.clip.embed_images_grid(images)
            image_emb = image_emb.mean(dim=2)  # Average across grid positions
        elif image_embeddings is not None:
            image_emb = image_embeddings
            if image_emb.dim() > 2:  # If grid features
                image_emb = image_emb.mean(dim=2)
        else:
            # Default to zeros if no image conditioning
            image_emb = torch.zeros(
                [len(x), self.clip.grid_feature_dim],
                dtype=x.dtype,
                device=x.device,
            )
            
        # Get text embeddings
        if texts is not None:
            text_emb = self.clip(batch_size=len(x), texts=texts)
        elif text_embeddings is not None:
            text_emb = text_embeddings
        else:
            # Default to zeros if no text conditioning
            text_emb = torch.zeros(
                [len(x), self.clip.feature_dim],
                dtype=x.dtype,
                device=x.device,
            )

        # Apply dropout during training if specified
        if self.training and self.cond_drop_prob > 0:
            mask = torch.rand(size=[len(x)], device=x.device) >= self.cond_drop_prob
            image_emb = image_emb * mask[:, None].to(image_emb)
            text_emb = text_emb * mask[:, None].to(text_emb)

        # Fuse text and image embeddings
        fused_emb = self.fusion_module(text_emb, image_emb)
        
        # Condition and run through backbone
        cond = [(t_embed, self.time_token_cond), (fused_emb, True)]
        return self._forward_with_cond(x, cond)