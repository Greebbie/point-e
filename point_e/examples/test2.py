# test_multimodal.py
import torch
import torch.nn as nn
from typing import Optional, Iterable, Dict, Any
from PIL import Image
import os
import shutil # Added for cache clearing

# --- Check for Point-E Installation and Import Components ---
try:
    from point_e.models.configs import MODEL_CONFIGS
    from point_e.models.transformer import CLIPImageGridPointDiffusionTransformer, timestep_embedding
    from point_e.models.pretrained_clip import FrozenImageCLIP, ImageType # ImageType is often Union[str, Image.Image]
    print("Point-E library found and imported successfully.")
except ImportError:
    print("Error: Point-E library not found.")
    print("Please install it using: pip install point-e")
    # Define a placeholder ImageType if import fails, to allow script parsing
    ImageType = Any
    # Exit if point-e is critical and not found
    # import sys
    # sys.exit("Exiting due to missing Point-E library.")
    # Or raise the error again:
    raise

# --- Fusion Module Code ---

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
    def __init__(self, text_dim=768, image_dim=1024, fusion_dim=512, heads=8):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
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

# --- Multimodal Transformer Code ---

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
                text_dim=self.text_feature_dim,
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

    # Methods _get_clip_embeddings, _compute_fused_embedding, cached_model_kwargs, forward
    # now correctly use self.device and self.dtype which were set in __init__

    def _get_clip_embeddings(
        self, batch_size: int, images: Optional[Iterable[ImageType]] = None,
        texts: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:

        if self.clip is None:
            raise RuntimeError("CLIP model was not initialized successfully.")

        outputs = {}
        # Ensure CLIP model component is on the instance's device
        self.clip.model.to(self.device)
        with torch.no_grad():
            if texts is not None:
                # Ensure output tensor is on the correct device and dtype
                outputs['text_emb'] = self.clip(batch_size=batch_size, texts=texts).to(device=self.device, dtype=self.dtype)
            if images is not None:
                # Ensure output tensor is on the correct device and dtype
                grid = self.clip.embed_images_grid(images).to(device=self.device, dtype=self.dtype)
                grid = grid[:, :, 1:] # Drop CLS
                grid = grid.permute(0, 2, 1).contiguous() # [B, N, C_grid]
                outputs['img_tokens'] = grid
        return outputs

    def _compute_fused_embedding(
        self, text_emb: Optional[torch.Tensor] = None, img_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Ensure fusion module component is on the instance's device
        self.fusion.to(self.device)
        if self.use_cross_attention:
            assert text_emb is not None and img_tokens is not None
            # Ensure inputs are on correct device/dtype before passing to fusion module
            text_emb = text_emb.to(device=self.device, dtype=self.dtype)
            img_tokens = img_tokens.to(device=self.device, dtype=self.dtype)
            fused = self.fusion(text_emb=text_emb, img_tokens=img_tokens)
        else:
            assert text_emb is not None and img_tokens is not None
            # Ensure inputs are on correct device/dtype
            text_emb = text_emb.to(device=self.device, dtype=self.dtype)
            img_tokens = img_tokens.to(device=self.device, dtype=self.dtype)
            img_emb_avg = img_tokens.mean(dim=1)
            concat_emb = torch.cat([text_emb, img_emb_avg], dim=1)
            fused = self.fusion(concat_emb)
        # Ensure final fused embedding is on the correct device/dtype
        return fused.to(device=self.device, dtype=self.dtype)

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
        # Ensure input point cloud 'x' has the expected number of channels
        if x.shape[1] != self.expected_input_channels:
             # This is just a warning, the code will proceed. The error would occur later if dimensions mismatch.
             print(f"Warning: Input point cloud 'x' has {x.shape[1]} channels, "
                   f"but model expected {self.expected_input_channels}. Mismatch may cause errors.")

        # Move inputs to the instance's device
        x = x.to(self.device)
        t = t.to(self.device)

        # Calculate timestep embedding and ensure it's on the correct device/dtype
        # Assume self.time_embed and self.backbone are correctly placed on self.device by parent __init__
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        t_embed = t_embed.to(device=self.device, dtype=self.dtype)

        # Determine the fused conditioning embedding
        if embeddings is not None:
            fused = embeddings.to(device=self.device, dtype=self.dtype)
        else:
            assert images is not None and texts is not None
            clip_embeddings = self._get_clip_embeddings(B, images, texts)
            text_emb = clip_embeddings.get('text_emb')
            img_tokens = clip_embeddings.get('img_tokens')
            fused = self._compute_fused_embedding(text_emb, img_tokens)
            # fused is already on correct device/dtype from _compute_fused_embedding

        # Prepare conditioning inputs
        cond = [(t_embed, self.time_token_cond), (fused, True)]

        # Ensure backbone is on the correct device before calling its forward pass
        # The parent __init__ should handle placing the backbone on the correct device.
        # If errors occur here, may need explicit self.backbone.to(self.device)
        # self.backbone.to(self.device)
        output = self._forward_with_cond(x, cond)
        return output

# --- Main Execution Block for Testing ---
if __name__ == "__main__":
    print("\n--- Starting Test Execution ---")

    # --- Define Cache Directory ---
    cache_dir = "./point_e_cache_test"
    print(f"Using cache directory: {cache_dir}")

    # --- Clear Cache Directory (to prevent checksum errors) ---
    clear_cache = True # Set to False if you want to keep the cache
    if clear_cache and os.path.exists(cache_dir):
        print(f"Clearing cache directory: {cache_dir}...")
        try:
            # Attempt to remove the directory tree
            shutil.rmtree(cache_dir)
            print("Cache cleared successfully.")
        except OSError as e:
            # Print error if removal fails (e.g., due to permissions)
            print(f"Error clearing cache: {e}. Please clear manually if issues persist.")
            # Optionally, exit if cache clearing is critical
            # import sys
            # sys.exit("Exiting due to cache clearing failure.")
    # Ensure the directory exists after attempting to clear it
    os.makedirs(cache_dir, exist_ok=True)


    # --- Setup Device and Dtype ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        dtype = torch.float16
    else:
        device = torch.device('cpu')
        dtype = torch.float32
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")


    # --- Prepare Dummy Data ---
    batch_size = 2
    # Using 6 input channels (e.g., XYZ+RGB) based on common Point-E usage
    input_channels = 6
    n_ctx = 1024        # Point cloud context length
    clip_image_size = 224

    # Dummy point cloud input (with corrected channels)
    dummy_point_cloud = torch.randn(
        batch_size, input_channels, n_ctx, # Use corrected input_channels
        device=device, dtype=dtype
    )
    # Dummy timestep input
    dummy_timesteps = torch.randint(
        0, 1000, (batch_size,),
        device=device
    )
    # Dummy text input
    dummy_texts = ["a red chair", "a blue table"] * (batch_size // 2)
    # Dummy image input (create blank PIL images)
    dummy_images = [
        Image.new('RGB', (clip_image_size, clip_image_size), color = 'red'),
        Image.new('RGB', (clip_image_size, clip_image_size), color = 'blue')
    ] * (batch_size // 2)

    print(f"\nPrepared dummy data:")
    print(f"  Point Cloud Shape: {dummy_point_cloud.shape}") # Should show [B, 6, 1024]
    print(f"  Timesteps Shape: {dummy_timesteps.shape}")
    print(f"  Texts: {dummy_texts}")
    print(f"  Number of Images: {len(dummy_images)}")

    # --- Test Model Instantiation and Methods ---
    all_tests_passed = True # Flag to track overall success
    for use_cross_attn_flag in [False, True]:
        print(f"\n--- Testing with use_cross_attention={use_cross_attn_flag} ---")
        try:
            # Instantiate the model
            model = SimpleMultimodalTransformer(
                device=device,
                dtype=dtype,
                cache_dir=cache_dir,
                use_cross_attention=use_cross_attn_flag
            )
            print("Model instantiated successfully.")

            # Determine expected output channels from the model's config
            expected_output_channels = model.output_channels # Access from parent class

            # Test cached_model_kwargs
            print("Testing cached_model_kwargs...")
            model_kwargs_input = {'images': dummy_images, 'texts': dummy_texts}
            cached_kwargs = model.cached_model_kwargs(batch_size, model_kwargs_input)
            fused_embedding_cached = cached_kwargs['embeddings']
            print(f"  cached_model_kwargs output shape: {fused_embedding_cached.shape}")
            assert fused_embedding_cached.shape == (batch_size, model.backbone.width)
            print("  cached_model_kwargs test PASSED.")

            # Test forward pass with raw inputs (like during training)
            print("Testing forward pass with raw images/texts...")
            output_raw = model(
                x=dummy_point_cloud,
                t=dummy_timesteps,
                images=dummy_images,
                texts=dummy_texts
            )
            print(f"  Forward pass (raw) output shape: {output_raw.shape}")
            assert output_raw.shape[0] == batch_size
            assert output_raw.shape[1] == expected_output_channels
            assert output_raw.shape[2] == n_ctx
            print(f"  Forward pass (raw) test PASSED.")


            # Test forward pass with cached embeddings (like during sampling)
            print("Testing forward pass with cached embeddings...")
            output_cached = model(
                x=dummy_point_cloud,
                t=dummy_timesteps,
                embeddings=fused_embedding_cached # Pass the cached embedding
            )
            print(f"  Forward pass (cached) output shape: {output_cached.shape}")
            assert output_cached.shape[0] == batch_size
            assert output_cached.shape[1] == expected_output_channels
            assert output_cached.shape[2] == n_ctx
            print("  Forward pass (cached) test PASSED.")

            # Optional: Check if outputs are numerically close
            if torch.allclose(output_raw.float(), output_cached.float(), atol=1e-4):
                 print("  Outputs from raw and cached forward passes are numerically close.")
            else:
                 print("  Warning: Outputs from raw and cached forward passes differ numerically (this might be ok).")


        except Exception as e:
            print(f"!!! Test FAILED for use_cross_attention={use_cross_attn_flag} !!!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            all_tests_passed = False # Mark failure

    print("\n--- Test Execution Finished ---")
    if all_tests_passed:
        print("All tests passed successfully!")
    else:
        print("Some tests failed.")

