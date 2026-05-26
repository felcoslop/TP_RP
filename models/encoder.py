import torch
import torch.nn as nn
from functools import partial
from segment_anything.modeling.image_encoder import ImageEncoderViT

class SAMEncoder(nn.Module):
    """
    Encoder ViT-B compatível com os checkpoints SAM-Road (PATCH_SIZE=512).
    Constrói o ImageEncoderViT diretamente com img_size=512, evitando
    o mismatch de pos_embed/rel_pos que ocorria ao usar sam_model_registry
    (que assume img_size=1024).
    """
    def __init__(self, img_size=512):
        super().__init__()
        # Mesma configuração do SAM ViT-B, mas com img_size=512
        self.encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        )
        # Registrar as mesmas normalizações de pixel do SAM-Road
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

    def forward(self, x):
        # x: [B, 3, H, W] com valores 0-255 (float)
        # Normalizar como o SAM-Road faz internamente
        x = (x - self.pixel_mean) / self.pixel_std
        return self.encoder(x)  # [B, 256, 32, 32] para img_size=512
