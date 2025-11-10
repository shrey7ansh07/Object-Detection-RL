import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models 
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import torchvision.transforms as T # NEW IMPORT for standard ViT transforms


VISUAL_INPUT_SIZE = (896,896) 
VISUAL_EMBEDDING_DIM = 128 
VIT_BASE_DIM = 768


# class VisualFeatureExtractor(nn.Module):
#     def __init__(self, output_dim=VISUAL_EMBEDDING_DIM):
#         super(VisualFeatureExtractor, self).__init__()
#         self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
#         self.vit.heads = nn.Identity()

#         for param in self.vit.parameters():
#             param.requires_grad = False
            
#         self.projection_head = nn.Linear(VIT_BASE_DIM, output_dim)
        
#         for param in self.projection_head.parameters():
#             param.requires_grad = False
        
#         self.vit.eval()
#         self.projection_head.eval()

#     def forward(self, x):
#         with torch.no_grad():
#              features = self.vit(x) 
#         embedding = self.projection_head(features)
#         return embedding




class VisualFeatureExtractor(nn.Module):
    """
    Extract contextual ViT feature map once per image and produce region embeddings on demand.
    """
    def __init__(self, output_dim=VISUAL_EMBEDDING_DIM, input_size=896, patch_size=16, device=torch.device("cpu")):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.device = device

        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.conv_proj = self.vit.conv_proj
        self.encoder = self.vit.encoder
        self.embed_dim = self.vit.hidden_dim  # 768

        # Projection to desired embedding dim
        self.projection_head = nn.Linear(self.embed_dim, output_dim).to(device)

        # Freeze ViT backbone
        for p in self.conv_proj.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.eval().to(device)
        self.to(self.device)
        # Normalization transform (for full image)
        self.transform = T.Compose([
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def _resize_pos_embed(self, pos_embed, new_size, cls_token=True):
        num_extra_tokens = 1 if cls_token else 0
        orig_num_tokens = pos_embed.shape[1] - num_extra_tokens
        orig_size = int(orig_num_tokens ** 0.5)
        new_h, new_w = new_size

        extra_tokens = pos_embed[:, :num_extra_tokens]
        pos_tokens = pos_embed[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)

        pos_tokens = F.interpolate(
            pos_tokens,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False
        )

        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        return new_pos_embed
    
    @torch.no_grad()
    def forward_feature_map(self, img_pil: Image.Image) -> Tuple[torch.Tensor, dict]:
        """
        Run ViT once on resized+square-padded image.
        Returns contextual feature map and resize metadata.
        Handles arbitrary resolutions via positional embedding interpolation.
        """
        # ----------------------------------------------------------------------
        # 1️⃣ Resize & pad while preserving aspect ratio
        # ----------------------------------------------------------------------
        orig_w, orig_h = img_pil.size
        scale = min(self.input_size / orig_w, self.input_size / orig_h)
        resized_w, resized_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img_pil.resize((resized_w, resized_h), resample=Image.BILINEAR)

        # create padded canvas (square)
        canvas = Image.new('RGB', (self.input_size, self.input_size), (0, 0, 0))
        pad_left = (self.input_size - resized_w) // 2
        pad_top = (self.input_size - resized_h) // 2
        canvas.paste(img_resized, (pad_left, pad_top))

        resize_info = {
            'scale_w': resized_w / orig_w,
            'scale_h': resized_h / orig_h,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'orig_w': orig_w,
            'orig_h': orig_h
        }

        # ----------------------------------------------------------------------
        # 2️⃣ Transform and move to correct device
        # ----------------------------------------------------------------------
        img_tensor = self.transform(canvas).unsqueeze(0)
        target_device = next(self.conv_proj.parameters()).device
        img_tensor = img_tensor.to(target_device)

        # ----------------------------------------------------------------------
        # 3️⃣ Forward through ViT patch embedding
        # ----------------------------------------------------------------------
        x = self.conv_proj(img_tensor)  # (1, 768, Hf, Wf)
        B, C, Hf, Wf = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (1, N, 768)

        # ----------------------------------------------------------------------
        # 4️⃣ Resize positional embeddings to match (e.g., 56×56 for 896 input)
        # ----------------------------------------------------------------------
        pos_embed = self.encoder.pos_embedding
        num_patches = Hf * Wf
        if pos_embed.shape[1] != num_patches + 1:
            new_pos_embed = self._resize_pos_embed(pos_embed, (Hf, Wf), cls_token=True)
            self.encoder.pos_embedding = nn.Parameter(new_pos_embed)
        
        # Add CLS token before feeding into encoder
        cls_token = self.vit.class_token.expand(B, -1, -1)  # (1, 1, 768)
        x_seq = torch.cat((cls_token, x_seq), dim=1)  # (1, 3137, 768)

        # ----------------------------------------------------------------------
        # 5️⃣ Forward through transformer encoder (contextualization)
        # ----------------------------------------------------------------------
        x_encoded = self.encoder(x_seq)  # (1, N+1, 768)
        x_encoded = x_encoded[:, 1:, :]  # drop CLS
        feature_map = x_encoded.transpose(1, 2).reshape(B, C, Hf, Wf)

        return feature_map, resize_info


    @torch.no_grad()
    def extract_region_embedding(self, feature_map: torch.Tensor, bbox: np.ndarray, resize_info: dict) -> np.ndarray:
        """
        Pool feature map region corresponding to bbox (in original pixel coords).
        bbox: (x1, y1, x2, y2)
        """
        C, Hf, Wf = feature_map.shape
        x1, y1, x2, y2 = bbox

        # Map bbox from original image -> resized/padded coordinates
        sx, sy = resize_info['scale_w'], resize_info['scale_h']
        pl, pt = resize_info['pad_left'], resize_info['pad_top']
        input_size = self.input_size

        rx1 = x1 * sx + pl
        ry1 = y1 * sy + pt
        rx2 = x2 * sx + pl
        ry2 = y2 * sy + pt

        # Map to patch grid indices
        px1 = int(np.floor(rx1 / input_size * Wf))
        py1 = int(np.floor(ry1 / input_size * Hf))
        px2 = int(np.ceil(rx2 / input_size * Wf))
        py2 = int(np.ceil(ry2 / input_size * Hf))

        # Clip safely
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(Wf - 1, px2), min(Hf - 1, py2)

        if px2 <= px1 or py2 <= py1:
            pooled = feature_map.mean(dim=(1, 2))
        else:
            region = feature_map[:, py1:py2, px1:px2]  # (C, h, w)
            pooled = region.mean(dim=(1, 2))           # (C,)

        # embedding = self.projection_head(pooled.T).cpu().numpy().flatten()  # (output_dim,)
        pooled = pooled.to(next(self.projection_head.parameters()).device)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)  # ensure batch dim
        try:
            torch.cuda.synchronize()
            # embedding = self.projection_head(pooled.T).detach().cpu().numpy().flatten()
            pooled = pooled.to(next(self.projection_head.parameters()).device)

            # Ensure shape = (1, 768)
            if pooled.ndim == 1:
                pooled = pooled.unsqueeze(0)

            embedding = self.projection_head(pooled).detach().cpu().numpy().flatten()

        except RuntimeError as e:
            torch.cuda.synchronize()
            print(f"[CUDA ERROR] during projection: {e}")
            print(f"pooled.device={pooled.device}, pooled.shape={pooled.shape}, "
                f"projection_head.device={next(self.projection_head.parameters()).device}")
            raise
        return embedding
