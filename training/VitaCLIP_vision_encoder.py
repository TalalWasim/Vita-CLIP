from typing import Tuple
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import mul
from functools import reduce
import math

from VitaCLIP_vision_encoder_utils import QuickGELU, LayerNorm, TransformerEncoderLayer, ImagePatchEmbed2D



class CLIPVisionEncoder(nn.Module):

    def __init__(
        self,
        # data shape
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 8,
        # model def
        feature_dim: int = 768,
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
        embed_dim: int = 512,
        # use summary token
        use_summary_token: bool = False,
        # use local prompts
        use_local_prompts: bool = False,
        # use global prompts
        use_global_prompts: bool = False,
        num_global_prompts: int = 8,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        
        self.patch_embed = ImagePatchEmbed2D(img_size=input_size[0], patch_size=patch_size[0], in_chans=3, embed_dim=feature_dim)
        self.num_patches = np.prod([x // y for x, y in zip(input_size, patch_size)]) + 1

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))
        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, feature_dim]))
        self.time_embed = nn.Parameter(torch.zeros([num_frames, feature_dim]))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                in_feature_dim=feature_dim, qkv_dim=feature_dim, num_heads=num_heads,
                mlp_factor=mlp_factor, act=act, use_summary_token=use_summary_token,
                use_local_prompts=use_local_prompts, num_frames=num_frames, patch_size=patch_size
            ) for _ in range(num_layers)
        ])

        self.ln_pre = LayerNorm(feature_dim)
        self.ln_post = LayerNorm(feature_dim)
        scale = feature_dim ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(feature_dim, embed_dim))

        # global prompts
        self.use_global_prompts = use_global_prompts
        self.num_global_prompts = num_global_prompts

        if self.use_global_prompts:
            self.global_prompts = nn.Parameter(torch.zeros(num_layers, self.num_global_prompts, feature_dim))
            self._initialize_global_prompts(patch_size, feature_dim)
        
        self._initialize_weights()


    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

    def _initialize_global_prompts(self, patch_size, prompt_dim):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.global_prompts.data, -val, val)

    def temporal_encoding(self, x, T, B):
        ## Time Embeddings
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)

        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(0):
            time_embed = self.time_embed.unsqueeze(0).transpose(1,2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2).squeeze(0)
            x = x + new_time_embed
        else:
            x = x + self.time_embed

        x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T)
        return x

    def forward(self, x: torch.Tensor):

        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)

        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.view(1, 1, -1).repeat(x.size(0), 1, 1), x], dim=1)
        
        x = x + self.pos_embed
        x = self.temporal_encoding(x, T, B)

        x = self.ln_pre(x)

        if self.use_global_prompts:
            for i, blk in enumerate(self.blocks):
                global_prompts = self.global_prompts[i].expand(B*T, -1, -1)

                x = torch.cat((x[:, :1, :], global_prompts, x[:, 1:, :]), dim=1)
                x = blk(x)
                x = torch.cat((x[:, :1, :], x[:, self.num_global_prompts+1:, :]), dim=1)
        else:
            for blk in self.blocks:
                x = blk(x)

        cls_x = self.ln_post(x[:, 0, :])
        cls_x = cls_x @ self.proj
        cls_x = rearrange(cls_x, '(b t) e -> b t e', b=B,t=T).mean(dim=1)

        return cls_x