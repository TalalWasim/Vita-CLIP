#!/usr/bin/env python

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn

from operator import mul
from functools import reduce
import math

'''
QuickGELU and LayerNorm w/ fp16 from official CLIP repo
(https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py)
'''
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
        self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
        qk_proj_dim: int, v_proj_dim: int, num_heads: int,
        out_dim: int
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)



    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0); assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1); assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        
        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        return out

class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        # attention def
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
        # use summary token
        use_summary_token: bool = False,
        # use local prompts
        use_local_prompts: bool = False,
        # model def
        num_frames: int = 8,
        patch_size: Tuple[int, int] = (16, 16),
    ):
        super().__init__()

        self.attn = Attention(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)

        self.use_summary_token = use_summary_token
        self.use_local_prompts = use_local_prompts

        # for both summary token and local prompts we need the cls_proj layer and the num_frames
        if self.use_summary_token or self.use_local_prompts:
            self.cls_proj = nn.Linear(in_feature_dim, in_feature_dim)
            self.num_frames = num_frames
        
        # for summary token we need a layer norm and attention
        if self.use_summary_token:
            self.summary_ln = LayerNorm(in_feature_dim)
            self.summary_attn_layer = Attention(
                                    q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
                                    qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim
                                )

        # for local prompts we init learnable tokens
        if self.use_local_prompts:
            self.local_prompts = nn.Parameter(torch.zeros(1, self.num_frames, in_feature_dim))
            self._initialize_cls_prompts(patch_size, in_feature_dim)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def _initialize_cls_prompts(self, patch_size, prompt_dim):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.local_prompts.data, -val, val)


    def forward(self, x: torch.Tensor):
        # get the cls tokens and apply fc
        # which is required for both summaru token
        # and local prompts
        if self.use_summary_token or self.use_local_prompts:
            BT, N, C = x.shape
            T = self.num_frames
            B = BT//T

            cls_token = x[:, 0, :].view(B, T, C)
            cls_token_proj = self.cls_proj(cls_token)
        
        # then apply ln and attn if summary token being used
        if self.use_summary_token:
            summary_token_norm = self.summary_ln(cls_token_proj)
            summary_token_attn = cls_token_proj + self.summary_attn_layer(summary_token_norm, summary_token_norm, summary_token_norm)
            summary_token_attn_reshape = summary_token_attn.view(BT, 1, C)
            x = torch.cat([x, summary_token_attn_reshape], dim=1)

        # then if local prompts are being used
        if self.use_local_prompts:
            local_prompts = self.local_prompts.expand(B, -1, -1)
            # If train time frames and
            # test time frames are not equal
            if T != self.num_frames:
                token_multiplier = T//self.num_frames
                local_prompts = local_prompts.repeat(1,token_multiplier,1)
            
            # use additive conditioning
            local_prompts = local_prompts + cls_token_proj

            # repeat across frames
            local_prompts = local_prompts.repeat_interleave(repeats=T, dim=0)
            x = torch.cat((x[:, :1, :], local_prompts, x[:, 1:, :]), dim=1)

        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)

        # remove the tokens after self attention
        if self.use_summary_token:
            x = x[:, :-1, :]
        if self.use_local_prompts:
            x = torch.cat((x[:, :1, :], x[:, local_prompts.shape[1]+1:, :]), dim=1)

        x = x + self.mlp(self.norm2(x))
        return x

class ImagePatchEmbed2D(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x