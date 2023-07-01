#!/usr/bin/env python

from typing import Tuple
import numpy as np

import torch
import torch.nn as nn

from VitaCLIP_vision_encoder import CLIPVisionEncoder
from VitaCLIP_text_encoder import CLIPTextEncoder, TextPromptLearner



class VitaCLIP(nn.Module):

    def __init__(
        self,
        # load weights
        backbone_path: str = '',
        # data shape
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 16,
        # model def
        feature_dim: int = 768,
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        embed_dim: int = 512,
        # use summary token
        use_summary_token: bool = False,
        # use local prompts
        use_local_prompts: bool = False,
        # use global prompts
        use_global_prompts: bool = False,
        num_global_prompts: int = 8,
        # use text prompt learning
        use_text_prompt_learning: bool = False,
        text_context_length: int = 77,
        text_vocab_size: int = 49408,
        text_transformer_width: int = 512,
        text_transformer_heads: int = 8,
        text_transformer_layers: int = 12,
        text_num_prompts: int = 8,
        text_prompt_pos: str = 'end',
        text_prompt_init: str = '',
        text_prompt_CSC: bool = False,
        text_prompt_classes_path: str = '',
        # zeroshot eval
        zeroshot_evaluation: bool = False,
        zeroshot_text_features_path: str = '',
        ):
        super().__init__()

        # frames and tubelet
        self.num_frames = num_frames

        # use summary token
        self.use_summary_token = use_summary_token

        # clip loss logit_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        # zeroshot text_features
        self.zeroshot_evaluation = zeroshot_evaluation
        if self.zeroshot_evaluation:
            self.text_features = torch.load(zeroshot_text_features_path, map_location='cpu')

        # visual model
        self.visual = CLIPVisionEncoder(
            # data shape
            input_size=input_size,
            num_frames=num_frames,
            # model def
            feature_dim=feature_dim,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_factor=mlp_factor,
            embed_dim=embed_dim,
            # use summary token
            use_summary_token=use_summary_token,
            # use local prompts
            use_local_prompts=use_local_prompts,
            # use global prompts
            use_global_prompts=use_global_prompts,
            num_global_prompts=num_global_prompts,
        )
        
        self.use_text_prompt_learning = use_text_prompt_learning

        # text prompt learning
        if self.use_text_prompt_learning:
            self.textual = CLIPTextEncoder(
                embed_dim=embed_dim,
                context_length=text_context_length,
                vocab_size=text_vocab_size,
                transformer_width=text_transformer_width,
                transformer_heads=text_transformer_heads,
                transformer_layers=text_transformer_layers,
            )
        
        if backbone_path:
            ckpt = torch.load(backbone_path)
            self.load_state_dict(ckpt, strict=False)

        if self.use_text_prompt_learning:
            with open(text_prompt_classes_path, 'r') as f:
                classes = f.read().strip().split('\n')
            
            self.prompt_learner = TextPromptLearner(
                            classnames=classes,
                            text_model=self.textual,
                            num_prompts=text_num_prompts,
                            prompts_init=text_prompt_init,
                            CSC=text_prompt_CSC,
                            ctx_pos=text_prompt_pos
                            )
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # freeze encoders
        self._freeze_visual_except_prompts_time_embed()
        self._freeze_textual()


    
    def _freeze_visual_except_prompts_time_embed(self):
        for name, param in self.visual.named_parameters():
                if 'summary' in name or 'local' in name or 'global' in name or 'time_embed' in name:
                    pass
                else:
                    param.requires_grad = False
    
    def _freeze_textual(self):
        for name, param in self.textual.named_parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.size()

        # used in training
        if self.use_text_prompt_learning:
            # text side
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.textual(prompts, tokenized_prompts)
            # vision side
            video_features = self.visual(x)
        # used in zeroshot evaluation
        else:
            # vision side
            video_features = self.visual(x)
            # text side
            text_features = self.text_features.to(video_features.device)

        # normalized features
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * video_features @ text_features.t()

        return logits