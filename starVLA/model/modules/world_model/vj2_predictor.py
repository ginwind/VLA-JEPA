# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from starVLA.model.modules.world_model.vj2_modules import ACBlock as Block
from starVLA.model.modules.world_model.vj2_modules import build_action_block_causal_attention_mask
from starVLA.model.modules.world_model.vj2_tensors import trunc_normal_


class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=False,
        # added
        num_add_tokens=8,
        **kwargs
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)

        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Position embedding
        self.uniform_power = uniform_power

        # Attention Blocks
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        attn_mask = None
        if self.is_frame_causal:
            grid_depth = self.num_frames // self.tubelet_size
            grid_height = self.img_height // self.patch_size
            grid_width = self.img_width // self.patch_size
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth, grid_height, grid_width, add_tokens=num_add_tokens
            )
        self.attn_mask = attn_mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, actions, extrinsics=None):
        """
        :param x: context tokens [B, T, p_H*p_W, D]
        :param actions: action tokens [B, T * num, D]
        """
        # Map tokens to predictor dimensions
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)
        #print(T, N_ctxt, self.grid_height, self.grid_width)
        #exit()

        # Interleave action tokens
        a = self.action_encoder(actions)
        a = a.view(B, T, -1, D)
        cond_tokens = a.shape[2]
        x = x.view(B, T, self.grid_height * self.grid_width, D)  # [B, T, H*W, D]
        if self.use_extrinsics:
            cond_tokens += 1
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, e, x], dim=2).flatten(1, 2)  # [B, T*(H*W+3), D]
        else:
            x = torch.cat([a, x], dim=2).flatten(1, 2)  # [B, T*(H*W+2), D]

        attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                )

        # Split out action and frame tokens
        x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)  # [B, T, K+H*W, D]
        x = x[:, :, cond_tokens:, :].flatten(1, 2)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return x


def vit_ac_predictor(**kwargs):
    model = VisionTransformerPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

if __name__ == "__main__":
    device = torch.device("cuda:0")
    from transformers import AutoVideoProcessor, AutoModel
    hf_repo = "/home/dataset-assist-0/algorithm/ginwind/models/vjepa2-vitl-fpc64-256"

    model = AutoModel.from_pretrained(hf_repo).to(device)
    processor = AutoVideoProcessor.from_pretrained(hf_repo)

    from torchcodec.decoders import VideoDecoder
    import numpy as np

    video_url = "/home/dataset-assist-0/algorithm/ginwind/raw_starVLA/data_view_0.mp4"
    vr = VideoDecoder(video_url)
    frame_idx = np.arange(0, 8) # choosing some frames. here, you can define more complex sampling strategy
    video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
    print(video.shape) #[8, 3, 256, 256]
    print(video.max(), video.min()) # [255, 0]
    video = processor(video, return_tensors="pt").to(model.device) # 不能并行
    print(video)
    print(video["pixel_values_videos"].shape)  #[1, 8, 3, 256, 256]
    print(torch.min(video["pixel_values_videos"]), torch.max(video["pixel_values_videos"])) #[-2.1179, 2.6051]

    from transformers.image_utils import load_image
    image = load_image("/home/dataset-assist-0/algorithm/ginwind/raw_starVLA/data_image_view_0.png")
    pixel_values = processor(image, return_tensors="pt").to(model.device)["pixel_values_videos"]
    pixel_values = pixel_values.repeat(1, 8, 1, 1, 1) # repeating image 16 times
    print(pixel_values.shape)    #[1, 8, 3, 256, 256]
    print(torch.min(pixel_values), torch.max(pixel_values))

    with torch.no_grad():
        image_embeddings = model.get_vision_features(pixel_values)    
    print(image_embeddings.shape) # [1, 1024, 1024]

    test_model = VisionTransformerPredictorAC(
        num_frames=4,
        img_size=((256, 256)),
        tubelet_size=1,
        depth=12,
        num_heads=4,
        embed_dim=1024,
        action_embed_dim=1024,
        num_add_tokens=3,
    ).to(device)
    #x = torch.randn(4, 392, 768).to(device)
    x = image_embeddings[:, :768, :]
    actions = torch.randn(1, 12, 1024).to(device)
    #states = torch.randn(1, 12, 1024).to(device)
    outputs = test_model(x, actions)
    print(outputs.shape) #[1, 768, 1024]
