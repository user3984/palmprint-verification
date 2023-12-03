import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from functools import partial
from typing import Callable, Optional
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.bricks.drop import build_dropout


class InvertedResidual(nn.Module):
    """InvertedResidual block for MobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        layers.extend([
            ConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 1e-5.
        inplace (bool): inplace: can optionally do the
            operation in-place. Defaults to False.
        data_format (str): The input data format, could be 'channels_last'
             or 'channels_first', representing (B, C, H, W) and
             (B, N, C) format data respectively. Defaults to 'channels_last'.
    """

    def __init__(self,
                 dim: int,
                 layer_scale_init_value: float = 1e-5,
                 inplace: bool = False,
                 data_format: str = 'channels_last'):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), \
            "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * layer_scale_init_value)

    def forward(self, x):
        if self.data_format == 'channels_first':
            if self.inplace:
                return x.mul_(self.weight.view(-1, 1, 1))
            else:
                return x * self.weight.view(-1, 1, 1)
        return x.mul_(self.weight) if self.inplace else x * self.weight


def scaled_dot_product_attention_pyimpl(query,
                                        key,
                                        value,
                                        attn_mask=None,
                                        dropout_p=0.,
                                        scale=None,
                                        is_causal=False):
    scale = scale or query.size(-1)**0.5
    if is_causal and attn_mask is not None:
        attn_mask = torch.ones(
            query.size(-2), key.size(-2), dtype=torch.bool).tril(diagonal=0)
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf'))

    attn_weight = query @ key.transpose(-2, -1) / scale
    if attn_mask is not None:
        attn_weight += attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value


class MultiheadAttention(nn.Module):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        use_layer_scale (bool): Whether to use layer scale. Defaults to False.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 use_layer_scale=False,
                 layer_scale_init_value=0.):
        super(MultiheadAttention, self).__init__()

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        if qk_scale is not None:
            self.scaled_dot_product_attention = partial(
                scaled_dot_product_attention_pyimpl,
                scale=self.head_dims**-0.5)
        else:
            self.scaled_dot_product_attention = F.scaled_dot_product_attention

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = build_dropout(dropout_layer)

        if use_layer_scale or (layer_scale_init_value > 0):
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(
                embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_drop = self.attn_drop if self.training else 0.
        x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN layer.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """  # noqa

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.,
        bias: bool = True,
        dropout_layer: Optional[dict] = None,
        # norm_cfg: Optional[dict] = None,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.out_dims = out_dims or embed_dims
        hidden_dims = feedforward_channels or embed_dims

        self.w12 = nn.Linear(self.embed_dims, 2 * hidden_dims, bias=bias)

        # if norm_cfg is not None:
        #     self.norm = build_norm_layer(norm_cfg, hidden_dims)
        # else:
        self.norm = nn.Identity()

        self.w3 = nn.Linear(hidden_dims, self.out_dims, bias=bias)

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(
                dim=embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self,
                x: torch.Tensor,
                identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.norm(hidden)
        out = self.w3(hidden)
        out = self.gamma2(out)
        out = self.dropout_layer(out)

        if self.out_dims != self.embed_dims or not self.add_identity:
            # due to the dimension inconsistence or user setting
            # not to apply residual operation
            return out

        if identity is None:
            identity = x
        return identity + out


class SwiGLUFFNFused(SwiGLUFFN):
    """SwiGLU FFN layer with fusing.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """  # noqa

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.,
        bias: bool = True,
    ) -> None:
        out_dims = out_dims or embed_dims
        feedforward_channels = feedforward_channels or embed_dims
        feedforward_channels = (int(feedforward_channels * 2 / 3) + 7) // 8 * 8
        super().__init__(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            out_dims=out_dims,
            layer_scale_init_value=layer_scale_init_value,
            bias=bias,
        )


class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 layer_scale_init_value=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 ffn_type='origin',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN')):
        super(TransformerEncoderLayer, self).__init__()

        self.embed_dims = embed_dims

        self.ln1 = nn.LayerNorm(self.embed_dims)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value)

        self.ln2 = nn.LayerNorm(self.embed_dims)

        if ffn_type == 'origin':
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                layer_scale_init_value=layer_scale_init_value)
        elif ffn_type == 'swiglu_fused':
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                layer_scale_init_value=layer_scale_init_value)
        else:
            raise NotImplementedError
        
        self.init_weights()

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = self.ffn(self.ln2(x), identity=x)
        return x

