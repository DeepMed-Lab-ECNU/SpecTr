# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, LayerNorm

from entmax import EntmaxAlpha
from einops import rearrange

from timm.models.layers import DropPath


def swish(x):
    return x * torch.sigmoid(x)

def sharpen(x, T, eps=1e-6):
    temp = x ** (1 / T)
    return (temp + eps) / (temp.sum(axis=-1, keepdims=True) + eps)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate,
                 use_entmax15, vis):
        super(Attention, self).__init__()
        self.vis = vis

        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.use_entmax15 = use_entmax15
        if use_entmax15 == 'softmax':
            self.att_fn = F.softmax
        elif use_entmax15 == 'adaptive_entmax':
            self.att_fn = EntmaxAlpha(self.num_attention_heads)
        else:
            raise ValueError("Oops! That was invalid attention function.Try again...")

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)


        if self.use_entmax15 == 'adaptive_entmax':
            attention_probs = self.att_fn(attention_scores)  # sharpen(attention_scores,0.5)
        else:
            attention_probs = self.att_fn(attention_scores, dim=-1)

        weights = attention_probs if self.vis else None

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Attention_query_global(nn.Module):#query spectral attention
    def __init__(self, hidden_size, num_heads, attention_dropout_rate,
                 use_entmax15, vis):
        super(Attention_query_global, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.use_entmax15 = use_entmax15
        if use_entmax15 == 'softmax':
            self.att_fn = F.softmax
        elif use_entmax15 == 'adaptive_entmax':
            self.att_fn = EntmaxAlpha(self.num_attention_heads)
        else:
            raise ValueError("Oops! That was invalid attention function.Try again...")

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        shape = hidden_states.shape

        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        mixed_query_layer = self.query(hidden_states.mean(1).unsqueeze(1))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if self.use_entmax15 == 'adaptive_entmax':
            attention_probs = self.att_fn(attention_scores)  # sharpen(attention_scores,0.5)
        else:
            attention_probs = self.att_fn(attention_scores, dim=-1)

        weights = attention_probs if self.vis else None

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, drop_out, out_dim=None):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        if out_dim is not None:
            self.fc2 = Linear(mlp_dim, out_dim)
        else:
            self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(drop_out)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size=784, seq_length=10, drop_out=0.):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=drop_out)

        pe = torch.zeros(seq_length, hidden_size)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # .transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, hidden_size=768, seq_length=10, drop_out=0.):
        super(Embeddings, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, hidden_size))
        self.dropout = Dropout(drop_out)

    def forward(self, x):
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, hidden_size, mlp_dim, num_heads, drop_out, attention_dropout_rate, use_entmax15, vis):
        super(Block, self).__init__()
        self.hidden_size = hidden_size

        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, drop_out)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate,
                              use_entmax15=use_entmax15, vis=vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights

class Block_LayerScale(nn.Module):
    def __init__(self, hidden_size, mlp_dim, num_heads, drop_out, attention_dropout_rate, use_entmax15, vis,
                 init_values=1e-1, use_layerscale=True):
        super(Block_LayerScale, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, drop_out)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate,
                              use_entmax15=use_entmax15, vis=vis)

        if use_layerscale==False:
            self.register_buffer("gamma_1", init_values * torch.ones((hidden_size)))
            self.register_buffer("gamma_2", init_values * torch.ones((hidden_size)))
        else:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((hidden_size)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((hidden_size)), requires_grad=True)

    def forward(self, x):

        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = self.gamma_1 * x
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.gamma_2 * x
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, mlp_dim, num_heads, drop_out, attention_dropout_rate, use_entmax15, vis,
                 block='att', init_values=1e-1, use_layerscale=True):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        if block == 'att':
            for n in range(num_layers):
                layer = Block(hidden_size, mlp_dim, num_heads, drop_out, attention_dropout_rate, use_entmax15, vis,
                              )
            self.layer.append(copy.deepcopy(layer))
        elif block == 'layerscale':
            for n in range(num_layers):
                layer = Block_LayerScale(hidden_size, mlp_dim, num_heads, drop_out, attention_dropout_rate,
                                         use_entmax15, vis, init_values=init_values, use_layerscale=use_layerscale)
            self.layer.append(copy.deepcopy(layer))
        else:
            raise ValueError("Oops! That was invalid attention layers.Try 'att','ca', 'layerscale'!!!")

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)

        return hidden_states, attn_weights

class Transformer(nn.Module):
    def __init__(self, seq_length, num_layers, hidden_size, mlp_dim, num_heads, drop_out,
                 attention_dropout_rate, use_entmax15, vis=False, block='att',
                 init_values=1e-1, use_layerscale=True):
        super(Transformer, self).__init__()
        self.vis = vis
        self.block = block

        self.embeddings = PositionalEncoding(hidden_size, seq_length, drop_out)

        self.encoder = Encoder(num_layers, hidden_size, mlp_dim, num_heads, drop_out, attention_dropout_rate,
                               use_entmax15, vis, init_values=init_values,
                               block=block, use_layerscale=use_layerscale)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

class ParallelBlock_CAT(nn.Module):
    """ Parallel block class. """

    def __init__(self, dims, num_heads, mlp_ratios=[],  drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, vis=False, use_entmax15='softmax',
                 upsample_mode='trilinear', init_values=1e-2, use_layerscale=False):
        super().__init__()

        self.norm12 = norm_layer(dims[1])
        self.norm13 = norm_layer(dims[2])
        self.norm14 = norm_layer(dims[3])
        self.att2 = Attention(
            dims[1], num_heads=num_heads, attention_dropout_rate=attn_drop,
            use_entmax15=use_entmax15, vis=vis
        )
        self.att3 = Attention(
            dims[2], num_heads=num_heads, attention_dropout_rate=attn_drop,
            use_entmax15=use_entmax15, vis=vis
        )
        self.att4 = Attention(
            dims[3], num_heads=num_heads, attention_dropout_rate=attn_drop,
            use_entmax15=use_entmax15, vis=vis
        )

        # from timm.models.layers import DropPath, to_2tuple, trunc_normal_
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm22 = norm_layer(dims[1])
        self.norm23 = norm_layer(dims[2])
        self.norm24 = norm_layer(dims[3])

        mlp_input_dim = sum((dims[1], dims[2], dims[3]))
        self.mlp2 = Mlp(hidden_size=mlp_input_dim, mlp_dim=int(dims[1]*mlp_ratios[1]),
                        drop_out=drop, out_dim=dims[1])
        self.mlp3 = Mlp(hidden_size=mlp_input_dim, mlp_dim=int(dims[2]*mlp_ratios[2]),
                        drop_out=drop, out_dim=dims[2])
        self.mlp4 = Mlp(hidden_size=mlp_input_dim, mlp_dim=int(dims[3]*mlp_ratios[3]),
                        drop_out=drop, out_dim=dims[3])

        self.upsample_mode = upsample_mode

        self.use_layerscale = use_layerscale
        if use_layerscale == False:
            self.register_buffer("gamma_1_1", init_values * torch.ones((dims[1])))
            self.register_buffer("gamma_1_2", init_values * torch.ones((dims[1])))
            self.register_buffer("gamma_2_1", init_values * torch.ones((dims[2])))
            self.register_buffer("gamma_2_2", init_values * torch.ones((dims[2])))
            self.register_buffer("gamma_3_1", init_values * torch.ones((dims[3])))
            self.register_buffer("gamma_3_2", init_values * torch.ones((dims[3])))
        else:
            self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dims[1])), requires_grad=True)
            self.gamma_1_2 = nn.Parameter(init_values * torch.ones((dims[1])), requires_grad=True)
            self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dims[2])), requires_grad=True)
            self.gamma_2_2 = nn.Parameter(init_values * torch.ones((dims[2])), requires_grad=True)
            self.gamma_3_1 = nn.Parameter(init_values * torch.ones((dims[3])), requires_grad=True)
            self.gamma_3_2 = nn.Parameter(init_values * torch.ones((dims[3])), requires_grad=True)



    def upsample(self, x, scale_size, size):
        """ Feature map up-sampling. """
        return self.interpolate(x, scale_size=scale_size, input_size=size)

    def downsample(self, x, scale_size, size):
        """ Feature map down-sampling. """
        return self.interpolate(x, scale_size=scale_size, input_size=size)

    def interpolate(self, x, scale_size, input_size):
        """ Feature map interpolation. """
        B, S, C = x.shape
        S, H, W = input_size
        # assert N == H * W
        img_tokens = x

        img_tokens = img_tokens.transpose(1, 2).reshape(-1, C, S, H, W)
        img_tokens = F.interpolate(img_tokens, size=scale_size, mode=self.upsample_mode)
        out = img_tokens.reshape(-1, C, scale_size[0]).transpose(1, 2)

        return out

    def forward(self, x1, x2, x3, x4):
        _, (_, _, S2, H2, W2), (_, _, S3, H3, W3), (_, _, S4, H4, W4) = x1.shape, x2.shape, x3.shape, x4.shape
        x2 = rearrange(x2, 'b c s h w -> (b h w) s c')
        x3 = rearrange(x3, 'b c s h w -> (b h w) s c')
        x4 = rearrange(x4, 'b c s h w -> (b h w) s c')
        # Conv-Attention.

        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur2, w2 = self.att2(cur2)
        cur3, w3 = self.att3(cur3)
        cur4, w4 = self.att4(cur4)

        if self.use_layerscale == True:
            x2 = x2 + self.drop_path(cur2) * self.gamma_1_1
            x3 = x3 + self.drop_path(cur3) * self.gamma_2_1
            x4 = x4 + self.drop_path(cur4) * self.gamma_3_1
        else:
            x2 = x2 + self.drop_path(cur2)
            x3 = x3 + self.drop_path(cur3)
            x4 = x4 + self.drop_path(cur4)

        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)

        upsample3_2 = self.upsample(cur3, scale_size=(S2, H2, W2), size=(S3, H3, W3))
        upsample4_3 = self.upsample(cur4, scale_size=(S3, H3, W3), size=(S4, H4, W4))
        upsample4_2 = self.upsample(cur4, scale_size=(S2, H2, W2), size=(S4, H4, W4))
        downsample2_3 = self.downsample(cur2, scale_size=(S3, H3, W3), size=(S2, H2, W2))
        downsample3_4 = self.downsample(cur3, scale_size=(S4, H4, W4), size=(S3, H3, W3))
        downsample2_4 = self.downsample(cur2, scale_size=(S4, H4, W4), size=(S2, H2, W2))


        cur2 = torch.cat((cur2, upsample3_2, upsample4_2), dim=-1)
        cur3 = torch.cat((cur3, upsample4_3, downsample2_3), dim=-1)
        cur4 = torch.cat((cur4, downsample3_4, downsample2_4), dim=-1)

        # MLP.
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)

        if self.use_layerscale == True:
            x2 = x2 + self.drop_path(cur2) * self.gamma_1_2
            x3 = x3 + self.drop_path(cur3) * self.gamma_2_2
            x4 = x4 + self.drop_path(cur4) * self.gamma_3_2
        else:
            x2 = x2 + self.drop_path(cur2)
            x3 = x3 + self.drop_path(cur3)
            x4 = x4 + self.drop_path(cur4)

        x2 = rearrange(x2, '(b p1 p2) s c -> b c s p1 p2', p1=H2, p2=W2)
        x3 = rearrange(x3, '(b p1 p2) s c -> b c s p1 p2', p1=H3, p2=W3)
        x4 = rearrange(x4, '(b p1 p2) s c -> b c s p1 p2', p1=H4, p2=W4)

        return x1, x2, x3, x4

class Spectral_ZipBlock_four(nn.Module):
    """ Parallel block class. """
    def __init__(self, dims, num_heads, mlp_ratios=[], drop=0., attn_drop=0., use_layerscale=False, init_values=1e-1,
                 drop_path=0., norm_layer=nn.LayerNorm, vis=False, use_entmax15='softmax',
                 upsample_mode='trilinear'):
        super().__init__()
        self.norm11 = norm_layer(dims[0])
        self.norm12 = norm_layer(dims[1])
        self.norm13 = norm_layer(dims[2])
        self.norm14 = norm_layer(dims[3])
        self.att1 = Attention_query_global(
            dims[0], num_heads=num_heads, attention_dropout_rate=attn_drop,
            use_entmax15=use_entmax15, vis=vis
        )

        self.att2 = Attention_query_global(
            dims[1], num_heads=num_heads, attention_dropout_rate=attn_drop,
            use_entmax15=use_entmax15, vis=vis
        )
        self.att3 = Attention_query_global(
            dims[2], num_heads=num_heads, attention_dropout_rate=attn_drop,
            use_entmax15=use_entmax15, vis=vis
        )
        self.att4 = Attention_query_global(
            dims[3], num_heads=num_heads, attention_dropout_rate=attn_drop,
            use_entmax15=use_entmax15, vis=vis
        )

        # from timm.models.layers import DropPath, to_2tuple, trunc_normal_
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm21 = norm_layer(dims[0])
        self.norm22 = norm_layer(dims[1])
        self.norm23 = norm_layer(dims[2])
        self.norm24 = norm_layer(dims[3])

        self.mlp1 = Mlp(hidden_size=dims[0], mlp_dim=int(dims[0] * mlp_ratios[0]),
                        drop_out=drop, out_dim=dims[0])
        self.mlp2 = Mlp(hidden_size=dims[1], mlp_dim=int(dims[1] * mlp_ratios[1]),
                        drop_out=drop, out_dim=dims[1])
        self.mlp3 = Mlp(hidden_size=dims[2], mlp_dim=int(dims[2] * mlp_ratios[2]),
                        drop_out=drop, out_dim=dims[2])
        self.mlp4 = Mlp(hidden_size=dims[3], mlp_dim=int(dims[3] * mlp_ratios[3]),
                        drop_out=drop, out_dim=dims[3])

        self.upsample_mode = upsample_mode

        self.use_layerscale = use_layerscale
        if use_layerscale == False:
            self.gamma_1_1 = torch.ones((dims[0]), requires_grad=True)
            self.gamma_1_2 = torch.ones((dims[0]), requires_grad=True)
            self.gamma_2_1 = torch.ones((dims[1]), requires_grad=True)
            self.gamma_2_2 = torch.ones((dims[1]), requires_grad=True)
            self.gamma_3_1 = torch.ones((dims[2]), requires_grad=True)
            self.gamma_3_2 = torch.ones((dims[3]), requires_grad=True)
            self.gamma_4_1 = torch.ones((dims[3]), requires_grad=True)
            self.gamma_4_2 = torch.ones((dims[3]), requires_grad=True)
        else:
            self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dims[0])), requires_grad=True)
            self.gamma_1_2 = nn.Parameter(init_values * torch.ones((dims[0])), requires_grad=True)
            self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dims[1])), requires_grad=True)
            self.gamma_2_2 = nn.Parameter(init_values * torch.ones((dims[1])), requires_grad=True)
            self.gamma_3_1 = nn.Parameter(init_values * torch.ones((dims[2])), requires_grad=True)
            self.gamma_3_2 = nn.Parameter(init_values * torch.ones((dims[2])), requires_grad=True)
            self.gamma_4_1 = nn.Parameter(init_values * torch.ones((dims[3])), requires_grad=True)
            self.gamma_4_2 = nn.Parameter(init_values * torch.ones((dims[3])), requires_grad=True)

    def upsample(self, x, scale_size, size):
        """ Feature map up-sampling. """
        return self.interpolate(x, scale_size=scale_size, input_size=size)

    def downsample(self, x, scale_size, size):
        """ Feature map down-sampling. """
        return self.interpolate(x, scale_size=scale_size, input_size=size)

    def interpolate(self, x, scale_size, input_size):
        """ Feature map interpolation. """
        B, S, C = x.shape
        S, H, W = input_size
        img_tokens = x

        img_tokens = img_tokens.transpose(1, 2).reshape(-1, C, S, H, W)
        img_tokens = F.interpolate(img_tokens, size=scale_size, mode=self.upsample_mode)
        out = img_tokens.reshape(-1, C, scale_size[0]).transpose(1, 2)

        return out

    def forward(self, x1, x2, x3, x4):
        (_, _, S1, H1, W1), (_, _, S2, H2, W2), (_, _, S3, H3, W3), (_, _, S4, H4, W4) = x1.shape, x2.shape, x3.shape, x4.shape
        x1 = rearrange(x1, 'b c s h w -> (b h w) s c')
        x2 = rearrange(x2, 'b c s h w -> (b h w) s c')
        x3 = rearrange(x3, 'b c s h w -> (b h w) s c')
        x4 = rearrange(x4, 'b c s h w -> (b h w) s c')
        # Conv-Attention.

        cur1 = self.norm11(x1)
        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)

        cur1, w1 = self.att1(cur1)
        cur2, w2 = self.att2(cur2)
        cur3, w3 = self.att3(cur3)
        cur4, w4 = self.att4(cur4)  # b 1 c
        if self.use_layerscale:
            x1 = x1.mean(1).unsqueeze(1) + cur1 * self.gamma_1_1
            x2 = x2.mean(1).unsqueeze(1) + cur2 * self.gamma_2_1
            x3 = x3.mean(1).unsqueeze(1) + cur3 * self.gamma_3_1
            x4 = x4.mean(1).unsqueeze(1) + cur4 * self.gamma_4_1
        else:
            x1 = x1.mean(1).unsqueeze(1) + cur1
            x2 = x2.mean(1).unsqueeze(1) + cur2
            x3 = x3.mean(1).unsqueeze(1) + cur3
            x4 = x4.mean(1).unsqueeze(1) + cur4

        cur1 = self.norm21(x1)
        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        # MLP.
        cur1 = self.mlp1(cur1)
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)

        if self.use_layerscale:
            x1 = x1 + cur1 * self.gamma_1_2
            x2 = x2 + cur2 * self.gamma_2_2
            x3 = x3 + cur3 * self.gamma_3_2
            x4 = x4 + cur4 * self.gamma_4_2
        else:
            x1 = x1 + cur1
            x2 = x2 + cur2
            x3 = x3 + cur3
            x4 = x4 + cur4

        x1 = rearrange(x1, '(b p1 p2) s c -> b c s p1 p2', p1=H1, p2=W1)
        x2 = rearrange(x2, '(b p1 p2) s c -> b c s p1 p2', p1=H2, p2=W2)
        x3 = rearrange(x3, '(b p1 p2) s c -> b c s p1 p2', p1=H3, p2=W3)
        x4 = rearrange(x4, '(b p1 p2) s c -> b c s p1 p2', p1=H4, p2=W4)

        # x1 = x1.mean(2)
        x1, x2, x3, x4 = x1.squeeze(2), x2.squeeze(2), x3.squeeze(2), x4.squeeze(2)
        return x1, x2, x3, x4
