import torch
from torch import nn as nn
from vit_modeling import Transformer
from einops import rearrange

def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

class Spectral_Normalize(nn.Module):
    """
    create a list of modules with different spetral channel's normalize(bn,gn,in,ln)
    """
    def __init__(self, num_features, num_spectral, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,normalize_type='g'):
        super(Spectral_Normalize, self).__init__()
        self.num_spectral= num_spectral
        #         self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        if normalize_type == 'b':
            base_norm = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        elif normalize_type == 'g':
            num_groups = 8
            if num_features < num_groups:
                num_groups = 1
            base_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, eps=eps, affine=affine)
        elif normalize_type == 'i':
            base_norm = nn.InstanceNorm2d(num_features, eps, momentum, affine, track_running_stats)
            
        self.bns = nn.ModuleList(
            [base_norm for _ in range(num_spectral)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
    def forward(self, x):
        self._check_input_dim(x)
        out = torch.zeros_like(x)
        for i in range(self.num_spectral):
            out[:,:,i] = self.bns[i](x[:,:,i])
        return out
   
def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding, num_spectral):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order or 's' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        elif char == 's':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            modules.append(('spectralnorm', Spectral_Normalize(num_features=num_channels, num_spectral=num_spectral)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 's']")

    return modules

class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1, num_spectral=10):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding, num_spectral=num_spectral):
            self.add_module(name, module)

class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8,
                 padding=1, num_spectral=10, shape=(192, 192)):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding,num_spectral=num_spectral))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding,num_spectral = num_spectral))

class Trans_block(nn.Module):
    def __init__(self, in_channels, spatial_size, depth_trans=2, transformer_dim=3,
                 dropout=0.1, use_entmax15=False, att_blocks='att',
                 seq_length=60, attention_dropout_rate=0.1, init_values=1e-1, use_layerscale=True):

        super(Trans_block, self).__init__()
        self.spatial_size = spatial_size
        self.seq_length = seq_length
        self.att_blocks = att_blocks

        self.trans = Transformer(seq_length=seq_length, num_layers=depth_trans, hidden_size=in_channels,
                                 mlp_dim=transformer_dim * in_channels,
                                 num_heads=8, drop_out=dropout, attention_dropout_rate=attention_dropout_rate,
                                 block=att_blocks, use_entmax15=use_entmax15, init_values=init_values,
                                 use_layerscale=use_layerscale)

    def forward(self, x):
        shape = x.shape
        x = rearrange(x, 'b c s h w -> (b h w) s c')
        x, att = self.trans(x)
        x = rearrange(x, '(b p1 p2) s c -> b c s p1 p2', p1=shape[-2], p2=shape[-1])
        return x

class AdaptivePool_Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True, output_size=(10, 256, 256),
                 pool_type='max', conv_layer_order='gcr', vis=False,
                 padding=1, transform=None):
        super(AdaptivePool_Encoder, self).__init__()
        self.vis = vis
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.AdaptiveMaxPool3d(output_size)
            else:
                self.pooling = nn.AdaptiveAvgPool3d(output_size)
        else:
            self.pooling = None

        if transform is not None:
            conv_kernel_size = (1, 3, 3)
            padding = (0, 1, 1)

        self.basic_module = DoubleConv(in_channels, out_channels,
                                       encoder=True,
                                       kernel_size=conv_kernel_size,
                                       order=conv_layer_order,
                                       num_groups=8,
                                       padding=padding,
                                       num_spectral=output_size[0],
                                       shape=(output_size[1], output_size[2]))
        self.transform = transform

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x1 = self.basic_module(x)

        if self.transform is not None:
            x1 = self.transform(x1)

        return x1
