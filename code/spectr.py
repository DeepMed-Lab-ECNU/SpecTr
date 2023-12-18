#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:53:41 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_modeling import Spectral_ZipBlock_four, ParallelBlock_CAT
from spectr_block import Trans_block, AdaptivePool_Encoder
from typing import Optional, Union

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, up_size=(7, 128, 128)):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            # nn.GroupNorm(32, out_channels),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.up_size = up_size

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, size=self.up_size, mode='trilinear')
            # x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x

class GNConv3x3ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, up_size=(128, 128)):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
        )
        self.up_size = up_size

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, size=self.up_size, mode='trilinear')
            # x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x

class GNConv3x3ReLU_2D(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            # x = F.interpolate(x, size=self.up_size, mode='trilinear')
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x

class Conv3x3GNReLU_2D(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x

class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels, out_size):
        super().__init__()
        self.skip_conv = nn.Conv3d(skip_channels, pyramid_channels, kernel_size=1)
        self.out_size = out_size

    def forward(self, x, skip=None):
        x = F.interpolate(x, size=self.out_size, mode="trilinear")
        skip = self.skip_conv(skip)
        x = x + skip
        return x

class FPNBlock_2D(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        skip = self.skip_conv(skip)
        x = x + skip
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0, up_size=(7, 128, 128), decode_order='cgr'):
        super().__init__()
        if decode_order == 'cgr':
            blocks = [
                Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples), up_size=up_size[n_upsamples - 1])]
        else:
            blocks = [
                GNConv3x3ReLU(in_channels, out_channels, upsample=bool(n_upsamples), up_size=up_size[n_upsamples - 1])]

        if n_upsamples > 1:
            for t in range(1, n_upsamples):
                if decode_order == 'cgr':
                    blocks.append(
                        Conv3x3GNReLU(out_channels, out_channels, upsample=True, up_size=up_size[n_upsamples - t - 1]))
                else:
                    blocks.append(
                        GNConv3x3ReLU(out_channels, out_channels, upsample=True, up_size=up_size[n_upsamples - t - 1]))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.block(x)
        return x

class SegmentationBlock_2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0, decode_order='cgr'):
        super().__init__()
        if decode_order == 'cgr':
            blocks = [Conv3x3GNReLU_2D(in_channels, out_channels, upsample=bool(n_upsamples))]
        else:
            blocks = [
                GNConv3x3ReLU_2D(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for t in range(1, n_upsamples):
                if decode_order == 'cgr':
                    blocks.append(Conv3x3GNReLU_2D(out_channels, out_channels, upsample=True))
                else:
                    blocks.append(GNConv3x3ReLU_2D(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.block(x)
        return x

class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )

class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            max_seq=60,
            spatial_size=(128, 128),
            coscale_depth=1,
            coscale_entmax='softmax',
            use_coscale=True,
            decode_order='cgr',
            use_layerscale=False,
            init_values=1,
            zoom_spectral=True,
    ):
        super().__init__()

        self.out_channels = segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[encoder_depth - 4:][::-1]
        # encoder_channels = encoder_channels[:encoder_depth + 1]
        if zoom_spectral:
            f_spatials = [[max_seq // 2 ** i, spatial_size[1] // 2 ** i, spatial_size[0] // 2 ** i] for i in
                          range(encoder_depth)]
        else:
            f_spatials = [[max_seq, spatial_size[1] // 2 ** i, spatial_size[0] // 2 ** i] for i in
                          range(encoder_depth)]

        f_spectrums = [i[0] for i in f_spatials]
        self.use_coscale = use_coscale

        if use_coscale:
            self.parallel_blocks = nn.ModuleList([ParallelBlock_CAT(
                dims=encoder_channels[::-1], num_heads=8, mlp_ratios=[3, 3, 3, 3],
                drop=dropout, attn_drop=dropout, drop_path=dropout,
                use_entmax15=coscale_entmax,
                use_layerscale=use_layerscale,
                init_values=init_values,
            )
                for idx_p in range(coscale_depth)])

        self.p5 = nn.Conv3d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1], out_size=tuple(f_spatials[-2]))
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2], out_size=tuple(f_spatials[-3]))
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3], out_size=tuple(f_spatials[-4]))

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples, up_size=f_spatials,
                              decode_order=decode_order)
            for n_upsamples in [3, 2, 1, 0]
        ])

        self.merge = MergeBlock('cat')
        self.dropout = nn.Dropout3d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        if self.use_coscale:
            for blk in self.parallel_blocks:
                c2, c3, c4, c5 = blk(c2, c3, c4, c5)
        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        return x

class FPNDecoder_2D(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            max_seq=60,
            spatial_size=(128, 128),
            coscale_depth=1,
            condense_entmax='adaptive_entmax',
            coscale_entmax='adaptive_entmax',
            use_coscale=True,
            decode_order='cgr',
            use_layerscale=False
    ):
        super().__init__()
        self.out_channels = segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[encoder_depth - 4:][::-1]
        # encoder_channels = encoder_channels[:encoder_depth + 1]
        f_spatials = [[max_seq // 2 ** i, spatial_size[1] // 2 ** i, spatial_size[0] // 2 ** i] for i in
                      range(encoder_depth)]

        self.use_coscale = use_coscale
        if use_coscale:
            self.parallel_blocks = nn.ModuleList([
                ParallelBlock_CAT(
                    dims=encoder_channels[::-1], num_heads=8, mlp_ratios=[3, 3, 3, 3],
                    drop=dropout, attn_drop=dropout, drop_path=dropout,
                    use_entmax15=coscale_entmax,
                    use_layerscale=use_layerscale
                )
                for _ in range(coscale_depth)]
            )

        self.zip_blocks = nn.ModuleList([
            Spectral_ZipBlock_four(
                dims=encoder_channels[::-1], num_heads=8, mlp_ratios=[3, 3, 3, 3],
                drop=dropout, attn_drop=dropout, drop_path=dropout,
                use_entmax15=condense_entmax, use_layerscale=False,
            )
            for _ in range(1)])

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock_2D(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock_2D(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock_2D(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock_2D(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples,
                                 decode_order=decode_order)
            for n_upsamples in [3, 2, 1, 0]
        ])

        self.merge = MergeBlock('cat')
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]
        if self.use_coscale:
            for blk in self.parallel_blocks:
                c2, c3, c4, c5 = blk(c2, c3, c4, c5)

        for blk in self.zip_blocks:
            c2, c3, c4, c5 = blk(c2, c3, c4, c5)

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class Spectr_backbone(nn.Module):
    """
    non-zoom-spectr and downsample by transformer block and after conv
    """

    def __init__(self, in_channels, f_maps=64, encode_layer_order='gcr', choose_translayer=[0, 1, 1, 1],
                 tran_enc_layers=[1, 1, 1, 1], dropout_att=0.1, dropout=0.1,
                 num_levels=4, spatial_size=(256, 256), zoom_spectral=True,
                 transformer_dim=3, init_values=1.0, use_layerscale=True,
                 conv_kernel_size=(1, 3, 3), conv_padding=(0, 1, 1), max_seq=60, use_entmax15='entmax_bisect',
                 att_blocks='att'):

        super(Spectr_backbone, self).__init__()

        assert len(tran_enc_layers) == num_levels, "input correct choiced transformer layers"

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        if zoom_spectral:
            if isinstance(spatial_size, int):
                f_spatials = [[max_seq // 2 ** i, spatial_size // 2 ** i, spatial_size // 2 ** i] for i in
                              range(num_levels)]
            else:
                f_spatials = [[max_seq // 2 ** i, spatial_size[1] // 2 ** i, spatial_size[0] // 2 ** i] for i in
                              range(num_levels)]
        else:
            if isinstance(spatial_size, int):
                f_spatials = [[max_seq, spatial_size // 2 ** i, spatial_size // 2 ** i] for i in range(num_levels)]
            else:
                f_spatials = [[max_seq, spatial_size[1] // 2 ** i, spatial_size[0] // 2 ** i] for i in
                              range(num_levels)]

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        self.out_channels = []
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if choose_translayer[i]:
                transf = Trans_block(out_feature_num, spatial_size=f_spatials[i][1:], depth_trans=tran_enc_layers[i],
                                        dropout=dropout, attention_dropout_rate=dropout_att,
                                        use_entmax15=use_entmax15, att_blocks=att_blocks,
                                        transformer_dim=transformer_dim, init_values=init_values,
                                        seq_length=f_spatials[i][0], use_layerscale=use_layerscale)
            else:
                transf = None
            self.out_channels.append(out_feature_num)

            if i == 0:
                encoder = AdaptivePool_Encoder(in_channels, out_feature_num,
                                               # skip pooling in the first encoder
                                               apply_pooling=False,
                                               conv_layer_order=encode_layer_order,
                                               conv_kernel_size=conv_kernel_size,
                                               padding=conv_padding,
                                               output_size=f_spatials[i],
                                               transform=transf)

            else:
                encoder = AdaptivePool_Encoder(f_maps[i - 1], out_feature_num,
                                               apply_pooling=True,
                                               conv_layer_order=encode_layer_order,
                                               conv_kernel_size=conv_kernel_size,
                                               padding=conv_padding,
                                               output_size=f_spatials[i],
                                               transform=transf)

            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # in the last layer a 1×1 convolution reduces the number of output

    def forward(self, x):
        # encoder part
        encoders_features = []
        for idx, encoder in enumerate(self.encoders):
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.append(x)

        return encoders_features


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Sigmoid() if activation == 'sigmoid' else nn.Softmax(dim=1)
        super().__init__(conv2d, upsampling, activation)


class SpecTr(nn.Module):
    """
    Args:

        num_levels: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimentions than previous one (e.g. for depth 0 we will have features 
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 4
        f_maps: Dimension of the encoder's feature map from the first stage (e.g. 32 -> 64 -> 128 -> 256).
        encode_layer_order: The order of encoder module operation e.g.: cgr means Conv -> GroupNorm -> ReLU
        decode_layer_order: The order of decoder module operation e.g.: cgr means Conv -> GroupNorm -> ReLU
        choose_translayer: Choose the encoder stages which contain transformer e.g. [0, 1, 1, 1] means the 2nd, 3rd, 4th
        encoder layer with Conv+Transformer, and the first layer with Conv.
        tran_enc_layers:  the number of layers in transformer
        spatial_size: the input HSI spatial size
        zoom_spectral: False : don't downsample spectral dimension in encoder. True downsample spectra e.g. (spectral, h, w)
        -> (spectral/2, h/2, w/2) in each encoder stages.
        conv_kernel_size: 3D conv kernel size
        conv_padding: 3D conv padding setting
        max_seq: spectral number
        use_entmax15: wo. & w. sparsity operation (choose ['softmax', 'adaptive_entmax']) in attention block.
        decoder_pyramid_channels: A number of convolution filters in decoder_pyramid blocks
        decoder_segmentation_channels: A number of convolution filters in segmentation blocks
        dropout_att: Dropout ration in self-attention attention map. Default is 0.1
        dropout: Dropout ration in self-attention FFN modules. Default is 0.1
        in_channels: input channel in encoder, default is 1
        classes: the output class number, default is 1.
        activation: the activation of segmentation head, default is 'sigmod'.
        upsampling: Final upsampling factor. Default is 1 to preserve input-output spatial shape identity
        att_blocks: wo. & w. layerscale operation (choose ['att', 'layerscale']) in attention block.
        decode_choice: Decode A. : 3D decoder  and Decoder B. Lite 2D decoder for faster spectral (choose ['3D', 'decoder_2D'])
        coscale_depth: the number of transformer layer in Inter-scale Spatiospectral Feature Extractor
        use_coscale: Use Inter-scale Spatiospectral Feature Extractor.
        transformer_dim: the ffn's MLP dimension ration in the transfromer
        use_layerscale: Use layerscale operation
        init_values: the init_values alpha in layerscale operation
        condense_entmax: wo. & w. sparsity operation (choose ['softmax', 'adaptive_entmax']) on attention block in decode lite.
        coscale_entmax: wo. & w. sparsity operation (choose ['softmax', 'adaptive_entmax']) on attention block in
        Inter-scale Spatiospectral Feature Extractor

    Returns:
        ``torch.nn.Module``: **SpecTr**

    """

    def __init__(
            self,
            num_levels: int = 4,
            f_maps: int = 32,
            encode_layer_order: str = 'scr',
            decode_layer_order: str = 'cgr',
            choose_translayer: list = [0, 1, 1, 1],
            tran_enc_layers: list = [1, 1, 1, 1],
            spatial_size: tuple = (256, 256),
            zoom_spectral: bool = True,
            conv_kernel_size: Union[tuple, int] = (1, 3, 3),
            conv_padding: Union[tuple, int] = (0, 1, 1),
            max_seq: int = 60,
            use_entmax15: str = 'adaptive_entmax',
            decoder_pyramid_channels: int = 128,
            decoder_segmentation_channels: int = 64,
            dropout_att: float = 0.1,
            dropout: float = 0.1,
            in_channels: int = 1,
            classes: int = 1,
            activation: Optional[str] = 'sigmoid',
            upsampling: int = 1,
            att_blocks: str = 'layerscale',
            decode_choice: str = '3D',
            coscale_depth: int = 1,
            use_coscale: bool = True,
            transformer_dim: int = 3,
            use_layerscale: bool = True,
            init_values: float = 1.0,
            condense_entmax: str = 'adaptive_entmax',
            coscale_entmax: str = 'adaptive_entmax',
    ):
        super().__init__()

        self.decode_choice = decode_choice
        assert len(choose_translayer) == len(tran_enc_layers), "transformer block number must equal depth of length "

        self.encoder = Spectr_backbone(
            in_channels, f_maps=f_maps, encode_layer_order=encode_layer_order, dropout_att=dropout_att, dropout=dropout,
            choose_translayer=choose_translayer, tran_enc_layers=tran_enc_layers, num_levels=num_levels,
            spatial_size=spatial_size, zoom_spectral=zoom_spectral,
            conv_kernel_size=conv_kernel_size, conv_padding=conv_padding, max_seq=max_seq,
            use_entmax15=use_entmax15, att_blocks=att_blocks, transformer_dim=transformer_dim, init_values=init_values,
            use_layerscale=use_layerscale
        )

        if decode_choice == '3D':
            self.decoder = FPNDecoder(
                encoder_channels=self.encoder.out_channels,
                encoder_depth=num_levels,
                pyramid_channels=decoder_pyramid_channels,
                segmentation_channels=decoder_segmentation_channels,
                max_seq=max_seq,
                spatial_size=spatial_size,
                coscale_depth=coscale_depth,
                coscale_entmax=coscale_entmax,
                use_coscale=use_coscale,
                decode_order=decode_layer_order,
                use_layerscale=use_layerscale,
                zoom_spectral=zoom_spectral)

        elif decode_choice == "decoder_2D":
            self.decoder = FPNDecoder_2D(
                encoder_channels=self.encoder.out_channels,
                encoder_depth=num_levels,
                pyramid_channels=decoder_pyramid_channels,
                segmentation_channels=decoder_segmentation_channels,
                max_seq=max_seq,
                spatial_size=spatial_size,
                coscale_depth=coscale_depth,
                condense_entmax=condense_entmax,
                coscale_entmax=coscale_entmax,
                use_coscale=use_coscale,
                decode_order=decode_layer_order,
                use_layerscale=use_layerscale,
            )
        else:
            raise ValueError("please choice correct decode methods : '3D', 'decoder_2D'!")

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        if self.decode_choice != 'decoder_2D':
            decoder_output = decoder_output.mean(2)
            masks = self.segmentation_head(decoder_output)
        else:
            masks = self.segmentation_head(decoder_output)
        return masks

    def forward_encoder(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        return features
