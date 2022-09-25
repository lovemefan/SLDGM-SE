#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @FileName  :ffcunet.py
# @Time      :2022/9/19 20:25
# @Author    :lovemefan
# @email     :lovemefan@outlook.com
from sgmse.backbones import BackboneRegistry
import torch.nn as nn
import torch
import numpy as np

from sgmse.backbones.dcunet import OnReIm, get_activation
from sgmse.backbones.modules.ffc import FFC
from sgmse.backbones.shared import GaussianFourierProjection, DiffusionStepEmbedding, ComplexLinear


class FFCResidualBlocks(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 alpha,
                 ):
        """
        Args:
            alpha: the radio of channels used in the global branch of the module
        """
        super(FFCResidualBlocks, self).__init__()
        self.ffc_modules = nn.Sequential(

        )
        self.alpha = alpha
        self.ffc1 = FFC(in_channels, in_channels, kernel_size, alpha, alpha, padding=1)
        self.ffc2 = FFC(in_channels, in_channels, kernel_size, alpha, alpha, padding=1)

    def forward(self, x):
        residual = x
        x = self.ffc1(x)
        x = self.ffc2(x)

        if type(residual) is tuple:
            channel = residual[0].size(1) + residual[1].size(1)
            channel_g = int(self.alpha * channel)
            channel_l = channel - channel_g
            residual = torch.cat(residual, dim=1)
            residual = residual.split([channel_l, channel_g], dim=1)
            return x[0] + residual[0], x[1] + residual[1]
        else:
            residual = torch.concat((residual, torch.zeros_like(residual)), dim=1)
            channel = residual.size(1)
            channel_g = int(self.alpha*channel)
            channel_l = channel - channel_g
            return x[0] + residual[:, :channel_l, ...], x[1] + residual[:, channel_l:, ...]


class DownSampleBlock(nn.Module):
    def __init__(self, n, channel_in, channel_out, alpha):
        super(DownSampleBlock, self).__init__()

        self.ffc_residuals = nn.Sequential(
            *[FFCResidualBlocks(channel_in, (3, 3), alpha) for _ in range(n)]
        )
        self.conv = nn.Conv2d(channel_in, channel_out, 1, stride=2)

    def forward(self, x):

        for module in self.ffc_residuals:
            x = module(x)
        x = torch.concat(x, dim=1)
        x = self.conv(x)
        x = x.chunk(2, dim=1)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, n, channel_in, channel_out, alpha):
        super(UpSampleBlock, self).__init__()
        self.alpha = alpha
        self.convT = nn.ConvTranspose2d(channel_in, channel_out, 1, stride=2, output_padding=1)
        self.ffc_residuals = nn.Sequential(
            *[FFCResidualBlocks(channel_out, (3, 3), alpha) for i in range(n)]
        )
        self.conv = nn.Conv2d(channel_in, channel_out, 1)

    def forward(self, x, residual):
        x = self.convT(x)
        channel_g = int(x.size(1)*self.alpha)
        x = x.split([x.size(1) - channel_g, channel_g], dim=1)
        for module in self.ffc_residuals:
            x = module(x)

        if type(residual) is not tuple:
            residual = residual, torch.zeros_like(residual)
        x = torch.concat(x + residual, dim=1)

        x = self.conv(x)
        return x


@BackboneRegistry.register("ffcunet")
class FFCUNET(nn.Module):
    @staticmethod
    def add_argparse_args(parser):
        pass
        return parser

    def __init__(
            self,
            in_ch=32,
            n=4,
            k=4,
            time_embedding="gfp",
            embed_dim: int = 128,
            activation: str = "silu",
            **unused_kwargs
    ):
        """
        Args:
            in_ch (int): in_ch controls the overall width of the networks
            n (int): the number of ffc residual block
            k (int): depth of the ffc-unet architecture
            alpha: real number \in [0, 1] in case of ffc-unet controls
            the proportion of channels going to the global branch
        """
        super(FFCUNET, self).__init__()
        self.alpha = np.linspace(.75, 0, k)
        self.conv_in = nn.Conv2d(2, in_ch, (7, 7), padding=3)
        self.down_sample = nn.ModuleList(
            [DownSampleBlock(n, in_ch*2**i, in_ch*(2**(i+1)), self.alpha[i]) for i in range(1, k)]
        )
        self.ffc_residual_blocks = nn.Sequential(
            *[FFCResidualBlocks(in_ch*(2**k), (3, 3), 0.25) for i in range(n)]
        )
        self.up_sample = nn.ModuleList(
            [UpSampleBlock(n, in_ch * (2 ** (i+1)), in_ch * 2 ** i, self.alpha[i]) for i in range(k-1, 0, -1)]
        )
        self.conv_out = nn.Conv2d(in_ch*2, 2, (7, 7), padding=3)

        self.time_embedding = time_embedding

        embed_ops = []
        if self.time_embedding is not None:
            if self.time_embedding == "gfp":
                embed_ops += [GaussianFourierProjection(embed_dim=embed_dim, complex_valued=False)]

            elif self.time_embedding == "ds":
                embed_ops += [DiffusionStepEmbedding(embed_dim=embed_dim, complex_valued=False)]

            for _ in range(2):
                embed_ops += [
                    ComplexLinear(embed_dim, embed_dim, complex_valued=True),
                    OnReIm(get_activation(activation))
                ]
        self.embed = nn.Sequential(*embed_ops)

    def forward(self, x, t) -> torch.Tensor:
        """
        Input shape is expected to be $(batch, channel, nfreqs, time)$
        Args:
            spec (Tensor): batch, channel, nfreqs, time  spectrogram tensor. 1D, 2D or 3D tensor, time last.
        Returns:
            Tensor, of shape (batch, time) or (time).
        """
        t_embed = self.embed(t)
        unet_residual = []
        x = self.conv_in(x)
        for module in self.down_sample:
            unet_residual.append(x)
            x = module(x)

        x = self.ffc_residual_blocks(x)

        x = torch.concat(x, dim=1)
        for module in self.up_sample:
            residual = unet_residual.pop()
            x = module(x, residual)
            del residual

        x = self.conv_out(x)

        return x


if __name__ == '__main__':
    net = FFCUNET()
    import torchaudio
    wavform, samplerate = torchaudio.load('../../audio/train/noisy/trainS_99_noisy_louder.wav')
    window = torch.hann_window(510, periodic=True)
    spec = torch.stft(wavform[..., :32640], n_fft=510, hop_length=128, window=window, center=True,
                       return_complex=False)
    spec = spec.permute((0, 3, 1, 2))
    net(spec, None)
