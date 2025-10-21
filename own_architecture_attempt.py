import torch
from torch import nn as nn, Tensor
from functools import partial
import math


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def GN(num_channels):
    g_approx = math.sqrt(num_channels)
    g_actual = 2 ** round(math.log2(g_approx))  # map to the nearest pow of 2

    while num_channels % g_actual != 0:
        g_actual = g_actual // 2
        if g_actual == 0:
            g_actual = 1
            break

    return nn.GroupNorm(num_groups=g_actual, num_channels=num_channels)


class InvertedResidualConfig:
    def __init__(
        self,
        inp: int,
        kernel: int | tuple[int, int, int],
        exp: int,
        out: int,
        se: bool,
        activation: str,
        stride: int | tuple[int, int, int],
        width_mult: float = 1.0,
        expand: bool = True
    ):
        self.input_channels = self.adjust_channels(inp, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(exp, width_mult)
        self.out_channels = self.adjust_channels(out, width_mult)
        self.use_se = se
        self.use_hs = (activation == "HS") #True => HS; False => LReLU
        self.stride = stride
        self.expand = not (expand is False and inp == exp) #can`t set expand = False for inp != exp

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels*width_mult, divisible_by=8)



class SqueezeExcite3D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.squeeze_channels = make_divisible(in_channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)  # Squeeze: (B, C, 1, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.squeeze_channels, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.squeeze_channels, in_channels, bias=True),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()

        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1, 1)
        return x * scale.expand_as(x)


class InvertedResidual3D(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, norm_layer = None):
        super().__init__()

        self.cnf = cnf
        self.norm_layer = norm_layer
        if self.norm_layer is None:
            #self.norm_layer = partial(nn.BatchNorm3d, eps=1e-8, momentum=5e-4)

            self.norm_layer = GN

        self.use_residual_connection = ((cnf.input_channels == cnf.out_channels) and
                                        (cnf.stride == 1 or cnf.stride == (1,1,1) ))
        layers: list[nn.Module] = []

        if cnf.use_hs:
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)

        self.dw_kernel_tuple: tuple[int, int, int]
        if isinstance(cnf.kernel, int):
            self.dw_kernel_tuple = (cnf.kernel, cnf.kernel, cnf.kernel)
        else: self.dw_kernel_tuple = cnf.kernel

        self.dw_padding: tuple[int, int, int] = ((self.dw_kernel_tuple[0] - 1) // 2,
                                                 (self.dw_kernel_tuple[1] - 1) // 2,
                                                 (self.dw_kernel_tuple[2] - 1) // 2)

        #===== the block =====

        if cnf.expand:
            # expand
            layers.append(nn.Conv3d(
                in_channels=cnf.input_channels,
                out_channels=cnf.expanded_channels,
                kernel_size=1,
                stride=1, padding=0, bias=False))
            layers.append(self.norm_layer(cnf.expanded_channels))
            layers.append(self.activation)

        # depth-wise ip-CSN-style
        layers.append(nn.Conv3d(
            in_channels=cnf.expanded_channels,
            out_channels=cnf.expanded_channels,
            kernel_size=cnf.kernel,
            groups=cnf.expanded_channels,
            stride=cnf.stride, padding=self.dw_padding, bias=False))

        #no activation for the convs to factorize well
        #layers.append(self.norm_layer(cnf.expanded_channels))
        #layers.append(self.activation)

        # enable channel interactions
        layers.append(nn.Conv3d(
            in_channels=cnf.expanded_channels,
            out_channels=cnf.out_channels,
            kernel_size=1,
            stride=1, padding=0, bias=False))

        if cnf.use_se:
            layers.append(SqueezeExcite3D(in_channels=cnf.out_channels))

        layers.append(self.norm_layer(cnf.out_channels))
        layers.append(self.activation)

        #===== END OF THE BLOCK =====
        # converting created layers list
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual_connection:
            return x + self.block(x)
        return self.block(x)


class SqueezeExcite2D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcite2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #N,C,1,1
        self.squeeze_channels = make_divisible(in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.squeeze_channels, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.squeeze_channels, in_channels, bias=True),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.avg_pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale.expand_as(x)

class InvertedResidual2D(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: nn.BatchNorm2d = None, expand=True):
        super().__init__()

        self.cnf = cnf
        self.norm_layer = norm_layer
        if self.norm_layer is None:
            self.norm_layer = partial(nn.BatchNorm2d, eps=1e-8, momentum=5e-4)

        if not isinstance(cnf.kernel, int):
            raise ValueError("can`t use 3d tuples for 2D inverted residual. use Ints only")

        self.use_residual_connection = (cnf.input_channels == cnf.out_channels and cnf.stride == 1)
        layers: list[nn.Module] = []

        if cnf.use_hs:
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)

        if cnf.expand:
            # expand
            layers.append(nn.Conv2d(
                in_channels=cnf.input_channels,
                out_channels=cnf.expanded_channels,
                kernel_size=1,
                stride=1, padding=0, bias=False))
            layers.append(self.norm_layer(cnf.expanded_channels))
            layers.append(self.activation)

        # depth-wise
        layers.append(nn.Conv2d(
            in_channels=cnf.expanded_channels,
            out_channels=cnf.expanded_channels,
            kernel_size=cnf.kernel,
            groups=cnf.expanded_channels,
            stride=cnf.stride,
            padding=( (cnf.kernel-1) // 2,
                      (cnf.kernel-1) // 2
                      ),
            bias=False))
        layers.append(self.norm_layer(cnf.expanded_channels))
        layers.append(self.activation)

        if cnf.use_se:
            layers.append(SqueezeExcite2D(in_channels=cnf.expanded_channels))

        # project
        layers.append(nn.Conv2d(
            in_channels=cnf.expanded_channels,
            out_channels=cnf.out_channels,
            kernel_size=1,
            stride=1, padding=0, bias=False))
        layers.append(self.norm_layer(cnf.out_channels))
        # no activation

        # converting created layers list
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual_connection:
            return x + self.block(x)
        return self.block(x)


class MobileNet3D(nn.Module):
    def __init__(self, n_classes=33, width_mult=1.0):
        super().__init__()

        self.norm_layer = GN

        # Define 16 InvertedResidual blocks — first 8 in 3D, rest in 2D
        self.configs3D: list[InvertedResidualConfig] = [
            InvertedResidualConfig(inp=16, exp=16, out=16, kernel=3, stride=(1, 1, 1),
                                   se=False, activation='RE', width_mult=width_mult, expand=False),
            InvertedResidualConfig(inp=16, exp=64, out=24, kernel=3, stride=(1, 2, 2),
                                   se=False, activation='RE', width_mult=width_mult),
            InvertedResidualConfig(inp=24, exp=72, out=24, kernel=3, stride=(1, 1, 1),
                                   se=False, activation='RE', width_mult=width_mult),
            InvertedResidualConfig(inp=24, exp=72, out=40, kernel=5, stride=(1, 2, 2),
                                   se=True, activation='RE', width_mult=width_mult),
            InvertedResidualConfig(inp=40, exp=120, out=40, kernel=5, stride=(1, 1, 1),
                                   se=True, activation='RE', width_mult=width_mult),
            InvertedResidualConfig(inp=40, exp=120, out=40, kernel=5, stride=(1, 1, 1),
                                   se=True, activation='RE', width_mult=width_mult),
            InvertedResidualConfig(inp=40, exp=240, out=80, kernel=3, stride=(2, 2, 2),
                                   se=False, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=80, exp=200, out=80, kernel=3, stride=(1, 1, 1),
                                   se=False, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=80, exp=184, out=80, kernel=3, stride=(1, 1, 1),
                                   se=False, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=80, exp=184, out=80, kernel=3, stride=(2, 1, 1),
                                   se=False, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=80, exp=480, out=112, kernel=3, stride=(1, 1, 1),
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=112, exp=672, out=112, kernel=3, stride=(1, 1, 1),
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=112, exp=672, out=160, kernel=5, stride=(2, 2, 2),
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=160, exp=960, out=160, kernel=5, stride=(1, 1, 1),
                                   se=True, activation='HS', width_mult=width_mult),
            InvertedResidualConfig(inp=160, exp=960, out=160, kernel=5, stride=(1, 1, 1),
                                   se=True, activation='HS', width_mult=width_mult),
        ]
        self.configs2D: list[InvertedResidualConfig] = [


        ]

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels= self.configs3D[0].input_channels,
                kernel_size=3,
                stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            #nn.BatchNorm3d(self.configs3D[0].input_channels, eps=1e-8, momentum=5e-4),
            self.norm_layer(num_channels=self.configs3D[0].input_channels),
            nn.Hardswish()
        )

        #SE between 3D & 2D blocks

        #self.mid_se: SqueezeExcite3D = SqueezeExcite3D(in_channels=self.configs3D[-1].out_channels)

        self.blocks3D = nn.ModuleList()
        self.blocks2D = nn.ModuleList()

        for cfg in self.configs3D:
                self.blocks3D.append(InvertedResidual3D(cnf=cfg))
        for cfg in self.configs2D:
                self.blocks2D.append(InvertedResidual2D(cnf=cfg))

        # Final layers
        classifier_fc1_in_features = max(make_divisible(960*width_mult), 256)
        classifier_fc2_in_features = max(make_divisible(1280*width_mult), 256)
        self.final_conv = nn.Sequential(
            nn.Conv3d(make_divisible(160*width_mult), classifier_fc1_in_features,
                      kernel_size=(2,1,1), padding=0, bias=False),
            #nn.BatchNorm3d(classifier_fc1_in_features, eps=1e-8, momentum=5e-4),
            self.norm_layer(num_channels=classifier_fc1_in_features),
            nn.Hardswish(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=classifier_fc1_in_features,
                      out_features=classifier_fc2_in_features, bias=True),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(in_features=classifier_fc2_in_features, out_features=n_classes, bias=True)
        )

    def forward(self, x: Tensor):
        # x: (B, C, T, H, W)
        x = self.conv1(x)
        for block in self.blocks3D:
            x = block(x)

        #x = self.mid_se(x).sum(dim=2) #SE returns (B,C) w/weighted (T,H,W) -> sum across T

        #for block in self.blocks2D:
        #    x = block(x)

        x = self.final_conv(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x
