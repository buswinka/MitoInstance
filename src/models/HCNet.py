import torch
import torch.nn as nn
from src.models.modules.HCBlock import HCBlock
from warnings import filterwarnings

import glob
from typing import Sequence
filterwarnings("ignore", category=UserWarning)


class HCNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, complexity: int = 30):
        super(HCNet, self).__init__()

        self.conv3x3_1 = nn.Conv3d(in_channels=in_channels, out_channels=10, padding=1, kernel_size=3)
        self.bn_1 = nn.BatchNorm3d(10)

        self.strided_conv = nn.Conv3d(10, complexity, kernel_size=5, stride=(5,5,3), padding=1)
        self.bn_strided = nn.BatchNorm3d(complexity)

        self.hcblock = torch.jit.script(HCBlock(in_channels=complexity))

        self.transposed_conv = nn.ConvTranspose3d(in_channels=complexity, out_channels=complexity,
                                                  stride=(5,5,3), kernel_size=5, padding=(1, 1, 1))

        self.batch_norm_transpose = nn.BatchNorm3d(complexity)

        self.conv5x5_1 = nn.Conv3d(in_channels=complexity, out_channels=complexity*2, kernel_size=5, padding=2)
        self.bn_5x5_1 = nn.BatchNorm3d(complexity*2)

        self.conv3x3_2 = nn.Conv3d(complexity*2, complexity*3, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm3d(complexity*3)

        self.out_conv = nn.Conv3d(complexity * 3, out_channels=out_channels, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, i: int = 5) -> torch.Tensor:
        x = self.activation(self.bn_1(self.conv3x3_1(x)))
        x = self.activation(self.bn_strided(self.strided_conv(x)))
        y = torch.zeros(x.shape, device=x.device)

        for _ in range(i):
            y = self.activation(self.hcblock(torch.cat((x, y), dim=1)) + y)

        y = self.activation(self.batch_norm_transpose(self.transposed_conv(y)))
        y = self.activation(self.bn_5x5_1(self.conv5x5_1(y)))
        y = self.activation(self.bn_2(self.conv3x3_2(y)))
        y = self.out_conv(y)

        y = self.tanh(y)

        return y

class tinyHCNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, complexity: int = 30):
        super(tinyHCNet, self).__init__()

        self.conv3x3_1 = nn.Conv3d(in_channels=in_channels, out_channels=10, padding=1, kernel_size=3)
        self.bn_1 = nn.BatchNorm3d(10)

        self.strided_conv = nn.Conv3d(10, complexity, kernel_size=4, stride=4, padding=1)
        self.bn_strided = nn.BatchNorm3d(complexity)

        self.hcblock = torch.jit.script(HCBlock(in_channels=complexity))

        self.transposed_conv = nn.ConvTranspose3d(in_channels=complexity, out_channels=3*complexity,
                                                  stride=(4, 4, 4), kernel_size=(4, 4, 4), padding=(2, 2, 2))

        self.batch_norm_transpose = nn.BatchNorm3d(3 * complexity)
        self.out_conv = nn.Conv3d(complexity * 3, out_channels=out_channels, kernel_size=1, padding=1)

        self.activation = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, i: int = 5) -> torch.Tensor:
        x = self.activation(self.bn_1(self.conv3x3_1(x)))
        x = self.activation(self.bn_strided(self.strided_conv(x)))
        y = torch.zeros(x.shape, device=x.device)

        for _ in range(i):
            y = self.hcblock(torch.cat((x, y), dim=1)) + y

        y = self.activation(self.batch_norm_transpose(self.transposed_conv(y)))
        y = self.out_conv(y)

        y = self.tanh(y)

        return y
