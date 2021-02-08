import torch
import torch.nn as nn
from src.models.modules.StackedDilation import StackedDilation
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)


class HCBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(HCBlock, self).__init__()

        self.conv1x1 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2, stride=1, dilation=1)
        self.activation = nn.LeakyReLU()

        self.batch_norm_conv = nn.BatchNorm3d(in_channels)
        self.batch_norm_group = nn.BatchNorm3d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.activation(self.batch_norm_conv(self.conv1x1(x)))
        x = self.activation(self.batch_norm_group(self.conv3x3(x)))

        return x
