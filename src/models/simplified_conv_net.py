from dataclasses import dataclass

from torch import nn
from typing import Type, Optional, Callable

from src.models import layers
from .layers.basic.first_channels import FirstChannels
from .layers.basic.zero_channel_concatenation import ZeroChannelConcatenation


BASE_WIDTHS = {"XS": 16, "S": 32, "M": 64, "L": 128}


@dataclass
class SimplifiedConvNetHyperparameters:
    get_activation: Type[nn.Module] = layers.MaxMin
    get_conv: Type[nn.Module] = layers.Conv2d
    get_conv_first: Type[nn.Module] = None
    get_conv_head: Type[nn.Module] = None
    get_linear: Type[nn.Linear] = None

    # Size:
    base_width: int = 16
    nrof_blocks: int = 5
    nrof_layers_per_block: int = 5
    kernel_size: int = 3

    # Classification head:
    nrof_classes: Optional[int] = 10

    def __post_init__(self):
        if self.get_conv_first is None:
            self.get_conv_first = self.get_conv
        if self.get_conv_head is None:
            self.get_conv_head = self.get_conv


def get_conv_block(get_conv: Type[nn.Module],
                   get_activation: Callable,
                   in_channels: int,
                   length: int,
                   kernel_size: int):
    block = nn.Sequential()
    for _ in range(length):
        conv = get_conv(in_channels, in_channels, kernel_size)
        block.append(conv)
        block.append(get_activation())
    block.first_channels = FirstChannels(in_channels // 2)
    block.pooling = nn.PixelUnshuffle(2)
    return block


def get_conv_net(hp: SimplifiedConvNetHyperparameters) -> nn.Sequential:
    conv_net = nn.Sequential()
    conv_net.zero_concat = ZeroChannelConcatenation(hp.base_width)
    conv_net.first_conv = hp.get_conv_first(
        in_channels=hp.base_width,
        out_channels=hp.base_width,
        kernel_size=1
    )
    conv_net.first_activation = hp.get_activation()

    kernel_sizes = [hp.kernel_size for _ in range(hp.nrof_blocks)]
    kernel_sizes[-1] = 1  # 2x2 blocks do not allow kernel size >= 3.
    for i in range(hp.nrof_blocks):
        block = get_conv_block(
            hp.get_conv,
            hp.get_activation,
            hp.base_width * 2 ** i,
            hp.nrof_layers_per_block,
            kernel_sizes[i],
        )
        conv_net.append(block)

    # conv_net.pooling = nn.AdaptiveAvgPool2d(1)

    final_width = hp.base_width * 2 ** hp.nrof_blocks
    if hp.get_linear is not None:
        conv_net.flatten = nn.Flatten()
        conv_net.head = hp.get_linear(final_width, final_width)
    else:
        conv_net.head = hp.get_conv_head(final_width, final_width, 1)
        conv_net.flatten = nn.Flatten()

    conv_net.first_channels = FirstChannels(hp.nrof_classes)
    return conv_net


def create(*args, **kwargs) -> nn.Sequential:
    hp = SimplifiedConvNetHyperparameters(*args, **kwargs)
    return get_conv_net(hp)


def create_from_size(size_name: str, nrof_blocks: int = 5, **kwargs):
    # nrof_blocks = integer_log2(input_resolution)
    final_width = BASE_WIDTHS[size_name] * 2**5
    base_width = final_width // 2**nrof_blocks

    return create(base_width=base_width, nrof_blocks=nrof_blocks, **kwargs)


def integer_log2(x: int) -> int:
    return (x - 1).bit_length()
