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


# class SimplifiedConvNet(nn.Sequential):
#     """
#     Similar to LipConvnet, e.g. from SOC
#     (https://arxiv.org/pdf/2105.11417.pdf)
#     """
#
#     def __init__(self, *args, seed: Union[int, None] = None, **kwargs):
#         if seed is not None:
#             torch.manual_seed(seed)
#         self.hp = SimplifiedConvNetHyperparameters(*args, **kwargs)
#         layers = get_layers(self.hp)
#         super().__init__(OrderedDict(layers))


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
            kernel_sizes[i]
        )
        conv_net.append(block)

    # conv_net.pooling = nn.AdaptiveAvgPool2d(1)

    final_width = hp.base_width * 2 ** hp.nrof_blocks
    if hp.get_linear is not None:
        conv_net.flatten = nn.Flatten(),
        conv_net.head = hp.get_linear(final_width, final_width)
    else:
        conv_net.head = hp.get_conv_head(final_width, hp.nrof_classes, 1)
        conv_net.flatten = nn.Flatten()

    conv_net.first_channels = FirstChannels(hp.nrof_classes)
    return conv_net


def create(*args, **kwargs) -> nn.Sequential:
    hp = SimplifiedConvNetHyperparameters(*args, **kwargs)
    return get_conv_net(hp)


def create_from_size(size_name: str, input_resolution: int = 32, **kwargs):
    nrof_blocks = integer_log2(input_resolution)
    base_width_32 = BASE_WIDTHS[size_name]
    base_width = base_width_32 * 2**5 // 2**nrof_blocks
    return create(base_width=base_width, nrof_blocks=nrof_blocks, **kwargs)


def integer_log2(x: int) -> int:
    return (x - 1).bit_length()


# create_xs = partial(create, base_width=16)
# create_s = partial(create, base_width=32)
# create_m = partial(create, base_width=64)
# create_l = partial(create, base_width=128)
#
# create_ks1_xs = partial(create, base_width=16, kernel_size=1)
# create_ks1_s = partial(create, base_width=32, kernel_size=1)
# create_ks1_m = partial(create, base_width=64, kernel_size=1)
# create_ks1_l = partial(create, base_width=128, kernel_size=1)
#
# create_64x64_xs = partial(create, base_width=8, nrof_blocks=6)
# create_64x64_s = partial(create, base_width=16, nrof_blocks=6)
# create_64x64_m = partial(create, base_width=32, nrof_blocks=6)
# create_64x64_l = partial(create, base_width=64, nrof_blocks=6)
#
# create_256x256_xs = partial(create, base_width=2, nrof_blocks=8)
# create_256x256_s = partial(create, base_width=4, nrof_blocks=8)
# create_256x256_m = partial(create, base_width=8, nrof_blocks=8)
# create_256x256_l = partial(create, base_width=16, nrof_blocks=8)



