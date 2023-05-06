import abc
from typing import Tuple, List

import paddle
import paddle.nn as nn

from saicinpainting.training.modules.depthwise_sep_conv import DepthWiseSeperableConv
from saicinpainting.training.modules.multidilated_conv import MultidilatedConv


class BaseDiscriminator(nn.Layer):
    @abc.abstractmethod
    def forward(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, List[paddle.Tensor]]:
        """
        Predict scores and get intermediate activations. Useful for feature matching loss
        :return tuple (scores, list of intermediate activations)
        """
        raise NotImplemented()


def get_conv_block_ctor(kind='default'):
    if not isinstance(kind, str):
        return kind
    if kind == 'default':
        return nn.Conv2D
    if kind == 'depthwise':
        return DepthWiseSeperableConv   
    if kind == 'multidilated':
        return MultidilatedConv
    raise ValueError(f'Unknown convolutional block kind {kind}')


def get_norm_layer(kind='bn'):
    if not isinstance(kind, str):
        return kind
    if kind == 'bn':
        return nn.BatchNorm2D
    if kind == 'in':
        return nn.InstanceNorm2D
    raise ValueError(f'Unknown norm block kind {kind}')


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class SimpleMultiStepGenerator(nn.Layer):
    def __init__(self, steps: List[nn.Layer]):
        super().__init__()
        self.steps = nn.LayerList(steps)

    def forward(self, x):
        cur_in = x
        outs = []
        for step in self.steps:
            cur_out = step(cur_in)
            outs.append(cur_out)
            cur_in = paddle.concat((cur_in, cur_out), axis =1)
        return paddle.concat(outs[::-1], axis =1)

def deconv_factory(kind, ngf, mult, norm_layer, activation, max_features):
    if kind == 'convtranspose':
        return [nn.Conv2DTranspose(min(max_features, ngf * mult), 
                    min(max_features, int(ngf * mult / 2)), 
                    kernel_size=3, stride=2, padding=1, output_padding=1),
                    norm_layer(min(max_features, int(ngf * mult / 2))), activation]
    elif kind == 'bilinear':
        return [nn.Upsample(scale_factor=2, mode='bilinear'),
                DepthWiseSeperableConv(min(max_features, ngf * mult), 
                    min(max_features, int(ngf * mult / 2)), 
                    kernel_size=3, stride=1, padding=1), 
                norm_layer(min(max_features, int(ngf * mult / 2))), activation]
    else:
        raise Exception(f"Invalid deconv kind: {kind}")