"""
This MobileNetV2 implementation is modified from the following repository:
https://github.com/tonylins/pypaddle-mobilenet-v2
"""

import paddle.nn as nn
import math
# from .utils import load_url
# from .segm_lib.nn import SynchronizedBatchNorm2D

# BatchNorm2D = SynchronizedBatchNorm2D
BatchNorm2D = nn.BatchNorm2D

__all__ = ['mobilenetv2']


model_urls = {
    'mobilenetv2': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/mobilenet_v2.pth.tar',
}


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 3, stride, 1, bias_attr=False),
        BatchNorm2D(oup),
        nn.ReLU6()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
        BatchNorm2D(oup),
        nn.ReLU6()
    )


class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_axis = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2D(hidden_axis, hidden_axis, 3, stride, 1, groups=hidden_axis, bias_attr=False),
                BatchNorm2D(hidden_axis),
                nn.ReLU6(),
                # pw-linear
                nn.Conv2D(hidden_axis, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2D(inp, hidden_axis, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_axis),
                nn.ReLU6(),
                # dw
                nn.Conv2D(hidden_axis, hidden_axis, 3, stride, 1, groups=hidden_axis, bias_attr=False),
                BatchNorm2D(hidden_axis),
                nn.ReLU6(),
                # pw-linear
                nn.Conv2D(hidden_axis, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Layer):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias_attr is not None:
                    m.bias_attr.data.zero_()
            elif isinstance(m, BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias_attr.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias_attr.data.zero_()


def mobilenetv2(pretrained=False, **kwargs):
    """Constructs a MobileNet_V2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2(n_class=1000, **kwargs)
    # if pretrained:
    #     model.load_state_dict(load_url(model_urls['mobilenetv2']), strict=False)
    return model