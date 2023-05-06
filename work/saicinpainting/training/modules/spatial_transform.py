import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from kornia.transform import rotate


class LearnableSpatialTransformWrapper(nn.Layer):
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = paddle.rand(1) * angle_init_range
        if train_angle:
            self.angle = paddle.create_parameter(self.angle)
        self.pad_coef = pad_coef

    def forward(self, x):
        if paddle.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans)
            return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

    def transform(self, x):
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        x_padded_rotated = rotate(x_padded, angle=self.angle)
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)

        y_padded = rotate(y_padded_rotated, angle=-self.angle)
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
        return y


if __name__ == '__main__':
    layer = LearnableSpatialTransformWrapper(nn.Identity())
    x = paddle.arange(2* 3 * 15 * 15).view(2, 3, 15, 15).float()
    y = layer(x)
    assert x.shape == y.shape
    assert paddle.allclose(x[:, :, 1:, 1:][:, :, :-1, :-1], y[:, :, 1:, 1:][:, :, :-1, :-1])
    print('all ok')
