from mindspore import nn
from mindspore.common.initializer import XavierUniform


class Conv2dReLU(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, has_bias=False):
        super(Conv2dReLU, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               pad_mode, padding, weight_init=XavierUniform(), has_bias=has_bias)
        self._bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self._relu = nn.ReLU()

    def construct(self, x):
        x = self._conv(x)
        x = self._bn(x)
        x = self._relu(x)
        return x


class Conv2dSigmoid(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, has_bias=False):
        super(Conv2dSigmoid, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               pad_mode, padding, weight_init=XavierUniform(), has_bias=has_bias)
        self._bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self._sigmoid = nn.Sigmoid()

    def construct(self, *inputs, **kwargs):
        return self._sigmoid(self._bn(self._conv(inputs)))


class Conv2dBatchNormReLU(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, pad_mode='same', padding=0, has_bias=False):
        super(Conv2dBatchNormReLU, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               pad_mode, padding, weight_init=XavierUniform(), has_bias=has_bias)
        self._bn = nn.BatchNorm2d(out_channels, beta_init=beta, gamma_init=gamma, eps=0.001, momentum=0.9997)
        self._relu = nn.ReLU()

    def construct(self, *inputs, **kwargs):
        return self._relu(self._bn(self._conv))
