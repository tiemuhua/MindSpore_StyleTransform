from mindspore import nn
from mindspore.common.initializer import Normal, Constant


class Conv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, has_bias=False,
                 activation_fn="relu"):
        super(Conv2d, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               pad_mode, padding, weight_init=Normal(), has_bias=has_bias)
        self._bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        if activation_fn == "relu":
            self._activation_fn = nn.ReLU()
        else:
            self._activation_fn = nn.Sigmoid()

    def construct(self, x):
        return self._activation_fn(self._bn(self._conv(x)))


class Conv2dBatchNormReLU(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, pad_mode='same', padding=0, has_bias=False):
        super(Conv2dBatchNormReLU, self).__init__()
        # TODO perhaps there should be a random initializer
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               pad_mode, padding, weight_init=Normal(), has_bias=has_bias, bias_init=Constant())
        self._bn = _BatchNorm2dDynamic(out_channels, eps=0.001, momentum=0.9997)
        self._relu = nn.ReLU()

    def construct(self, x, beta, gamma):
        return self._relu(self._bn(self._conv(x), beta, gamma))


class _BatchNorm2dDynamic(nn.Cell):
    # TODO 这里似乎应该有eps和momentum两个参数？
    def __init__(self, out_channels, eps, momentum):
        super(_BatchNorm2dDynamic, self).__init__()

    # TODO x 是一个四维张量，[batch_norm, channel, height, width]
    def construct(self, x, beta, gamma):
        pass
