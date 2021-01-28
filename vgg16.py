from mindspore import nn

from parameters import Parameter

layers_depth = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

'''
This class is called in loss.py, cast input pictures(style, content and generated) into features.
Loss function will compare features of different pictures after different layers.
'''


class Vgg16(nn.Cell):
    def __init__(self):
        super(Vgg16, self).__init__()
        pad_mode = Parameter.LossParams.VggParams.pad_mode
        padding = Parameter.LossParams.VggParams.padding
        has_bias = Parameter.LossParams.VggParams.has_bias
        weight_init = Parameter.LossParams.VggParams.weight_init
        self._conv00 = _Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv01 = _Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv10 = _Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv11 = _Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv20 = _Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv21 = _Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv22 = _Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv30 = _Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv31 = _Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv32 = _Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv40 = _Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv41 = _Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._conv42 = _Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._max_pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def get_outputs_in_different_layers(self, x):
        layers2outputs = {}
        x = self._conv00(x)
        x = self._conv01(x)
        layers2outputs["conv0"] = x
        x = self._max_pool0(x)
        layers2outputs["pool0"] = x
        x = self._conv10(x)
        x = self._conv11(x)
        layers2outputs["conv1"] = x
        x = self._max_pool1(x)
        layers2outputs["pool1"] = x
        x = self._conv20(x)
        x = self._conv21(x)
        x = self._conv22(x)
        layers2outputs["conv2"] = x
        x = self._max_pool2(x)
        layers2outputs["pool2"] = x
        x = self._conv30(x)
        x = self._conv31(x)
        x = self._conv32(x)
        layers2outputs["conv3"] = x
        x = self._max_pool3(x)
        layers2outputs["pool3"] = x
        x = self._conv40(x)
        x = self._conv41(x)
        x = self._conv42(x)
        layers2outputs["conv4"] = x
        x = self._max_pool4(x)
        layers2outputs["pool4"] = x
        return layers2outputs


class _Conv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, pad_mode, padding, has_bias, weight_init):
        super(_Conv2d, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                               pad_mode=pad_mode, padding=padding, has_bias=has_bias, weight_init=weight_init)
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU()

    def construct(self, x):
        return self._relu(self._batch_norm(self._conv(x)))
