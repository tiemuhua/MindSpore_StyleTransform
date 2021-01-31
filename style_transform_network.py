from mindspore import nn, ops

from conv2d import Conv2d

'''
This class takes style embedded vector and content picture as input, output generated image.
This class will be called in file "main_network.py" class "MainNetwork"
'''

residual_block_depth = 128
residual_block_kernel_size = 3

transform_network_names_2_depth = {
    'residual0/conv0': 128,
    'residual0/conv1': 128,
    'residual1/conv0': 128,
    'residual1/conv1': 128,
    'residual2/conv0': 128,
    'residual2/conv1': 128,
    'residual3/conv0': 128,
    'residual3/conv1': 128,
    'residual4/conv0': 128,
    'residual4/conv1': 128,
    'up_sampling0': 64,
    'up_sampling1': 32,
    'up_sampling2': 3}
# style vector depth is equal to the transform network depth
transform_network_depths = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3]


class StyleTransformNetwork(nn.Cell):
    def __init__(self):
        super(StyleTransformNetwork, self).__init__()
        self._conv0 = Conv2d(in_channels=1, out_channels=32, kernel_size=9, stride=1)
        self._conv1 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self._conv2 = Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self._residual0 = _ResidualBlock()
        self._residual1 = _ResidualBlock()
        self._residual2 = _ResidualBlock()
        self._residual3 = _ResidualBlock()
        self._residual4 = _ResidualBlock()
        self._up_sampling0 = _UpSampling(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self._up_sampling1 = _UpSampling(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self._up_sampling2 = _UpSampling(in_channels=32, out_channels=3, kernel_size=9, stride=9,
                                         activation_fn="sigmoid")

    def construct(self, x, beta, gamma):
        x = self._conv0(x)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._residual0(x, beta0=beta[0], gamma0=gamma[0], beta1=beta[1], gamma1=gamma[1])
        x = self._residual1(x, beta0=beta[2], gamma0=gamma[2], beta1=beta[3], gamma1=gamma[3])
        x = self._residual2(x, beta0=beta[4], gamma0=gamma[4], beta1=beta[5], gamma1=gamma[5])
        x = self._residual3(x, beta0=beta[6], gamma0=gamma[6], beta1=beta[7], gamma1=gamma[7])
        x = self._residual4(x, beta0=beta[8], gamma0=gamma[8], beta1=beta[9], gamma1=gamma[9])
        x = self._up_sampling0(x, beta=beta[10], gamma=gamma[10])
        x = self._up_sampling1(x, beta=beta[11], gamma=gamma[11])
        x = self._up_sampling2(x, beta=beta[12], gamma=gamma[12])
        return x


# TODO how to do the normalization????
class _DynamicNormalize(nn.Cell):
    def __init__(self, out_channels, eps, momentum):
        super(_DynamicNormalize, self).__init__()
        a=nn.BatchNorm2d

    def construct(self, x, beta, gamma):
        pass


'''
This class is called in class StyleTransformNetwork
'''


class _ResidualBlock(nn.Cell):
    def __init__(self, activation_fn="relu"):
        super(_ResidualBlock, self).__init__()
        # TODO residual_block_depth depends on input image size
        self.conv2d0 = _PadConv2dBatchNorm(residual_block_depth, residual_block_depth, residual_block_kernel_size, 1,
                                           activation_fn)
        self.conv2d1 = _PadConv2dBatchNorm(residual_block_depth, residual_block_depth, residual_block_kernel_size, 1,
                                           activation_fn)

    def construct(self, x, beta0, gamma0, beta1, gamma1):
        return self.conv2d1(self.conv2d0(x, beta0, gamma0), beta1, gamma1) + x


'''
This class is called in class StyleTransformNetwork
'''


class _UpSampling(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_fn="relu"):
        super(_UpSampling, self).__init__()
        self._conv2d = _PadConv2dBatchNorm(in_channels, out_channels, kernel_size, 1, activation_fn)
        self._stride = stride

    def construct(self, x, beta, gamma):
        shape = x.shape
        height = shape[1]
        width = shape[2]
        resize_nearest_neighbor = ops.operations.array_ops.ResizeNearestNeighbor(
            (self._stride * height, self._stride * width)
        )
        up_sampled_img = resize_nearest_neighbor(x)
        return self._conv2d(up_sampled_img, beta, gamma)


'''
本类在类_UpSampling、类_ResidualBlock中被调用
参数：
    activation_fn：string，激活函数种类，可以是"relu"或者"sigmoid"，默认"relu"
'''


class _PadConv2dBatchNorm(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_fn="relu"):
        super(_PadConv2dBatchNorm, self).__init__()
        padding = kernel_size // 2
        self._pad = nn.Pad(paddings=((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="REFLECT")
        self._conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self._normalize=_DynamicNormalize()
        if activation_fn == "relu":
            self._activation_fn = nn.ReLU()
        else:
            self._activation_fn = nn.Sigmoid()

    def construct(self, x, beta, gamma):
        return self._activation_fn(self._normalize(self._conv2d(self._pad(x))))
