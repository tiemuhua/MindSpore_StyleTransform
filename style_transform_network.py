from mindspore import nn, ops

from conv2d import Conv2dReLU, Conv2dSigmoid, Conv2dBatchNormReLU

'''
This class takes style embedded vector and content picture as input, output generated image.
This class will be called in file "main_network.py" class "MainNetwork"
'''

residual_block_depth = 128  # TODO 此参数能不能在运行时确定
residual_block_kernel_size = 3


class StyleTransformNetwork(nn.Cell):
    def __init__(self):
        super(StyleTransformNetwork, self).__init__()
        # TODO 感觉好像有点不对劲，beta和gamma的维度能对上号吗？
        # TODO gamma和beta是在运行过程中动态调整的，不能当成入参
        self._conv0 = Conv2dBatchNormReLU(in_channels=1, out_channels=32, kernel_size=9, stride=1)
        self._conv1 = Conv2dBatchNormReLU(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self._conv2 = Conv2dBatchNormReLU(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self._residual0 = _ResidualBlock()
        self._residual1 = _ResidualBlock()
        self._residual2 = _ResidualBlock()
        self._residual3 = _ResidualBlock()
        self._residual4 = _ResidualBlock()
        self._up_sampling0 = _UpSampling(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self._up_sampling1 = _UpSampling(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self._up_sampling2 = _UpSampling(in_channels=32, out_channels=3, kernel_size=9, stride=9,
                                         activation_fn="sigmoid")

    def construct(self, *inputs):
        # TODO 如何construct？？？？如何动态添加normalization参数？
        x = inputs[0]
        beta = inputs[1]
        gamma = inputs[2]
        x = self._conv0(x,beta=beta,gamma=gamma)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._residual0(x)
        x = self._residual1(x)
        x = self._residual2(x)
        x = self._residual3(x)
        x = self._residual4(x)
        x = self._up_sampling0(x)
        x = self._up_sampling1(x)
        x = self._up_sampling2(x)
        return x


'''
本类在类StyleTransferNetwork、类_UpSampling、类_ResidualBlock中被调用
参数：
    activation_fn：string，激活函数种类，可以是"relu"或者"sigmoid"，默认"relu"
'''


class _Conv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_fn="relu"):
        super(_Conv2d, self).__init__()
        padding = kernel_size // 2
        self._pad = nn.Pad(paddings=((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="REFLECT")
        if activation_fn == "relu":
            self._conv2d = Conv2dReLU(in_channels, out_channels, kernel_size, stride)
        else:
            self._conv2d = Conv2dSigmoid(in_channels, out_channels, kernel_size, stride)

    def construct(self, x):
        return self._conv2d(self._pad(x))


'''
This class is called in class StyleTransformNetwork
'''


class _UpSampling(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_fn="relu"):
        super(_UpSampling, self).__init__()
        self._conv2d = _Conv2d(in_channels, out_channels, kernel_size, 1, activation_fn)
        self._stride = stride

    def construct(self, x):
        shape = x.shape
        height = shape[1]
        width = shape[2]
        resize_nearest_neighbor = ops.operations.array_ops.ResizeNearestNeighbor(
            (self._stride * height, self._stride * width)
        )
        up_sampled_img = resize_nearest_neighbor(x)
        return self._conv2d(up_sampled_img)


'''
This class is called in class StyleTransformNetwork
'''


class _ResidualBlock(nn.Cell):
    def __init__(self, activation_fn="relu"):
        super(_ResidualBlock, self).__init__()
        # TODO residual_block_depth depends on input image size
        self.conv2d0 = _Conv2d(residual_block_depth, residual_block_depth, residual_block_kernel_size, 1, activation_fn)
        self.conv2d1 = _Conv2d(residual_block_depth, residual_block_depth, residual_block_kernel_size, 1, activation_fn)

    def construct(self, inputs):
        return self.conv2d0(self.conv2d1(inputs)) + inputs
