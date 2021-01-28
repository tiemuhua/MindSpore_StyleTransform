from mindspore import nn, ops

import inception_v3
from conv2d import Conv2dReLU

'''
This class can cast the style pictures into embedded style vector beta and gamma.
First we will employ a pretrained Inception-v3 architecture to calculate a feature vector with the dimension of 768.
Then we will apply two fully connected layers on top of the Inception-v3 architecture to predict the final embedding S~.
transformer StyleNorm
'''

style_vector_scope_names = [
    'residual/residual1/conv1',
    'residual/residual1/conv2',
    'residual/residual2/conv1',
    'residual/residual2/conv2',
    'residual/residual3/conv1',
    'residual/residual3/conv2',
    'residual/residual4/conv1',
    'residual/residual4/conv2',
    'residual/residual5/conv1',
    'residual/residual5/conv2',
    'expand/conv1/conv',
    'expand/conv2/conv',
    'expand/conv3/conv']
style_vector_depths = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3]

"""
Maps style images to the style embeddings (beta and gamma parameters).
This class will be called in file "main_network.py" class "MainNetwork"
"""


class StylePredictionNetwork(nn.Cell):
    ''''''
    '''
    参数：
        bottle_neck_depth: int
            第一个全连接层的输出特征的深度。默认值为100
    '''

    def __init__(self, bottle_neck_depth=100):
        super(StylePredictionNetwork, self).__init__()
        self._inception_v3 = inception_v3.InceptionV3()
        self._fully_connected_layer1 = Conv2dReLU(in_channels=3, out_channels=bottle_neck_depth, kernel_size=1)
        self._fully_connected_layer_beta = []
        self._fully_connected_layer_gamma = []
        self._squeeze = ops.Squeeze((1, 2))
        for i in range(0, len(style_vector_depths)):
            self._fully_connected_layer_beta[i] \
                = Conv2dReLU(in_channels=bottle_neck_depth, out_channels=style_vector_depths[i], kernel_size=1)
            self._fully_connected_layer_gamma[i] \
                = Conv2dReLU(in_channels=bottle_neck_depth, out_channels=style_vector_depths[i], kernel_size=1)

    '''
    参数：
        images: 4D张量，shape(images) = (batch_size, width, height, channel=3)
            为输入的风格画
    '''

    def get_style_embedded_vector(self, *inputs, **kwargs):
        inception_v3_output = self._inception_v3(inputs)
        reduce_mean = ops.ReduceMean()
        inception_v3_output_reduce_mean = reduce_mean(inception_v3_output, (1, 2))
        bottle_neck_feature = self._fully_connected_layer1(inception_v3_output_reduce_mean)
        beta = []
        gamma = []
        for i in range(0, len(style_vector_depths)):
            beta[i] = self._fully_connected_layer_beta[i](bottle_neck_feature)
            beta[i] = self._squeeze(beta[i])
            gamma[i] = self._fully_connected_layer_gamma[i](bottle_neck_feature)
            gamma[i] = self._squeeze(gamma[i])
        return beta, gamma
