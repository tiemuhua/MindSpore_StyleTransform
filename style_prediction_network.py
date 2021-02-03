from mindspore import nn, ops

import inception_v3
from conv2d import Conv2d
from style_transform_network import transform_network_depths

'''
This class can cast the style pictures into embedded style vector beta and gamma.
First we will employ a pretrained Inception-v3 architecture to calculate a feature vector with the dimension of 768.
Then we will apply two fully connected layers on top of the Inception-v3 architecture to predict the final embedding S~.
transformer StyleNorm
'''

inception_v3_output_channel = 768

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
        self._fully_connected_layer1 = Conv2d(in_channels=inception_v3_output_channel,
                                              out_channels=bottle_neck_depth, kernel_size=1)
        self._fully_connected_layer_beta = []
        self._fully_connected_layer_gamma = []
        for depth in transform_network_depths:
            self._fully_connected_layer_beta.append(
                Conv2d(in_channels=bottle_neck_depth, out_channels=depth, kernel_size=1))
            self._fully_connected_layer_gamma.append(
                Conv2d(in_channels=bottle_neck_depth, out_channels=depth, kernel_size=1))

    '''
    参数：
        images: 4D张量，shape(images) = (batch_size, width, height, channel=3)
            为输入的风格画
    '''

    def construct(self, style_img):
        inception_v3_output = self._inception_v3(style_img)
        print("inception v3 output finished")
        reduce_mean = ops.ReduceMean(keep_dims=True)
        inception_v3_output_reduce_mean = reduce_mean(inception_v3_output, (2, 3))
        print("reduce mean finished")
        print("inception_v3_output_reduce_mean.shape")
        print(inception_v3_output_reduce_mean.shape)
        bottle_neck_feature = self._fully_connected_layer1(inception_v3_output_reduce_mean)
        betas = []
        gammas = []
        for i in range(0, len(self._fully_connected_layer_beta)):
            betas.append(self._fully_connected_layer_beta[i](bottle_neck_feature))
            gammas.append(self._fully_connected_layer_gamma[i](bottle_neck_feature))
        return betas, gammas
