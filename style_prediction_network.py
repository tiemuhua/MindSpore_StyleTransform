import inception_v3
from mindspore import nn, ops

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

  Args:
    style_input_: Tensor. Batch of style input images.
    activation_names: string. 
        Scope names of the activations of the transformer network which are used to apply style normalization.
    activation_depths: 
        Shapes of the activations of the transformer network which are used to apply style normalization.
    is_training: bool. Is it training phase or not?
    trainable: bool. Should the parameters be marked as trainable?
    inception_end_point: string. 
        Specifies the endpoint to construct the inception_v3 network up to. 
        This network is part of the style prediction network.
    style_prediction_bottleneck: int. 
        Specifies the bottleneck size in the number of parameters of the style embedding.
    reuse: bool. Whether to reuse model parameters. Defaults to False.

  Returns:
    Tensor for the output of the style prediction network, 
    Tensor for the bottleneck of style parameters of the style prediction network.
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
        self._fully_connected_layer1 = nn.Conv2d(in_channels=3, out_channels=bottle_neck_depth,kernel_size=1)
        self._fully_connected_layer_beta = {}
        self._fully_connected_layer_gamma = {}
        print(len(style_vector_depths))
        for i in range(0, len(style_vector_depths)):
            self._fully_connected_layer_beta[i] \
                = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=style_vector_depths[i],kernel_size=1)
            self._fully_connected_layer_gamma[i] \
                = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=style_vector_depths[i],kernel_size=1)

    '''
    参数：
        images: 4D张量，shape(images) = (batch_size, width, height, channel=3)
            为输入的风格画
    '''

    def construct(self, *inputs, **kwargs):
        inception_v3_output = self._inception_v3(inputs)
        reduce_mean = ops.ReduceMean()
        inception_v3_output_reduce_mean = reduce_mean(inception_v3_output, (1, 2))
        bottle_neck_feature = self._fully_connected_layer1(inception_v3_output_reduce_mean)
        print(bottle_neck_feature)
        beta = []
        gamma = []
        '''
        for i in range(0, len(style_vector_depths)):
            beta[i] = self._fully_connected_layer_beta[i](bottle_neck_feature)
            gamma[i] = self._fully_connected_layer_gamma[i](bottle_neck_feature)
            '''
        
        #return beta, gamma
        return 0

from mindspore import Tensor
import mindspore
import file_operations as f
import numpy as np
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
#x=Tensor(f.load_np_image('eiffel_tower.jpg'),dtype=mindspore.int32)
x=Tensor(np.ones([1,3,3,3]),mindspore.float32)
nett = nn.Conv2d(3, 2, 3, has_bias=False, weight_init='normal')
y=nett(x)
print(x.dtype)
net=StylePredictionNetwork()
x=net(x)

