from mindspore import nn, ops

import vgg16
from parameters import Parameter

'''
Loss类对外可见的接口有且只有两个，一个是构造函数，一个是total_loss。
total_loss函数将会调用_smooth_loss、_style_loss和_content_loss
'''


class Loss(nn.loss.loss._Loss):
    """"""
    _vgg = vgg16.Vgg16()

    def __init__(self):
        super(Loss, self).__init__()

    '''
        本函数计算总损失函数
        $$total_loss = style_weight*_style_loss + content_weight*_content_loss + smooth_weight*_smooth_loss$$
    '''

    def construct(self, generated_pictures, target):
        content_pictures = target["content"]
        style_pictures = target["style"]
        generated_features_vgg = Loss._vgg(generated_pictures)
        content_features_vgg = Loss._vgg(content_pictures)
        style_features_vgg = Loss._vgg(style_pictures)
        return Parameter.LossParams.smooth_weight * _smooth_loss(generated_pictures) \
               + Parameter.LossParams.content_weight * _content_loss(generated_features_vgg, content_features_vgg) \
               + Parameter.LossParams.style_weight * _style_loss(generated_features_vgg, style_features_vgg)


"""
This function calculates $$sum_ij (x[i][j]-x[i][j-1])^2+(x[i][j]-x[i-1][j])^2$$.
Minimize this cost can make generated picture more smooth
This function will be called in Loss.total_loss()
Args:
    generated_pictures: 4D ms.Tensor, [batch2, height, width, channel=3]
return:
    smooth_loss: float
本函数计算光滑项损失函数，即$$sum_ij (x[i][j]-x[i][j-1])^2+(x[i][j]-x[i-1][j])^2$$.
最小化此项可使生成的图片尽可能光滑。
本函数被total_loss函数调用
参数:
    generated_pictures: 4D ms.Tensor, [batch, width, height, channel]
返回值:
    smooth_loss: float
"""


def _smooth_loss(generated_pictures):
    loss = nn.MSELoss()
    horizontal_diff_loss = loss(generated_pictures[:, 1:, :, :], generated_pictures[:, :-1, :, :])
    vertical_diff_loss = loss(generated_pictures[:, :, 1:, :], generated_pictures[:, :, :-1, :])
    return horizontal_diff_loss + vertical_diff_loss


'''
This function calculates the content difference between the origin content pictures and the generated pictures.
This function will be called in Loss.total_loss()
Args:
    generated_pictures: 4D ms.Tensor, [batch1, height, width, channel=3]
    content_pictures: 4D ms.Tensor, [batch2, height, width, channel=3]
return:
    style_loss: float
'''


def _content_loss(generated_features_vgg, content_features_vgg):
    total_content_loss = 0
    for layer_name, weight in Parameter.LossParams.content_layers_weight.items():
        content_feature = content_features_vgg[layer_name]
        generated_feature = generated_features_vgg[layer_name]
        reduce_mean = ops.ReduceMean()
        total_content_loss += weight * reduce_mean((content_feature - generated_feature) ** 2)
    return total_content_loss


'''
This function calculates the style difference between the origin style pictures and the generated pictures.
This function will be called in Loss.total_loss()
Args:
    generated_pictures: 4D ms.Tensor, [batch1, height, width, channel=3]
    style_pictures: 4D ms.Tensor, [batch2, height, width, channel=3]
return:
    style_loss: float
本函数计算原风格图片和生成的图片之间的风格差异，被Loss.total_loss()调用
参数：
    generated_pictures: 4维ms.Tensor, [batch2, height, width, channel=3]
    style_pictures: 4维ms.Tensor, [batch2, height, width, channel=3]
返回值：
    style_loss: float
'''


def _style_loss(generated_features_vgg, style_features_vgg):
    total_style_loss = 0
    for layer_name, weight in Parameter.LossParams.style_layers_weight.items():
        reduce_mean = ops.ReduceMean()
        style_feature = style_features_vgg[layer_name]
        generated_feature = generated_features_vgg[layer_name]
        loss = reduce_mean(gram_matrix(style_feature - generated_feature) ** 2)
        total_style_loss += loss * weight
    return total_style_loss


def gram_matrix(tenser4d):
    batch_size, width, height, channel = tenser4d.shape
    denominator = height * width
    reshape = ops.Reshape()
    flattened_tensor = reshape(tenser4d, (batch_size, width * height, channel))
    mat_mul = nn.layer.math.MatMul()
    transpose = ops.Transpose()
    return mat_mul(transpose(flattened_tensor), flattened_tensor) / denominator
