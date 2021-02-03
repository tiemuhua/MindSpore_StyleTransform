from mindspore import Tensor
import mindspore as ms
from style_transform_network import StyleTransformNetwork
from mindspore import context
import file_operations as f
from style_prediction_network import StylePredictionNetwork

context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
style_img = Tensor(f.load_np_image('La_forma.jpg'), ms.float32)
print("style_img shape")
print(style_img.shape)
net = StylePredictionNetwork()
print("StylePredictionNetwork construction finished")
beta, gamma = net(style_img)

content_img = Tensor(f.load_np_image('eiffel_tower.jpg'), ms.float32)
net = StyleTransformNetwork()
generated_img = net(content_img, beta, gamma)
print(generated_img)
# x = Tensor(np.ones([1, 3, 256, 256]), mindspore.float32)
# net = contract_conv(in_channels=3, out_channels=32, kernel_size=9, stride=1)
# y = net(x)
# print(y)
#
# net2 = contract()
# y2 = net2(x)
# print(y2)
#
# x2 = Tensor(np.ones([1, 128, 14, 14]), mindspore.float32)
# net3 = residual_block(128, 128)
# beta = Tensor(np.ones([1, 128, 1, 1]), mindspore.float32)
# y3 = net3(x2, beta, beta, beta, beta)
# print(y3)
#
# net4 = residual(128, 128)
# betas = [beta, beta, beta, beta, beta, beta, beta, beta, beta, beta]
# gammas = [beta, beta, beta, beta, beta, beta, beta, beta, beta, beta]
# y4 = net4(x2, betas, gammas)
# print(y4)
#
# y5 = net2(x)
# y5 = net4(y5, betas, gammas)
# print(y5)
