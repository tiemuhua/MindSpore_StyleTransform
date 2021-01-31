import mindspore as ms
from mindspore import Tensor
from mindspore import context
import numpy as np

import file_operations as f
import style_prediction_network as predict

context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
x = Tensor(f.load_np_image('eiffel_tower.jpg'), ms.float32)
print("x shape")
print(x.shape)
net = predict.StylePredictionNetwork()
x, y = net(x)
# for i in range(0,len(x)):
#     print(x[i].shape)
# for i in range(0,len(y)):
#     print(y[i].shape)
