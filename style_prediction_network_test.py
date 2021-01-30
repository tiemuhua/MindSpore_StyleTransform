import mindspore
from mindspore import Tensor
from mindspore import context

import file_operations as f
import style_prediction_network as predict

context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
x = Tensor(f.load_np_image('eiffel_tower.jpg'), mindspore.float32)
net = predict.StylePredictionNetwork()
x, y = net(x)
print(x)
print(y)
