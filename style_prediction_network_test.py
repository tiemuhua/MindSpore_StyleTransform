from mindspore import Tensor
import mindspore
import file_operations as f
import numpy as np
from mindspore import context
import style_prediction_network as predict
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
x=Tensor(f.load_np_image('eiffel_tower.jpg'),mindspore.float32)
net=predict.StylePredictionNetwork()
x,y=net(x)
print(x)
print(y)
