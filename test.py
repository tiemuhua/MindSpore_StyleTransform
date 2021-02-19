from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context
context.set_context(mode=context.GRAPH_MODE)
param_dict = load_checkpoint("/root/xl/lab/arrange/dtd_slow-1_1000.ckpt")
print(param_dict)
print("??")
print(param_dict['prediction._inception_v3.Conv2d_1a.conv.weight'])

import network
net=network.test_network()
print(net.trainable_params())
load_param_into_net(net, param_dict)

import file_operations as f
from mindspore import Tensor
import mindspore
content=f.load_np_image('colva_beach_sq.jpg')
style=f.load_np_image("/root/xl/lab/arrange/cobwebbed_0157.jpg")
import numpy as np
#data=Tensor(np.append([content],[style],axis=0),mindspore.float32)
image=net(Tensor(content),Tensor(style))
f.Tensor_to_image(image,'dtd_1000steps')
import sys, os
sys.path.append('./vgg')
import loss
lossfunc=loss.total_loss()
'''
total_loss=lossfunc(image,data)
print("loss:")
print(total_loss)
'''