
from mindspore import nn
import numpy as np
import time
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
import mindspore as ms
import mindspore.ops as ops
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, )
class BatchNorm(nn.Cell):
    def __init__(self,eps):
        super(BatchNorm, self).__init__()
        self.squeeze = ops.Squeeze()
        self.moments=nn.Moments(axis=(0,2,3))
        self.bn_train = P.BatchNorm(is_training=True,epsilon=eps,)
    def construct(self, x, gamma, beta):
        mean,variance=self.moments(x)   
        gamma= self.squeeze(gamma)
        beta=self.squeeze(beta)
        output=self.bn_train(x,
                             gamma,
                             beta,
                             mean,
                             variance)[0]
        return output

net = BatchNorm(0.01)
input = Tensor(np.ones([1,3,256,256]), ms.float32)
gamma = Tensor(np.ones([1,3,1,1]), ms.float32)
beta = Tensor(np.ones([1,3,1,1]), ms.float32)
out = net(input, gamma, beta)
print("#####################")
print(out)