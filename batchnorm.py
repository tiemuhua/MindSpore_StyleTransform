import mindspore 
from mindspore import nn
from mindspore import Tensor
import numpy as np
import mindspore.ops as P
x=Tensor(np.ones([1,3,3,5]),mindspore.float32)
print(x)
net=nn.Moments(axis=(1,2),keep_dims=True)
mean,variance=net(x)
print(mean)
print(variance)
print("??")
print(mean.shape)
print("??")
class BatchNorm(nn.Cell):
    def __init__(self,eps):
        super(BatchNorm,self).__init__()
        self.moments=nn.Moments(axis=(2,3),keep_dims=True)
        self.sqrt=P.Sqrt()
        self.eps=eps
    def construct(self,x,beta,gamma):
        mean,variance=self.moments(x)
        temp=(x-mean)/self.sqrt(variance+self.eps)
        return temp*beta+gamma

x=Tensor(np.ones([1,128,14,14]),mindspore.float32)
beta=Tensor(np.ones([1,128,1,1]),mindspore.float32)
gamma=Tensor(np.ones([1,128,1,1]),mindspore.float32)
net=BatchNorm(0.01)
z=net(x,beta,gamma)
print(z)
print(z.shape)