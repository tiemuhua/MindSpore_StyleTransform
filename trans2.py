import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
import numpy as np
import batchnorm as BN
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, )
class contract_conv(nn.Cell):
    def __init__(self,in_channels,out_channels,kernel_size, stride,activation_fn=nn.ReLU()):
        super(contract_conv, self).__init__()
        if(kernel_size%2==0):
            raise ValueError('kernel_size is expected to be odd.')
        padding = kernel_size // 2
        self.pad = nn.Pad(paddings=((0,0),(0,0),(padding,padding),(padding,padding)), mode="REFLECT")
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same',has_bias=True)
        self.bn=nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.activation_fn=activation_fn
    def construct(self,x):
        x=self.pad(x)
        x=self.conv1(x)
        x=self.bn(x)
        x=self.activation_fn(x)
        return x
class contract(nn.Cell):
    def __init__(self):
        super(contract, self).__init__()
        self.conv1=contract_conv(in_channels=3,out_channels=32,kernel_size=9,stride=1)
        self.conv2=contract_conv(in_channels=32,out_channels=64,kernel_size=3,stride=2)
        self.conv3=contract_conv(in_channels=64,out_channels=128,kernel_size=3,stride=2)
    def construct(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        return x
class residual_conv1(nn.Cell): 
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,activation_fn=nn.ReLU()):
        super(residual_conv1, self).__init__()
        if(kernel_size%2==0):
            raise ValueError('kernel_size is expected to be odd.')
        padding = kernel_size // 2
        self.pad = nn.Pad(paddings=((0,0),(0,0),(padding,padding),(padding,padding)), mode="REFLECT")
        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, pad_mode='valid')
        self.bn=BN.BatchNorm(eps=0.001)
        self.activation_fn=activation_fn
    def construct(self,x,beta,gamma):
        x=self.pad(x)
        x=self.conv1(x)
        x=self.bn(x,beta,gamma)
        x=self.activation_fn(x)
        return x
class residual_conv2(nn.Cell): 
    def __init__(self,in_channels,out_channels,kernel_size=3):
        super(residual_conv2, self).__init__()
        if(kernel_size%2==0):
            raise ValueError('kernel_size is expected to be odd.')
        padding = kernel_size // 2
        self.pad = nn.Pad(paddings=((0,0),(0,0),(padding,padding),(padding,padding)), mode="REFLECT")
        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, pad_mode='valid')
        self.bn=BN.BatchNorm(eps=0.001)
    def construct(self,x,beta,gamma):
        x=self.pad(x)
        x=self.conv1(x)
        x=self.bn(x,beta,gamma)
        return x
class residual_block(nn.Cell):
    def __init__(self,in_channels,out_channels):
        super(residual_block, self).__init__()
        self.conv1=residual_conv1(in_channels,out_channels)
        self.conv2=residual_conv2(in_channels,out_channels)
    def construct(self,x,beta1,gamma1,beta2,gamma2):
        temp=self.conv1(x,beta1,gamma1)
        temp=self.conv2(x,beta2,gamma2)
        return x+temp

class residual(nn.Cell):
    def __init__(self,in_channels,out_channels):
        super(residual,self).__init__()
        self.block1=residual_block(in_channels,out_channels)
        self.block2=residual_block(in_channels,out_channels)
        self.block3=residual_block(in_channels,out_channels)
        self.block4=residual_block(in_channels,out_channels)
        self.block5=residual_block(in_channels,out_channels)
    def construct(self,x,gamma1,gamma2,gamma3,gamma4,gamma5,gamma6,gamma7,gamma8,gamma9,gamma10,
                         beta1,beta2,beta3,beta4,beta5,beta6,beta7,beta8,beta9,beta10):
        x=self.block1(x,gamma1,beta1,gamma2,beta2)
        x=self.block2(x,gamma3,beta3,gamma4,beta4)
        x=self.block3(x,gamma5,beta5,gamma6,beta6)
        x=self.block4(x,gamma7,beta7,gamma8,beta8)
        x=self.block5(x,gamma9,beta9,gamma10,beta10)
        return x

import mindspore.ops.composite.array_ops as C
class expand_block(nn.Cell):
    def __init__(self,in_channels,out_channels,kernel_size,stride,activation_fn=nn.ReLU()):
        super(expand_block, self).__init__()
        self.conv1=residual_conv1(in_channels,out_channels,kernel_size=kernel_size,stride=stride,activation_fn=activation_fn)
        self.stride=stride
    def construct(self,x,beta,gamma):
        x=C.repeat_elements(x, rep = self.stride, axis = 2)
        x=C.repeat_elements(x, rep = self.stride, axis = 3)
        x=self.conv1(x,beta,gamma)
        return x

class expand(nn.Cell):
    def __init__(self):
        super(expand, self).__init__() 
        self.block1=expand_block(128,64,3,2)
        self.block2=expand_block(64,32,3,2)
        self.block3=expand_block(32,3,9,1,activation_fn=nn.Sigmoid())
    def construct(self,x,gamma1,gamma2,gamma3,beta1,beta2,beta3):
        x=self.block1(x,gamma1,beta1)
        x=self.block2(x,gamma2,beta2)
        x=self.block3(x,gamma3,beta3)
        return x 
          
x=Tensor(np.ones([1,128,14,14]),mindspore.float32)
net6=expand_block(128,64,9,1)
beta=Tensor(np.ones([1,64,1,1]),mindspore.float32)
print(x.shape)

print(net6(x,beta,beta))
print("finish1")

beta64=Tensor(np.ones([1,64,1,1]),mindspore.float32)
beta32=Tensor(np.ones([1,32,1,1]),mindspore.float32)
beta3=Tensor(np.ones([1,3,1,1]),mindspore.float32)
net7=expand()
y2=net7(x,beta64,beta32,beta3,beta64,beta32,beta3)
print(y2)


x=Tensor(np.ones([1,3,256,256]),mindspore.float32)
 
net2=contract()
y2=net2(x)
print(y2)

print("finish")
x2=Tensor(np.ones([1,128,14,14]),mindspore.float32)
net3=residual_block(128,128)
beta=Tensor(np.ones([1,128,1,1]),mindspore.float32) 
y3=net3(x2,beta,beta,beta,beta)
print(y3)

net4=residual(128,128)
y4=net4(x2,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta)
print(y4)

y5=net2(x)
y5=net4(y5,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta,beta)
print(y5)
