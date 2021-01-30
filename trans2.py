import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
import numpy as np
import batchnorm as BN
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
class residual_conv(nn.Cell): 
    def __init__(self,in_channels,out_channels,kernel_size=3,activation_fn=nn.ReLU()):
        super(residual_conv, self).__init__()
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
class residual_block(nn.Cell):
    def __init__(self,in_channels,out_channels):
        super(residual_block, self).__init__()
        self.conv1=residual_conv(in_channels,out_channels)
        self.conv1=residual_conv(in_channels,out_channels)
    def construct(self,x,beta1,gamma1,beta2,gamma2):
        temp=self.conv1(x,beta1,gamma1)
        temp=self.conv1(x,beta2,gamma2)
        return x+temp

class residual(nn.Cell):
    def __init__(self,in_channels,out_channels):
        super(residual,self).__init__()
        self.block1=residual_block(in_channels,out_channels)
        self.block2=residual_block(in_channels,out_channels)
        self.block3=residual_block(in_channels,out_channels)
        self.block4=residual_block(in_channels,out_channels)
        self.block5=residual_block(in_channels,out_channels)
    def construct(self,x,betas,gammas):
        x=self.block1(x,betas[0],gammas[0],betas[1],gammas[1])
        x=self.block2(x,betas[2],gammas[2],betas[3],gammas[3])
        x=self.block3(x,betas[4],gammas[4],betas[5],gammas[5])
        x=self.block4(x,betas[6],gammas[6],betas[7],gammas[7])
        x=self.block5(x,betas[8],gammas[8],betas[9],gammas[9])
        return x
x=Tensor(np.ones([1,3,256,256]),mindspore.float32)
net=contract_conv(in_channels=3,out_channels=32,kernel_size=9,stride=1)
y=net(x)
print(y)
 
net2=contract()
y2=net2(x)
print(y2)


x2=Tensor(np.ones([1,128,14,14]),mindspore.float32)
net3=residual_block(128,128)
beta=Tensor(np.ones([1,128,1,1]),mindspore.float32) 
y3=net3(x2,beta,beta,beta,beta)
print(y3)

net4=residual(128,128)
betas=[beta,beta,beta,beta,beta,beta,beta,beta,beta,beta]
gammas=[beta,beta,beta,beta,beta,beta,beta,beta,beta,beta]
y4=net4(x2,betas,gammas)
print(y4)

y5=net2(x)
y5=net4(y5,betas,gammas)
print(y5)