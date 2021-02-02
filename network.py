import trans2 as T
import style_prediction_network as S
import mindspore.nn as nn
import file_operations as f
from mindspore import Tensor
import mindspore
import numpy as np
import matplotlib.pyplot as plt
class Transform(nn.Cell):
    def __init__(self):
        super(Transform,self).__init__()
        self.contract=T.contract()
        self.residual=T.residual(128,128)
        self.expand=T.expand()
    def construct(self,x,gamma1,gamma2,gamma3,gamma4,
                         gamma5,gamma6,gamma7,gamma8,
                         gamma9,gamma10,gamma11,gamma12,gamma13,
                         beta1,beta2,beta3,beta4,
                         beta5,beta6,beta7,beta8,
                         beta9,beta10,beta11,beta12,beta13):
        x=self.contract(x)
        x=self.residual(x,gamma1,gamma2,gamma3,gamma4,gamma5,gamma6,gamma7,gamma8,gamma9,gamma10,
                          beta1,beta2,beta3,beta4,beta5,beta6,beta7,beta8,beta9,beta10)
        x=self.expand(x,gamma11,gamma12,gamma13,beta11,beta12,beta13)
        return x
class network(nn.Cell):
    def __init__(self):
        super(network,self).__init__()
        self.prediction=S.StylePredictionNetwork()
        self.transform=Transform()
    def construct(self,content,style):
        gamma1,gamma2,gamma3,gamma4,\
        gamma5,gamma6,gamma7,gamma8,\
        gamma9,gamma10,gamma11,gamma12,gamma13,\
        beta1,beta2,beta3,beta4,\
        beta5,beta6,beta7,beta8,\
        beta9,beta10,beta11,beta12,beta13=self.prediction(style)
        return self.transform(content,gamma1,gamma2,gamma3,gamma4,
                                      gamma5,gamma6,gamma7,gamma8,
                                      gamma9,gamma10,gamma11,gamma12,gamma13,
                                      beta1,beta2,beta3,beta4,
                                      beta5,beta6,beta7,beta8,
                                      beta9,beta10,beta11,beta12,beta13)
x=Tensor(f.load_np_image('eiffel_tower.jpg'),mindspore.float32)
y=Tensor(f.load_np_image('s2.jpg'),mindspore.float32)
net=network()
z=net(x,y)
print("output")
print(z)
z=z.asnumpy()
print(z.shape)
z=np.swapaxes(z,1,2)
print(z.shape)
z=np.swapaxes(z,2,3)
print(z.shape)
z=z.squeeze()
print(z.shape)

plt.imshow(z)
plt.savefig('a5.jpg')
print("output")
print(z)
