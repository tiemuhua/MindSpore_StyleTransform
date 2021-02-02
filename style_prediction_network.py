import inception_v3
from mindspore import nn, ops
style_vector_scope_names = [
    'residual/residual1/conv1',
    'residual/residual1/conv2',
    'residual/residual2/conv1',
    'residual/residual2/conv2',
    'residual/residual3/conv1',
    'residual/residual3/conv2',
    'residual/residual4/conv1',
    'residual/residual4/conv2',
    'residual/residual5/conv1',
    'residual/residual5/conv2',
    'expand/conv1/conv',
    'expand/conv2/conv',
    'expand/conv3/conv']
style_vector_depths = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3]

class StylePredictionNetwork(nn.Cell):
    def __init__(self, bottle_neck_depth=100):
        super(StylePredictionNetwork, self).__init__()
        self._inception_v3 = inception_v3.InceptionV3()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.bottle_neck = nn.Conv2d(in_channels=768, out_channels=bottle_neck_depth,kernel_size=1,has_bias=True)
        self.beta1 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.gamma1 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.beta2 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.gamma2 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.beta3 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.gamma3 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.beta4 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.gamma4 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.beta5 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.gamma5 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.beta6 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.gamma6 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.beta7 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.gamma7 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.beta8 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.gamma8 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.beta9 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.gamma9 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.beta10 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.gamma10 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=128,kernel_size=1,has_bias=True)
        self.beta11 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=64,kernel_size=1,has_bias=True)
        self.gamma11 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=64,kernel_size=1,has_bias=True)
        self.beta12 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=32,kernel_size=1,has_bias=True)
        self.gamma12 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=32,kernel_size=1,has_bias=True) 
        self.beta13 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=3,kernel_size=1,has_bias=True)
        self.gamma13 = nn.Conv2d(in_channels=bottle_neck_depth, out_channels=3,kernel_size=1,has_bias=True) 
    def construct(self, x):
        x = self._inception_v3(x)
        x = self.reduce_mean(x,(2,3))
        x = self.bottle_neck(x)
        beta1=self.beta1(x)
        gamma1=self.gamma1(x)
        beta2=self.beta2(x)
        gamma2=self.gamma2(x)
        beta3=self.beta3(x)
        gamma3=self.gamma3(x)
        beta4=self.beta4(x)
        gamma4=self.gamma4(x)
        beta5=self.beta5(x)
        gamma5=self.gamma5(x)
        beta6=self.beta6(x)
        gamma6=self.gamma6(x)
        beta7=self.beta7(x)
        gamma7=self.gamma7(x)
        beta8=self.beta8(x)
        gamma8=self.gamma8(x)
        beta9=self.beta9(x)
        gamma9=self.gamma9(x)
        beta10=self.beta10(x)
        gamma10=self.gamma10(x)
        beta11=self.beta11(x)
        gamma11=self.gamma11(x)
        beta12=self.beta12(x)
        gamma12=self.gamma12(x)
        beta13=self.beta13(x)
        gamma13=self.gamma13(x)
        return beta1,beta2,beta3,beta4,beta5,beta6,beta7,beta8,beta9,beta10,beta11,beta12,beta13,gamma1,gamma2,gamma3,gamma4,gamma5,gamma6,gamma7,gamma8,gamma9,gamma10,gamma11,gamma12,gamma13


