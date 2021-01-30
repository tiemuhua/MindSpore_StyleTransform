import numpy as np
import mindspore.dataset as ds
import mindspore as ms
from mindspore import nn

a=ms.Tensor(np.random.rand(2,2))
b=ms.Tensor(np.random.rand(2,2))
print(a)
print(b)
print(a**b)
print(a*b)