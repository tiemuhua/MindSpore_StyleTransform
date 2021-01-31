import numpy as np
import mindspore.dataset as ds
import mindspore as ms
from mindspore import nn, ops

a=ms.Tensor(np.random.rand(2,2))
b=ms.Tensor(np.random.rand(2,1))
print(a)
print(b)
print(a/b)
# print(a**b)
# print(a*b)
# print(ops.tensor_mul(a,b))