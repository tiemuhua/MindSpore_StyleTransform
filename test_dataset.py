import mindspore as ms
import numpy as np
from mindspore import nn, context
from mindspore import dataset as ds

a=np.random.rand(2,3,4)
len,_,_=a.shape
print(len)


class GetDatasetGenerator:
    def __init__(self):
        size1 = size2 = 2
        np0 = np.random.rand(size1, size2, 2)
        np1 = np.random.rand(size1, size2)
        np2 = np.random.rand(size1, size2)
        np0[:, :, 0] = np1
        np0[:, :, 1] = np2
        self.__data = np.random.rand(size1, size2)
        self.__label = np0

    def __getitem__(self, index):
        return self.__data[index], self.__label[index]

    def __len__(self):
        len,_=self.__data.shape
        return len


dataset_generator = GetDatasetGenerator()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)

for data in dataset.create_dict_iterator():
    print("data")
    print(data["data"])
    print("label")
    print(data["label"])


class Loss(nn.loss.loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()
        self._loss = nn.MSELoss()

    def construct(self, base, target):
        print("base")
        print(base)
        print("style")
        print(target["style"])
        print("content")
        print(target["content"])
        return self._loss(base, target["style"]) + self._loss(base, target["content"])


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x):
        return x

# net = Net()
# loss = Loss()
# lr = 0.05
# momentum = 0.9
# net_opt = nn.Momentum(net.trainable_params(), lr, momentum)
# model = ms.Model(net, loss, net_opt, metrics={"Accuracy": nn.Accuracy()})
#
# model.train(1, dataset)
