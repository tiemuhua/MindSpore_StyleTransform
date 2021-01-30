import mindspore as ms
import numpy as np
from mindspore import dataset as ds
from mindspore import nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig


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
        self._conv = nn.Conv2d()
        self._bn = nn.BatchNorm2d(num_features=3)
        self._rely = nn.ReLU()

    def construct(self, x):
        return self._conv(self._bn(self._conv(x)))


class IterDatasetGenerator:
    def __init__(self):
        np.random.seed(58)
        self.__index = 0
        height = 5
        weight = 2
        self.__data = np.random.rand(height, weight)
        self.__label = np.random.rand(height, weight)

    def __next__(self):
        if self.__index >= len(self.__data):
            raise StopIteration
        else:
            item = (self.__data[self.__index], self.__label[self.__index])
            self.__index += 1
            return item

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.__data)


dataset_generator = IterDatasetGenerator()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)

for data in dataset.create_dict_iterator():
    print(data["data"], data["label"])

net = Net()
loss = Loss()
lr = 0.1
momentum = 0.9
op = nn.optim.Momentum(net.trainable_params(), learning_rate=lr, momentum=momentum)
model = ms.Model(net, loss, op, metrics={"Accuracy": nn.Accuracy()})
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
model.train(1, dataset)
