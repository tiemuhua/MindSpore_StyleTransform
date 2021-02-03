import mindspore as ms
import numpy as np
from mindspore import dataset as ds, nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor


class IterDatasetGenerator:
    def __init__(self):
        np.random.seed(58)
        self.__index = 0
        self.__data = np.random.sample((4, 4, 4, 4))
        self.__label = np.random.sample((4, 4, 4, 4))

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


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self._conv = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1)

    def construct(self, x):
        print(type(x))
        return self._conv(x)


class Loss(nn.loss.loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def construct(self, base, target):
        print("base")
        print(base)
        return base ** target


dataset_generator = IterDatasetGenerator()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
for data in dataset.create_dict_iterator():
    print(data["data"])
    print(data["label"])
net = Net()
loss = nn.loss.MSELoss()
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# save the network model and parameters for subsequence fine-tuning
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
# group layers into an object with training and evaluation features
model = ms.Model(net, loss, net_opt, metrics={"Accuracy": nn.Accuracy()})
model.train(1, dataset, callbacks=[ckpoint, LossMonitor()], dataset_sink_mode=False)
