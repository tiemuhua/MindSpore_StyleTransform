import mindspore as ms
from mindspore import dataset as ds
from mindspore import Tensor, nn, ops
import os
import numpy as np


class _DataSetGenerator:
    def __init__(self, dataset_path):
        self._content_img_np_NHWC = np.random.rand(2, 3, 4)

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def get_dataset(dataset_path):
    dataset_generator = _DataSetGenerator(dataset_path)
    return ds.GeneratorDataset(dataset_generator)
