import os
import cv2
import numpy as np
from mindspore import dataset as ds, Tensor
from parameters import Parameter


class _DataSetGenerator:
    def __init__(self, dataset_path):
        filenames = os.listdir(dataset_path + '/style')
        file_nums = len(filenames)
        images_np = np.ones(2, file_nums, Parameter.DataParams.height, Parameter.DataParams.width, 3)
        for i in range(0, file_nums):
            img = cv2.imread(dataset_path + '/style/' + filenames[i])
            images_np[0, i, :, :, :] = img
        filenames = os.listdir(dataset_path + '/content')
        file_nums = len(filenames)
        for i in range(0, file_nums):
            img = cv2.imread(dataset_path + '/content/' + filenames[i])
            images_np[1, i, :, :, :] = img
        self._images_tensor = Tensor(images_np)

    def __getitem__(self, index):
        return
        pass

    def __len__(self):
        pass


def get_dataset(dataset_path):
    dataset_generator = _DataSetGenerator(dataset_path)
    return ds.GeneratorDataset(dataset_generator)
