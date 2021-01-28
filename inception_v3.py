# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Inception-v3 model definition"""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import XavierUniform

'''
This class 
Note that this is not the entire InceptionV3 network.
Because in this project, we only need the first a few networks before "Mixed_6e", 
    so we did not implement other layers.
'''


class InceptionV3(nn.Cell):
    def __init__(self, num_classes=10, is_training=True, has_bias=False, dropout_keep_prob=0.8, include_top=True):
        super(InceptionV3, self).__init__()
        self.is_training = is_training
        self.Conv2d_1a = _BasicConv2d(3, 32, kernel_size=3, stride=2, pad_mode='valid', has_bias=has_bias)
        self.Conv2d_2a = _BasicConv2d(32, 32, kernel_size=3, stride=1, pad_mode='valid', has_bias=has_bias)
        self.Conv2d_2b = _BasicConv2d(32, 64, kernel_size=3, stride=1, has_bias=has_bias)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b = _BasicConv2d(64, 80, kernel_size=1, has_bias=has_bias)
        self.Conv2d_4a = _BasicConv2d(80, 192, kernel_size=3, pad_mode='valid', has_bias=has_bias)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = _InceptionA(192, pool_features=32, has_bias=has_bias)
        self.Mixed_5c = _InceptionA(256, pool_features=64, has_bias=has_bias)
        self.Mixed_5d = _InceptionA(288, pool_features=64, has_bias=has_bias)
        self.Mixed_6a = _InceptionB(288, has_bias=has_bias)
        self.Mixed_6b = _InceptionC(768, channels_7x7=128, has_bias=has_bias)
        self.Mixed_6c = _InceptionC(768, channels_7x7=160, has_bias=has_bias)
        self.Mixed_6d = _InceptionC(768, channels_7x7=160, has_bias=has_bias)
        self.Mixed_6e = _InceptionC(768, channels_7x7=192, has_bias=has_bias)

    def construct(self, x):
        x = self.Conv2d_1a(x)
        x = self.Conv2d_2a(x)
        x = self.Conv2d_2b(x)
        x = self.max_pool1(x)
        x = self.Conv2d_3b(x)
        x = self.Conv2d_4a(x)
        x = self.max_pool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        return x


class _BasicConv2d(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, pad_mode='same', padding=0, has_bias=False):
        super(_BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              pad_mode=pad_mode, padding=padding, weight_init=XavierUniform(), has_bias=has_bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _InceptionA(nn.Cell):
    def __init__(self, in_channels, pool_features, has_bias=False):
        super(_InceptionA, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = _BasicConv2d(in_channels, 64, kernel_size=1, has_bias=has_bias)
        self.branch1 = nn.SequentialCell([
            _BasicConv2d(in_channels, 48, kernel_size=1, has_bias=has_bias),
            _BasicConv2d(48, 64, kernel_size=5, has_bias=has_bias)
        ])
        self.branch2 = nn.SequentialCell([
            _BasicConv2d(in_channels, 64, kernel_size=1, has_bias=has_bias),
            _BasicConv2d(64, 96, kernel_size=3, has_bias=has_bias),
            _BasicConv2d(96, 96, kernel_size=3, has_bias=has_bias)

        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            _BasicConv2d(in_channels, pool_features, kernel_size=1, has_bias=has_bias)
        ])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, x2, branch_pool))
        return out


class _InceptionB(nn.Cell):
    def __init__(self, in_channels, has_bias=False):
        super(_InceptionB, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = _BasicConv2d(in_channels, 384, kernel_size=3, stride=2, pad_mode='valid', has_bias=has_bias)
        self.branch1 = nn.SequentialCell([
            _BasicConv2d(in_channels, 64, kernel_size=1, has_bias=has_bias),
            _BasicConv2d(64, 96, kernel_size=3, has_bias=has_bias),
            _BasicConv2d(96, 96, kernel_size=3, stride=2, pad_mode='valid', has_bias=has_bias)

        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, branch_pool))
        return out


class _InceptionC(nn.Cell):
    def __init__(self, in_channels, channels_7x7, has_bias=False):
        super(_InceptionC, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch0 = _BasicConv2d(in_channels, 192, kernel_size=1, has_bias=has_bias)
        self.branch1 = nn.SequentialCell([
            _BasicConv2d(in_channels, channels_7x7, kernel_size=1, has_bias=has_bias),
            _BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), has_bias=has_bias),
            _BasicConv2d(channels_7x7, 192, kernel_size=(7, 1), has_bias=has_bias)
        ])
        self.branch2 = nn.SequentialCell([
            _BasicConv2d(in_channels, channels_7x7, kernel_size=1, has_bias=has_bias),
            _BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), has_bias=has_bias),
            _BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), has_bias=has_bias),
            _BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), has_bias=has_bias),
            _BasicConv2d(channels_7x7, 192, kernel_size=(1, 7), has_bias=has_bias)
        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            _BasicConv2d(in_channels, 192, kernel_size=1, has_bias=has_bias)
        ])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, x2, branch_pool))
        return out


