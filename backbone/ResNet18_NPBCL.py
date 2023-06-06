# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import avg_pool2d, relu

from backbone import MammothBackbone
from backbone.utils.npbcl_layers import MaksedLinear, MaskedConv2d, EsmBatchNorm2d, EsmLinear


def conv3x3(in_planes: int, out_planes: int, stride: int=1, args=None):
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return MaskedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, args=args)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1, args=None) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, args=args)
        self.bn1 = EsmBatchNorm2d(planes, args=args)
        self.conv2 = conv3x3(planes, planes, args=args)
        self.bn2 = EsmBatchNorm2d(planes, args=args)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = MaskedConv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False, args=args)
            self.shortcut_bn = EsmBatchNorm2d(self.expansion * planes, args=args)

    def forward(self, x: torch.Tensor, t) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x, t), t))
        out = self.bn2(self.conv2(out, t), t)
        if self.shortcut is not None:
            out += self.shortcut_bn(self.shortcut(x, t), t)
        else:
            out += x
        out = relu(out)
        return out


class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, args=None) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.args=args
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1, args=args)
        self.bn1 = EsmBatchNorm2d(nf * 1, args=args)
        self.layers = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layers += self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layers += self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layers += self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = EsmLinear(nf * 8 * block.expansion, num_classes, args=args)
        self.DM = [m for m in self.modules() if hasattr(m, 'stable_score')]

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, args=self.args))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, t) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :return: output tensor (output_classes)
        """

        out = relu(self.bn1(self.conv1(x, t), t)) # 64, 32, 32
        for layer in self.layers:
            out = layer(out, t)
        out = avg_pool2d(out, out.shape[2]) # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512

        out = self.linear(feature, t)
        return out

    def set_mode(self, mode):
        for m in self.modules():
            if hasattr(m, 'mode'):
                m.mode = mode

    def update_unused_weights(self, t):
        used_params = 0
        for m in self.DM:
            m.update_unused_weights(t)
            used_params = m.unused_weight.numel() - m.unused_weight.sum()
        print(f'Used params: {int(used_params)}')

    def freeze_used_weights(self):
        for m in self.DM:
            m.freeze_used_weights()

    def ERK_sparsify(self, sparsity=0.9):
        # print('initialize by ERK')
        density = 1 - sparsity
        erk_power_scale = 1

        total_params = 0
        for m in self.DM:
            total_params += m.weight.numel()
        is_epsilon_valid = False

        dense_layers = set()
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            for m in self.DM:
                m.raw_probability = 0
                n_param = np.prod(m.weight.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if m in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    m.raw_probability = (np.sum(m.weight.shape) / np.prod(m.weight.shape)) ** erk_power_scale
                    divisor += m.raw_probability * n_param

            epsilon = rhs / divisor
            max_prob = np.max([m.raw_probability for m in self.DM])
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for m in self.DM:
                    if m.raw_probability == max_prob:
                        # print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(m)
            else:
                is_epsilon_valid = True

        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        min_sparsity = 0.5
        for i, m in enumerate(self.DM):
            n_param = np.prod(m.weight.shape)
            if m in dense_layers:
                m.sparsity = min_sparsity
            else:
                probability_one = epsilon * m.raw_probability
                m.sparsity = max(1 - probability_one, min_sparsity)
            print(
                f"layer: {i}, shape: {m.weight.shape}, sparsity: {m.sparsity}"
            )
            total_nonzero += (1-m.sparsity) * m.weight.numel()
        print(f"Overall sparsity {1-total_nonzero / total_params}, Total params {total_params}")


def resnet18(nclasses: int, nf: int=64, args=None) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, args)

def resnet10(nclasses: int, nf: int=32, args=None) -> ResNet:
    return ResNet(BasicBlock, [1, 1, 1, 1], nclasses, nf, args)   
