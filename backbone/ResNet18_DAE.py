# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu

from backbone import MammothBackbone, _DynamicModel
from backbone.utils.dae_layers import DynamicLinear, DynamicConv2D, DynamicClassifier, _DynamicLayer

def conv3x3(in_planes: int, out_planes: int, stride: int=1, norm_type=None, args=None) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return DynamicConv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, norm_type=norm_type, args=args)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1, norm_type=None, args=None) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride,norm_type=norm_type, args=args)
        self.conv2 = conv3x3(planes, planes, norm_type=norm_type, args=args)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = DynamicConv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, norm_type=norm_type, args=args)

    def forward(self, x: torch.Tensor, t, mode='ets') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.conv1(x, t, mode))
        out = self.conv2(out, t, mode)
        if self.shortcut is not None:
            out += self.shortcut(x, t, mode)
        out = relu(out)
        return out


class ResNet(_DynamicModel):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, norm_type, args) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1, norm_type=norm_type, args=args)
        self.layers = self._make_layer(block, nf * 1, num_blocks[0], stride=1, norm_type=norm_type, args=args)
        self.layers += self._make_layer(block, nf * 2, num_blocks[1], stride=2, norm_type=norm_type, args=args)
        self.layers += self._make_layer(block, nf * 4, num_blocks[2], stride=2, norm_type=norm_type, args=args)
        self.layers += self._make_layer(block, nf * 8, num_blocks[3], stride=2, norm_type=norm_type, args=args)
        self.linear = DynamicClassifier(nf * 8 * block.expansion, num_classes, norm_type=norm_type, args=args, s=1)
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        
    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int, norm_type, args) -> nn.Module:
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
            layers.append(block(self.in_planes, planes, stride, norm_type, args))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, t, mode) -> torch.Tensor:

        out = relu(self.conv1(x, t, mode))
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        
        for layer in self.layers:
            out = layer(out, t, mode)  

        out = avg_pool2d(out, out.shape[2])
        feature = out.view(out.size(0), -1)

        out = self.linear(feature, t, mode)
        return out
    
    def expand(self, new_classes, task):
        if task == 0:
            self.DM[0].expand(add_in=None, add_out=None)
        else:
            self.DM[0].expand(add_in=0, add_out=None)

        for m in self.DM[1:-1]:
            m.expand(add_in=None, add_out=None)
        self.DM[-1].expand(add_in=None, add_out=new_classes)

    def squeeze(self, optim_state):
        mask_in = None
        mask_out = self.conv1.mask_out
        self.conv1.squeeze(optim_state, mask_in, mask_out)
        mask_in = mask_out

        for block in self.layers:
            mask_out = block.conv1.mask_out
            block.conv1.squeeze(optim_state, mask_in, mask_out)
            shared_mask = block.shortcut.mask_out + block.conv2.mask_out
            block.conv2.squeeze(optim_state, mask_out, shared_mask)
            block.shortcut.squeeze(optim_state, mask_in, shared_mask)
            mask_in = shared_mask
        
        self.linear.squeeze(optim_state, mask_in, None)


def resnet18(nclasses: int, nf: int=64, norm_type='bn', args=None) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, norm_type, args)
