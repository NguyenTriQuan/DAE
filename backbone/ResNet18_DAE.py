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

        # self.shortcut = None
        # if stride != 1 or in_planes != self.expansion * planes:
        self.shortcut = DynamicConv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, norm_type=norm_type, args=args)

    def forward(self, x: torch.Tensor, t, mode) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.conv1(x, t, mode))
        out = self.conv2(out, t, mode)
        if self.shortcut is not None:
            sc_out = self.shortcut(x, t, mode)
        else:
            sc_out = x
        
        if out.numel() == 0:
            out = sc_out
        else:
            out += sc_out
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
        if 'ets' in mode:
            self.get_kb_params(t)
        else:
            self.get_masked_kb_params(t)

        out = relu(self.conv1(x, t, mode))
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        
        for layer in self.layers:
            out = layer(out, t, mode)  

        out = avg_pool2d(out, out.shape[2])
        feature = out.view(out.size(0), -1)

        out = self.linear(feature, t, mode)
        return out
    
    def expand(self, new_classes, t):
        if t == 0:
            add_in = self.conv1.expand(add_in=self.conv1.base_in_features)
        else:
            add_in = self.conv1.expand(add_in=0)

        for block in self.layers:
            add_in_1 = block.conv1.expand(add_in=add_in)
            _, _, _, add_out_2 = block.conv2.get_expand_shape(-1, add_in_1)
            _, _, _, add_out_sc = block.shortcut.get_expand_shape(-1, add_in)
            add_out = min(add_out_2, add_out_sc)
            block.conv2.expand(add_in=add_in_1, add_out=add_out)
            block.shortcut.expand(add_in=add_in, add_out=add_out)
            add_in = add_out

            max_strength = max(block.conv2.strength_in, block.shortcut.strength_in)
            block.conv2.strength_in = max_strength
            block.shortcut.strength_in = max_strength

        self.linear.expand(add_in=add_in, add_out=new_classes)

        self.total_strength = 1
        for m in self.DM[:-1]:
            self.total_strength += m.strength_in

    def squeeze(self, optim_state):
        mask_in = None
        mask_out = self.conv1.mask_out
        self.conv1.squeeze(optim_state, mask_in, mask_out)
        mask_in = mask_out

        for block in self.layers:
            mask_out = block.conv1.mask_out
            block.conv1.squeeze(optim_state, mask_in, mask_out)
            shared_mask = block.conv2.mask_out + block.shortcut.mask_out
            block.shortcut.squeeze(optim_state, mask_in, shared_mask)
            block.conv2.squeeze(optim_state, mask_out, shared_mask)
            mask_in = shared_mask

            max_strength = max(block.conv2.strength_in, block.shortcut.strength_in)
            block.conv2.strength_in = max_strength
            block.shortcut.strength_in = max_strength
        
        self.linear.squeeze(optim_state, mask_in, None)
        self.total_strength = 1
        for m in self.DM[:-1]:
            self.total_strength += m.strength_in

    def get_masked_kb_params(self, t):
        if t == 0:
            add_in = self.conv1.get_masked_kb_params(t, add_in=self.conv1.base_in_features)
        else:
            add_in = self.conv1.get_masked_kb_params(t, add_in=0)

        for block in self.layers:
            add_in_1 = block.conv1.get_masked_kb_params(t, add_in=add_in)
            _, _, _, add_out_2 = block.conv2.get_expand_shape(t, add_in_1)
            _, _, _, add_out_sc = block.shortcut.get_expand_shape(t, add_in)
            add_out = min(add_out_2, add_out_sc)
            block.conv2.get_masked_kb_params(t, add_in=add_in_1, add_out=add_out)
            block.shortcut.get_masked_kb_params(t, add_in=add_in, add_out=add_out)
            add_in = add_out

    def set_jr_params(self):
        add_in = self.conv1.set_jr_params(add_in=0)
        for block in self.layers:
            add_in_1 = block.conv1.set_jr_params(add_in=add_in)
            _, _, _, add_out_2 = block.conv2.get_expand_shape(-1, add_in_1)
            _, _, _, add_out_sc = block.shortcut.get_expand_shape(-1, add_in)
            add_out = min(add_out_2, add_out_sc)
            block.conv2.set_jr_params(add_in=add_in_1, add_out=add_out)
            block.shortcut.set_jr_params(add_in=add_in, add_out=add_out)
            add_in = add_out


def resnet18(nclasses: int, nf: int=64, norm_type='bn_track_affine', args=None) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, norm_type, args)
