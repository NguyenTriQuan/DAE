# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu

# from backbone import MammothBackbone, _DynamicModel
from backbone.utils.dae_layers import DynamicLinear, DynamicConv2D, DynamicClassifier, _DynamicLayer, DynamicNorm, DynamicBlock

def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)

def ensemble_outputs(outputs):
    ## a list of outputs with length [num_member], each with shape [bs, num_cls]
    outputs = torch.stack(outputs, dim=-1) #[bs, num_cls, num_member]
    outputs = F.log_softmax(outputs, dim=-2)
    ## with shape [bs, num_cls]
    log_outputs = logmeanexp(outputs, dim=-1)
    return log_outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _DynamicModel(nn.Module):
    def __init__(self):
        super(_DynamicModel, self).__init__()
        self.DB = [m for m in self.modules() if isinstance(m, DynamicBlock)]
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        self.total_strength = 1
        self.ets_temp = -1
        self.kbts_temp = -1

    def get_optim_ets_params(self):
        params = []
        for m in self.DB:
            params += m.get_optim_ets_params()
        # params += self.mid.get_optim_ets_params()
        params += self.last.get_optim_ets_params()
        params += self.projector.parameters()
        return params
    
    def get_optim_kbts_params(self):
        params = []
        scores = []
        for m in self.DB:
            p, s = m.get_optim_kbts_params()
            params += p
            scores += s
        # params += self.mid.get_optim_kbts_params()
        params += self.last.get_optim_kbts_params()
        params += self.projector.parameters()
        return params, scores

    def count_params(self, t=-1):
        if t == -1:
            t = len(self.DM[-1].num_out)-1
        num_params = []
        num_neurons = []
        for m in self.DM:
            num_params.append(m.count_params(t))
            num_neurons.append(m.shape_out[t+1])
        return num_params, num_neurons

    def freeze_feature(self):
        for m in self.DB:
            m.freeze()
        # self.mid.freeze()

    def freeze_classifier(self):
        self.last.freeze()

    def proximal_gradient_descent(self, lr=0, lamb=0):
        with torch.no_grad():
            for block in self.DB:
                block.proximal_gradient_descent(lr, lamb, self.total_strength)
    
    def clear_memory(self):
        self.ets_temp = -1
        self.kbts_temp = -1
        for m in self.DM:
            if not isinstance(m, DynamicClassifier):
                m.clear_memory()
    
    def get_kb_params(self, t):
        for m in self.DM:
            if not isinstance(m, DynamicClassifier):
                m.get_kb_params(t)

    def ERK_sparsify(self, sparsity=0.9):
        # print('initialize by ERK')
        density = 1 - sparsity
        erk_power_scale = 1

        total_params = 0
        for m in self.DM:
            if not isinstance(m, DynamicClassifier):
                total_params += m.score.numel()
        is_epsilon_valid = False

        dense_layers = set()
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            for m in self.DM:
                if not isinstance(m, DynamicClassifier):
                    m.raw_probability = 0
                    n_param = np.prod(m.score.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if m in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        m.raw_probability = (np.sum(m.score.shape) / np.prod(m.score.shape)) ** erk_power_scale
                        divisor += m.raw_probability * n_param

            epsilon = rhs / divisor
            max_prob = np.max([m.raw_probability for m in self.DM if not isinstance(m, DynamicClassifier)])
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for m in self.DM:
                    if not isinstance(m, DynamicClassifier):
                        if m.raw_probability == max_prob:
                            # print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(m)
            else:
                is_epsilon_valid = True

        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        min_sparsity = 0.5
        for i, m in enumerate(self.DM):
            if not isinstance(m, DynamicClassifier):
                n_param = np.prod(m.score.shape)
                if m in dense_layers:
                    m.sparsity = min_sparsity
                else:
                    probability_one = epsilon * m.raw_probability
                    m.sparsity = max(1 - probability_one, min_sparsity)
                # print(
                #     f"layer: {i}, shape: {m.score.shape}, sparsity: {m.sparsity}"
                # )
                total_nonzero += (1-m.sparsity) * m.score.numel()
        print(f"Overall sparsity {1-total_nonzero / total_params}")

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
        conv1 = conv3x3(in_planes, planes, stride=stride, args=args)
        conv2 = conv3x3(planes, planes, stride=1, args=args)
        shortcut = DynamicConv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, args=args)
        self.conv1 = DynamicBlock([conv1], norm_type, args)
        self.conv2 = DynamicBlock([shortcut, conv2], norm_type, args)

    def ets_forward(self, x: torch.Tensor, t) -> torch.Tensor:
        out = self.conv1.ets_forward([x], t)
        out = self.conv2.ets_forward([x, out], t)
        return out
    
    def kbts_forward(self, x: torch.Tensor, t) -> torch.Tensor:
        out = self.conv1.kbts_forward([x], t)
        out = self.conv2.kbts_forward([x, out], t)
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
        self.args = args
        conv1 = conv3x3(3, nf * 1, args=args)
        self.conv1 = DynamicBlock([conv1], norm_type, args)
        self.layers = self._make_layer(block, nf * 1, num_blocks[0], stride=1, norm_type=norm_type, args=args)
        self.layers += self._make_layer(block, nf * 2, num_blocks[1], stride=2, norm_type=norm_type, args=args)
        self.layers += self._make_layer(block, nf * 4, num_blocks[2], stride=2, norm_type=norm_type, args=args)
        self.layers += self._make_layer(block, nf * 8, num_blocks[3], stride=2, norm_type=norm_type, args=args)

        feat_dim = nf * 8 * block.expansion
        self.mid = DynamicBlock([DynamicLinear(feat_dim, feat_dim, bias=False, args=args, s=1)], None, args)
        # self.mid = DynamicClassifier(feat_dim, feat_dim, norm_type=norm_type, args=args, s=1, bias=False)
        self.last = DynamicClassifier(feat_dim, num_classes, norm_type=norm_type, args=args, s=1, bias=True)
        self.DB = [m for m in self.modules() if isinstance(m, DynamicBlock)]
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]

        self.ets_cal_layers = nn.ModuleList([])
        self.kbts_cal_layers = nn.ModuleList([])
        
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 128)
        ).to(device)
        self.feat_dim = feat_dim
        
        
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


    def ets_forward(self, x: torch.Tensor, t, feat=False, cal=False) -> torch.Tensor:
        if t != self.ets_temp:
            self.get_kb_params(t)
            self.ets_temp = t
        out = self.conv1.ets_forward([x], t)
        
        for layer in self.layers:
            out = layer.ets_forward(out, t)  

        out = F.avg_pool2d(out, out.shape[2])
        feature = out.view(out.size(0), -1)
        feature = self.mid.ets_forward([feature], t)
        # feature = self.mid.ets_forward(feature, t)
        # feature = F.relu(feature)

        if feat:
            return self.projector(feature)
        else:
            out = self.last.ets_forward(feature, t)
            if cal:
                scales = self.ets_cal_layers[t](feature)
                alpha = scales[:, 0].view(-1, 1)
                beta = scales[:, 1].view(-1, 1)
                return out * alpha + beta
            else:
                return out
    
    def kbts_forward(self, x: torch.Tensor, t, feat=False, cal=False) -> torch.Tensor:
        if t != self.kbts_temp:
            if t != self.ets_temp:
                self.get_kb_params(t)
                self.ets_temp = t
            self.get_masked_kb_params(t)
            self.kbts_temp = t
        out = self.conv1.kbts_forward([x], t)
        
        for layer in self.layers:
            out = layer.kbts_forward(out, t)  

        out = F.avg_pool2d(out, out.shape[2])
        feature = out.view(out.size(0), -1)
        feature = self.mid.kbts_forward([feature], t)
        # feature = self.mid.kbts_forward(feature, t)
        # feature = F.relu(feature)
        if feat:
            return self.projector(feature)
        else:
            out = self.last.kbts_forward(feature, t)
            if cal:
                scales = self.kbts_cal_layers[t](feature)
                alpha = scales[:, 0].view(-1, 1)
                beta = scales[:, 1].view(-1, 1)
                return out * alpha + beta
            else:
                return out
    
    def expand(self, new_classes, t):
        if t == 0:
            add_in = self.conv1.expand([(None, None)], [(None, None)])
        else:
            add_in = self.conv1.expand([(0, 0)], [(None, None)])

        for block in self.layers:
            add_in_1 = block.conv1.expand([add_in], [(None, None)])
            add_in = block.conv2.expand([add_in, add_in_1], [(None, None), (None, None)])

        # do not expand and squeeze the last layer (fix number of features)
        # block = self.layers[-1]
        # if t == 0:
        #     add_in_1 = block.conv1.expand([add_in], [(None, None)])
        #     add_in = block.conv2.expand([add_in, add_in_1], [(None, None), (None, None)])
        # else:
        #     add_in_1 = block.conv1.expand([add_in], [(None, None)])
        #     add_in = block.conv2.expand([add_in, add_in_1], [(0, 0), (0, 0)])

        # if t == 0:
        #     add_in = self.mid.expand([add_in], [(None, None)])
        # else:
        #     add_in = self.mid.expand([add_in], [(0, 0)])
        # self.mid.expand(add_in, (self.feat_dim, self.feat_dim))
        # if t == 0:
        #     self.last.expand((self.feat_dim, self.feat_dim), (new_classes, new_classes))
        # else:
        #     self.last.expand((0, 0), (new_classes, new_classes))

        add_in = self.mid.expand([add_in], [(None, None)])
        self.last.expand(add_in, (new_classes, new_classes))
        self.total_strength = 1
        for m in self.DB:
            self.total_strength += m.strength


    def squeeze(self, optim_state):
        mask_in = None
        self.conv1.squeeze(optim_state, [mask_in])
        mask_in = self.conv1.mask_out

        for block in self.layers:
            block.conv1.squeeze(optim_state, [mask_in])
            block.conv2.squeeze(optim_state, [mask_in, block.conv1.mask_out])
            mask_in = block.conv2.mask_out
        
        # self.mid.squeeze(optim_state, [mask_in])
        # self.mid.squeeze(optim_state, mask_in, None)
        # self.linear.squeeze(optim_state, mask_in, None)
        self.mid.squeeze(optim_state, [mask_in])
        self.last.squeeze(optim_state, self.mid.mask_out, None)

        self.total_strength = 1
        for m in self.DB:
            self.total_strength += m.strength

    def get_masked_kb_params(self, t):
        if t == 0:
            add_in = self.conv1.get_masked_kb_params(t, [None], [None])
        else:
            add_in = self.conv1.get_masked_kb_params(t, [0], [None])

        for block in self.layers:
            add_in_1 = block.conv1.get_masked_kb_params(t, [add_in], [None])
            add_in = block.conv2.get_masked_kb_params(t, [add_in, add_in_1], [None, None])

        self.mid.get_masked_kb_params(t, [add_in], [None])
        # do not expand and squeeze the last layer (fix number of features)
        # block = self.layers[-1]
        # if t == 0:
        #     add_in_1 = block.conv1.get_masked_kb_params(t, [add_in], [None])
        #     add_in = block.conv2.get_masked_kb_params(t, [add_in, add_in_1], [None, None])
        # else:
        #     add_in_1 = block.conv1.get_masked_kb_params(t, [add_in], [None])
        #     add_in = block.conv2.get_masked_kb_params(t, [add_in, add_in_1], [0, 0])

        # if t == 0:
        #     self.mid.get_masked_kb_params(t, [add_in], [None])
        # else:
        #     self.mid.get_masked_kb_params(t, [add_in], [0])

    def set_cal_params(self, num_tasks):
        hidden_dim = 128

        self.ets_cal_layers = nn.ModuleList([])
        self.kbts_cal_layers = nn.ModuleList([])

        for i in range(num_tasks):
            # ets_dim = self.linear.weight_ets[i].shape[1]
            # kbts_dim = self.linear.weight_kbts[i].shape[1]
            self.ets_cal_layers.append(
                nn.Sequential(
                    nn.Linear(self.feat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    # nn.Linear(hidden_dim, hidden_dim),
                    # nn.ReLU(),
                    # nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 2),
                    nn.Sigmoid()
                ).to(device)
            )
            self.kbts_cal_layers.append(
                nn.Sequential(
                    nn.Linear(self.feat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    # nn.Linear(hidden_dim, hidden_dim),
                    # nn.ReLU(),
                    # nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 2),
                    nn.Sigmoid()
                ).to(device)
            )

    def reset_cal_params(self, num_tasks):
        hidden_dim = 128
        feat_dim = 128

        self.ets_cal_layers = nn.ModuleList([])

        for i in range(len(self.last.weight_ets)):
            ets_dim = self.last.weight_ets[i].shape[1]
            kbts_dim = self.last.weight_kbts[i].shape[1]
            self.ets_cal_layers.append(
                nn.Sequential(
                    nn.Linear(ets_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 2),
                    nn.Sigmoid()
                ).to(device)
            )
        

        
    def get_optim_cal_params(self, tc=False):
        return list(self.kbts_cal_layers.parameters()) + list(self.ets_cal_layers.parameters())
        # if not tc:
        #     return list(self.cal_head.parameters()) + list(self.task_feature_layers.parameters()) + list(self.projector.parameters()) + list(self.ets_cal_layers.parameters()) + list(self.kbts_cal_layers.parameters())
        # else:
        #     return list(self.cal_head.parameters())
        # return list(self.ets_cal_head.parameters()) + list(self.ets_cal_layers.parameters()) \
        #             + list(self.kbts_cal_head.parameters()) + list(self.kbts_cal_layers.parameters())
        
    
    def get_optim_tc_params(self):
        return list(self.task_feature_layers.parameters()) + list(self.projector.parameters()) + list(self.ets_cal_layers.parameters()) + list(self.kbts_cal_layers.parameters())
        # return list(self.task_feature_layers.parameters()) 


def resnet18(nclasses: int, nf: int=64, norm_type='bn_track_affine', args=None) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, 64, norm_type, args)

def resnet10(nclasses: int, nf: int=64, norm_type='bn_track_affine', args=None) -> ResNet:
    return ResNet(BasicBlock, [1, 1, 1, 1], nclasses, 32, norm_type, args)
