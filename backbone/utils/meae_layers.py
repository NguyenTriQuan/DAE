import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor, isin, seed
from typing import Optional, Any
from torch.nn import init
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from utils import *
from typing import Optional, List, Tuple, Union
import sys

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5,6,7"
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def apply_mask_out(param, mask_out, optim_state):
    param.data = param.data[mask_out].clone()
    param.grad = None
    param_states = optim_state[param]
    for name, state in param_states.items():
        if isinstance(state, torch.Tensor):
            if len(state.shape) > 0:
                param_states[name] = state[mask_out].clone()

def apply_mask_in(param, mask_in, optim_state):
    param.data = param.data[:, mask_in].clone()
    param.grad = None
    param_states = optim_state[param]
    for name, state in param_states.items():
        if isinstance(state, torch.Tensor):
            if len(state.shape) > 0:
                param_states[name] = state[:, mask_in].clone()

class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class _DynamicLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=False, norm_type=None, args=None, s=1):
        super(_DynamicLayer, self).__init__()
        self.args = args
        self.device = args.device
        self.dropout = args.dropout
        self.sparsity = args.sparsity
        self.s = s
        self.base_in_features = in_features
        self.base_out_features = out_features
        self.base_params = self.base_out_features * self.base_in_features * self.ks

        self.mask_in = None
        self.mask_out = None

        self.weight = nn.ParameterList([])
        self.fwt_weight = nn.ParameterList([])
        self.bwt_weight = nn.ParameterList([])

        self.register_buffer('bias', None)

        self.register_buffer('shape_out', torch.IntTensor([0]).to(self.device))
        self.register_buffer('shape_in', torch.IntTensor([0]).to(self.device))
        self.register_buffer('num_out', torch.IntTensor([]).to(self.device))
        self.register_buffer('num_in', torch.IntTensor([]).to(self.device))
        self.register_buffer('task', torch.tensor(-1, dtype=torch.int).to(self.device))
        # self.register_buffer('kbts_sparsities', torch.IntTensor([]).to(device))

        # self.shape_out = [0]
        # self.shape_in = [0]
        # self.num_out = []
        # self.num_in = []
        self.kbts_sparsities = []
        self.bound_std = []
        self.total_masked_kb = []

        self.jr_sparsity = 0

        # self.gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
        self.activation = nn.LeakyReLU(args.negative_slope)
        self.gain = torch.nn.init.calculate_gain('leaky_relu', args.negative_slope) ** 2
        self.gen_dummy()

    def gen_dummy(self):
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        self.dummy_weight = torch.Tensor(self.base_params).to(self.device)
        nn.init.normal_(self.dummy_weight, 0, 1)

        mean = self.dummy_weight.data.mean()
        self.dummy_weight.data -= mean
        var = (self.dummy_weight.data ** 2).mean()
        std = var.sqrt()
        self.dummy_weight.data /= std
    
    def get_expand_shape(self, t, add_in, add_out=None, kbts=False):
        # expand from knowledge base weights of task t
        if add_in is None:
            if 'fix' in self.args.mode:
                add_in = self.base_in_features - self.shape_in[t]
            else:
                add_in = self.base_in_features
        fan_in = self.shape_in[t] + add_in
        if add_out is None:
            # compute add_out
            if 'fix' in self.args.mode:
                add_out = self.base_out_features - self.shape_out[t]
            elif 'op' in self.args.mode and not kbts:
                add_out = self.base_out_features
            else:
                if kbts:
                    old_params = self.shape_out[t] * self.shape_in[t] * self.ks
                    total_params = max(self.base_params - old_params, 0) + old_params
                else:
                    total_params = self.shape_out[t] * self.shape_in[t] * self.ks + self.base_params
                fan_out = total_params // (fan_in * self.ks)
                add_out = max(fan_out - self.shape_out[t], 0)

        fan_out = self.shape_out[t] + add_out
        return int(fan_in), int(fan_out), int(add_in), int(add_out)

    def expand(self, add_in, add_out):
        self.task += 1
        fan_in = add_in[0] + self.shape_in[-1]
        fan_in_kbts = add_in[1] + self.shape_in[-1]
        fan_out = add_out[0] + self.shape_out[-1]
        fan_out_kbts = add_out[1] + self.shape_out[-1]

        add_in = add_in[0]
        add_out = add_out[0]

        self.num_out = torch.cat([self.num_out, torch.IntTensor([add_out]).to(self.device)])
        self.num_in = torch.cat([self.num_in, torch.IntTensor([add_in]).to(self.device)])

        self.shape_out = torch.cat([self.shape_out, torch.IntTensor([fan_out]).to(self.device)])
        self.shape_in = torch.cat([self.shape_in, torch.IntTensor([fan_in]).to(self.device)])

        # self.num_out.append(add_out)
        # self.num_in.append(add_in)
        # self.shape_out.append(fan_out)
        # self.shape_in.append(fan_in)
        
        bound_std = math.sqrt(self.gain / (fan_out * self.ks))
        self.bound_std.append(bound_std)

        if isinstance(self, DynamicConv2D):
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in // self.groups, *self.kernel_size).normal_(0, bound_std).to(self.device)))
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.shape_in[-2] // self.groups, *self.kernel_size).normal_(0, bound_std).to(self.device)))
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.shape_out[-2], add_in // self.groups, *self.kernel_size).normal_(0, bound_std).to(self.device)))
            self.score = nn.Parameter(torch.Tensor(fan_out_kbts, fan_in_kbts // self.groups, *self.kernel_size).to(self.device))
        else:
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in).normal_(0, bound_std).to(self.device)))
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.shape_in[-2]).normal_(0, bound_std).to(self.device)))
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.shape_out[-2], add_in).normal_(0, bound_std).to(self.device)))
            self.score = nn.Parameter(torch.Tensor(fan_out_kbts, fan_in_kbts).to(self.device))

        nn.init.kaiming_uniform_(self.score, a=math.sqrt(5))
        self.register_buffer('kbts_mask'+f'_{len(self.num_out)-1}', torch.ones_like(self.score).to(self.device).bool())
        
        self.set_reg_strength()


    def get_kb_params(self, t):
        # get knowledge base parameters for task t
        # kb weight std = 1
        
        self.kb_weight = torch.empty(0).to(self.device)

        for i in range(t):
            if 'scale' not in self.args.ablation:
                fwt_weight_scale = getattr(self, f'fwt_weight_scale_{i}')
                bwt_weight_scale = getattr(self, f'bwt_weight_scale_{i}')
                self.kb_weight = torch.cat([torch.cat([self.kb_weight, self.bwt_weight[i] / bwt_weight_scale], dim=1), 
                                torch.cat([self.fwt_weight[i], self.weight[i]], dim=1) / fwt_weight_scale], dim=0)
            else:
                self.kb_weight = torch.cat([torch.cat([self.kb_weight, self.bwt_weight[i]], dim=1), 
                                torch.cat([self.fwt_weight[i], self.weight[i]], dim=1)], dim=0)

    
    def get_masked_kb_params(self, t, add_in, add_out=None):
        # kb weight std = bound of the model size
        fan_in, fan_out, add_in, add_out = self.get_expand_shape(t, add_in, add_out, kbts=True)

        n_0 = add_out * (fan_in-add_in) * self.ks
        n_1 = fan_out * add_in * self.ks

        if self.dummy_weight is None:
            self.gen_dummy()
        num = (n_0 + n_1) // self.dummy_weight.numel() + 1
        dummy_weight = torch.cat([self.dummy_weight for _ in range(num)])
        # dummy_weight = dummy_weight * getattr(self, f'bound_std_{t}')

        if isinstance(self, DynamicConv2D):
            dummy_weight_0 = dummy_weight[:n_0].view(add_out, (fan_in-add_in) // self.groups, *self.kernel_size)
            dummy_weight_1 = dummy_weight[n_0:n_0+n_1].view(fan_out, add_in // self.groups, *self.kernel_size)
        else:
            dummy_weight_0 = dummy_weight[:n_0].view(add_out, (fan_in-add_in))
            dummy_weight_1 = dummy_weight[n_0:n_0+n_1].view(fan_out, add_in)
        self.masked_kb_weight = torch.cat([torch.cat([self.kb_weight, dummy_weight_0], dim=0), dummy_weight_1], dim=1)
        del dummy_weight, dummy_weight_0, dummy_weight_1

        bound_std = math.sqrt(self.gain / (fan_out * self.ks))
        self.masked_kb_weight = self.masked_kb_weight * bound_std
        return add_out * self.s * self.s

    def ets_forward(self, x, t):
        # get expanded task specific model
        weight = F.dropout(self.kb_weight * getattr(self, f'std_neurons_{t}'), self.dropout, self.training)

        weight = torch.cat([torch.cat([weight, self.bwt_weight[t]], dim=1), 
                                torch.cat([self.fwt_weight[t], self.weight[t]], dim=1)], dim=0)
        
        if isinstance(self, DynamicConv2D):
            output = F.conv2d(x, weight, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.linear(x, weight, None)
        return output
    
    def kbts_forward(self, x, t):
        if self.training and self.score is not None:
            mask = GetSubnet.apply(self.score.abs(), 1-self.kbts_sparsities[t])
            weight = self.masked_kb_weight * mask / (1-self.kbts_sparsities[t])
            self.register_buffer('kbts_mask'+f'_{t}', mask.detach().bool().clone())
        else:
            weight = self.masked_kb_weight * getattr(self, 'kbts_mask'+f'_{t}') / (1-self.kbts_sparsities[t])
        
        if isinstance(self, DynamicConv2D):
            output = F.conv2d(x, weight, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.linear(x, weight, None)
        return output    

    def clear_memory(self):
        # self.dummy_weight = None
        self.kb_weight = None
        self.masked_kb_weight = None

    def count_params(self, t):
        count = 0
        for i in range(t+1):
            count += self.weight[i].numel() + self.fwt_weight[i].numel() + self.bwt_weight[i].numel()
        return count

    def norm_in(self):
        weight = torch.cat([self.fwt_weight[-1], self.weight[-1]], dim=1)
        norm = weight.norm(2, dim=self.dim_in)
        return norm
    
    def set_reg_strength(self):
        self.strength = ((self.shape_in[-1] * self.shape_out[-1] * self.kernel_size[0] * self.kernel_size[1]) /
                        (self.shape_in[-1] + self.shape_out[-1] + self.kernel_size[0] + self.kernel_size[1])) 
        # self.strength = 1 - ((self.shape_in[-1] + self.shape_out[-1] + self.kernel_size[0] + self.kernel_size[1]) / 
        #                         (self.shape_in[-1] * self.shape_out[-1] * self.kernel_size[0] * self.kernel_size[1])) 
        # self.strength = (self.shape_in[-1] * self.num_out[-1] * self.ks)
        # self.strength = self.num_out[-1]
        # self.strength = 1


    def squeeze(self, optim_state, mask_in=None, mask_out=None):
        prune_out = mask_out is not None and mask_out.sum() != self.num_out[-1]
        prune_in = mask_in is not None and mask_in.sum() != self.num_in[-1]
        if prune_out:
            apply_mask_out(self.weight[-1], mask_out, optim_state)
            apply_mask_out(self.fwt_weight[-1], mask_out, optim_state)

            self.num_out[-1] = self.weight[-1].shape[0]
            self.shape_out[-1] = sum(self.num_out)

            mask = torch.ones(self.shape_out[-2], dtype=bool, device=self.device)
            mask = torch.cat([mask, mask_out])
            if self.bias is not None:
                apply_mask_out(self.bias[-1], mask, optim_state)
        
        if prune_in:
            if self.s != 1:
                mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0), self.s, self.s).contiguous().view(-1)
            apply_mask_in(self.weight[-1], mask_in, optim_state)
            apply_mask_in(self.bwt_weight[-1], mask_in, optim_state)

            self.num_in[-1] = self.weight[-1].shape[1]
            self.shape_in[-1] = sum(self.num_in)

        self.mask_out = None
        self.set_reg_strength()
    
    def to_device(self, device):
        self.device = device
        self.num_in = self.num_in.to(device)
        self.num_out = self.num_out.to(device)
        self.shape_in = self.shape_in.to(device)
        self.shape_out = self.shape_out.to(device)
        
        if hasattr(self, 'strength'):
            self.strength = self.strength.to(device)
        for p in self.parameters():
            p.to(device)

        for p in self.buffers():
            p.to(device)

        if self.dummy_weight is not None:
            self.dummy_weight = self.dummy_weight.to(device)      

        if hasattr(self, 'kb_weight'):
            if self.kb_weight is not None:
                self.kb_weight = self.kb_weight.to(device)


class DynamicLinear(_DynamicLayer):

    def __init__(self, in_features, out_features, bias=False, norm_type=None, args=None, s=1):
        self.view_in = [-1, 1]
        self.view_out = [1, -1]
        self.dim_in = [1]
        if s == 1:
            self.dim_out = [0]
        else:
            self.dim_out = [0, 2, 3]
        self.kernel_size = _pair(1)
        self.ks = 1
        super(DynamicLinear, self).__init__(in_features, out_features, bias, norm_type, args, s)
            
        
class _DynamicConvNd(_DynamicLayer):
    def __init__(self, in_features, out_features, kernel_size, 
                stride, padding, dilation, transposed, output_padding, groups, bias, norm_type, args, s):
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        super(_DynamicConvNd, self).__init__(in_features, out_features, bias, norm_type, args, s)
        if in_features % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_features % groups != 0:
            raise ValueError('out_channels must be divisible by groups')


class DynamicConv2D(_DynamicConvNd):
    def __init__(self, in_features, out_features, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=False, norm_type=None, args=None, s=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.view_in = [-1, 1, 1, 1]
        self.view_out = [1, -1, 1, 1]
        self.dim_in = [1, 2, 3]
        self.dim_out = [0, 2, 3]
        self.ks = np.prod(kernel_size)

        super(DynamicConv2D, self).__init__(in_features, out_features, kernel_size, 
                                            stride, padding, dilation, False, _pair(0), groups, bias, norm_type, args, s)

class DynamicBlock(nn.Module):
    # A block of dynamic layers, normalization, and activation. All layers are share the number of out features.
    def __init__(self, layers, norm_type=None, args=None, act='relu'):
        super(DynamicBlock, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.args = args
        self.device = args.device
        self.norm_type = norm_type
        self.ets_norm_layers = nn.ModuleList([])
        self.kbts_norm_layers = nn.ModuleList([])
        if act == 'relu':
            self.activation = nn.LeakyReLU(args.negative_slope, inplace=True)
            self.gain = torch.nn.init.calculate_gain('leaky_relu', args.negative_slope) ** 2
        else:
            self.activation = nn.Identity()
            self.gain = 1
        self.register_buffer('task', torch.tensor(-1, dtype=torch.int).to(self.device))
        self.mask_out = None
        self.last = False
        self.strength = torch.tensor(1)
    
    def to_device(self, device):
        self.task.to(device)
        self.device = device
        self.strength = self.strength.to(device)

    def ets_forward(self, inputs, t):
        out = 0
        for x, layer in zip(inputs, self.layers):
            out = out + layer.ets_forward(x, t)
            
        # out = self.activation(self.ets_norm_layers[t](out))
        out = self.activation(out)
        return out
    
    def kbts_forward(self, inputs, t):
        out = 0
        for x, layer in zip(inputs, self.layers):
            out = out + layer.kbts_forward(x, t)
        
        out = self.activation(self.kbts_norm_layers[t](out))
        return out
    
    def expand(self, add_ins, add_outs):
        self.task += 1
        add_outs_ = []
        add_outs_kbts_ = []
        add_ins_ = []
        for add_in, add_out, layer in zip(add_ins, add_outs, self.layers):
            _, _, add_in_, add_out_ = layer.get_expand_shape(-1, add_in[0], add_out[0])
            _, _, add_in_kbts_, add_out_kbts_ = layer.get_expand_shape(-1, add_in[1], add_out[1], kbts=True)
            add_outs_.append(add_out_)
            add_outs_kbts_.append(add_out_kbts_)
            add_ins_.append((add_in_, add_in_kbts_))

        add_out = min(add_outs_)
        add_out_kbts = min(add_outs_kbts_)

        if self.norm_type is None:
            self.ets_norm_layers.append(nn.Identity())
            self.kbts_norm_layers.append(nn.Identity())
        else:
            if 'affine' in self.norm_type:
                affine = True
            else:
                affine = False
            if 'track' in self.norm_type:
                track_running_stats = True
            else:
                track_running_stats = False
            self.ets_norm_layers.append(DynamicNorm(layer.shape_out[-1] + add_out, affine=affine, track_running_stats=track_running_stats).to(self.device))
            self.kbts_norm_layers.append(DynamicNorm(layer.shape_out[-1] + add_out_kbts, affine=affine, track_running_stats=track_running_stats).to(self.device))
            # self.kbts_norm_layers.append(nn.BatchNorm2d(layer.shape_out[-1] + add_out_kbts, affine=affine, track_running_stats=track_running_stats).to(self.device))
            # self.ets_norm_layers.append(nn.BatchNorm2d(layer.shape_out[-1] + add_out, affine=affine, track_running_stats=track_running_stats).to(self.device))

        for add_in, layer in zip(add_ins_, self.layers):
            layer.expand(add_in, (add_out, add_out_kbts))

        self.strength = max([layer.strength for layer in self.layers])
        return (add_out, add_out_kbts)
    
    def get_masked_kb_params(self, t, add_ins, add_outs):
        add_outs_ = []
        add_ins_ = []
        for add_in, add_out, layer in zip(add_ins, add_outs, self.layers):
            _, _, add_in_, add_out_ = layer.get_expand_shape(t, add_in, add_out, kbts=True)
            add_outs_.append(add_out_)
            add_ins_.append(add_in_)

        add_out = min(add_outs_)
        for add_in, layer in zip(add_ins_, self.layers):
            layer.get_masked_kb_params(t, add_in, add_out)
        return add_out

    def squeeze(self, optim_state, mask_ins):
        for mask_in, layer in zip(mask_ins, self.layers):
            layer.squeeze(optim_state, mask_in, self.mask_out)
        self.strength = max([layer.strength for layer in self.layers])
        if self.norm_type is not None and self.mask_out is not None:
            mask = torch.ones(self.layers[0].shape_out[-2], dtype=bool, device=self.device)
            mask = torch.cat([mask, self.mask_out])
            if self.ets_norm_layers[-1].affine:
                apply_mask_out(self.ets_norm_layers[-1].weight, mask, optim_state)
                apply_mask_out(self.ets_norm_layers[-1].bias, mask, optim_state)
                self.ets_norm_layers[-1].num_features = self.ets_norm_layers[-1].weight.shape[0]
            
            if self.ets_norm_layers[-1].track_running_stats:
                self.ets_norm_layers[-1].running_mean = self.ets_norm_layers[-1].running_mean[mask]
                self.ets_norm_layers[-1].running_var = self.ets_norm_layers[-1].running_var[mask]
                self.ets_norm_layers[-1].num_features = self.ets_norm_layers[-1].running_mean.shape[0]

    def initialize(self):
        # def compute_scale(layer, i, j):
        #     # if getattr(layer, f'weight_{i}_{j}').numel() <= getattr(layer, f'weight_{i}_{j}').size(0):
        #     if getattr(layer, f'weight_{i}_{j}').numel() == 0:
        #         return torch.tensor(1).to(self.device)
        #     else:
        #         var = (getattr(layer, f'weight_{i}_{j}').data ** 2).mean(layer.dim_in).detach()
        #         # print(var)
        #         return var.sqrt().view(layer.view_in)
        # # Initialize new weights and rescale old weights to have the same variance:
        
        sum_var = 0
        for layer in self.layers:
            sum_var += layer.ks * layer.shape_out[-1]

        std = math.sqrt(self.gain / sum_var)
        for layer in self.layers:
            # initialize new weights
            nn.init.normal_(layer.fwt_weight[-1], 0, std)
            nn.init.normal_(layer.bwt_weight[-1], 0, std)
            nn.init.normal_(layer.weight[-1], 0, std)

            # compute variance of old weights
            i = self.task - 1
            if self.task > 0:
                fwt_weight = torch.cat([layer.fwt_weight[i], layer.weight[i]], dim=1)
                if fwt_weight.numel() != 0:
                    w_std = (fwt_weight.data ** 2).mean(dim=layer.dim_in).sqrt()
                    layer.register_buffer(f'fwt_weight_scale_{i}', w_std.view(layer.view_in))
                else:
                    layer.register_buffer(f'fwt_weight_scale_{i}', torch.ones(1).to(layer.device).view(layer.view_in))

                bwt_weight = layer.bwt_weight[i]
                if bwt_weight.numel() != 0:
                    w_std = (bwt_weight.data ** 2).mean(dim=layer.dim_in).sqrt()
                    layer.register_buffer(f'bwt_weight_scale_{i}', w_std.view(layer.view_in))
                else:
                    layer.register_buffer(f'bwt_weight_scale_{i}', torch.ones(1).to(layer.device).view(layer.view_in))

        # # initial equal var for old neurons
        # # self.register_buffer(f'old_var_{self.task}', (std ** 2) * torch.ones(self.task).to(self.device))

        # for layer in self.layers:
        #     # rescale old weights
        #     # layer.register_buffer(f'old_var_{self.task}', getattr(self, f'old_var_{self.task}').data)
        #     layer.register_buffer(f'bound_std_{self.task}', torch.tensor(std).to(self.device))
        #     if self.task > 0:
        #         for i in range(self.task-1):
        #             layer.register_buffer(f'scale_{i}_{self.task-1}', compute_scale(layer, i, self.task-1))
        #             layer.register_buffer(f'scale_{self.task-1}_{i}', compute_scale(layer, self.task-1, i))
        #         layer.register_buffer(f'scale_{self.task-1}_{self.task-1}', compute_scale(layer, self.task-1, self.task-1))

        #     # initialize new weights
        #     for i in range(self.task):
        #         nn.init.normal_(getattr(layer, f'weight_{i}_{self.task}'), 0, std)
        #         nn.init.normal_(getattr(layer, f'weight_{self.task}_{i}'), 0, std)
        #     nn.init.normal_(getattr(layer, f'weight_{self.task}_{self.task}'), 0, std)

        self.check_var()
        self.normalize()
        self.check_var()

    def normalize(self):
        
        var_layers_out = 0
        for layer in self.layers:
            mean = layer.bwt_weight[-1].data.mean(layer.dim_in)
            layer.bwt_weight[-1].data -= mean.view(layer.view_in)
            bwt_var = (layer.bwt_weight[-1].data ** 2).mean(layer.dim_in)
            layer.register_buffer(f'std_neurons_{self.task}', bwt_var.sqrt().view(layer.view_in).clone())

            fwt_weight = torch.cat([layer.fwt_weight[-1], layer.weight[-1]], dim=1)
            mean = fwt_weight.data.mean(layer.dim_in)
            layer.fwt_weight[-1].data -= mean.view(layer.view_in)
            layer.weight[-1].data -= mean.view(layer.view_in)
            fwt_weight = torch.cat([layer.fwt_weight[-1], layer.weight[-1]], dim=1)
            fwt_var = (fwt_weight.data ** 2).mean(layer.dim_in)

            var = torch.cat([bwt_var, fwt_var], dim=0)
            var_layers_out += layer.ks * var
        
        var_layers_out /= self.gain
        std = var_layers_out.sum().sqrt()

        for layer in self.layers:
            layer.bwt_weight[-1].data /= std
            getattr(layer, f'std_neurons_{self.task}').data /= std
            layer.fwt_weight[-1].data /= std
            layer.weight[-1].data /= std

        # def layer_wise(layer, i, j):
        #     if getattr(layer, f'weight_{i}_{j}').numel() == 0:
        #     # if getattr(layer, f'weight_{i}_{j}').numel() <= getattr(layer, f'weight_{i}_{j}').size(0):
        #         return torch.zeros(getattr(layer, f'weight_{i}_{j}').shape[0]).to(self.device)
        #     mean = getattr(layer, f'weight_{i}_{j}').data.mean(layer.dim_in)
        #     getattr(layer, f'weight_{i}_{j}').data -= mean.view(layer.view_in)
        #     var = (getattr(layer, f'weight_{i}_{j}').data ** 2).mean(layer.dim_in)
        #     return var
        
        # var_new_neurons = []
        # var_layers_in = 0
        # for i in range(self.task):
        #     var_layers_out = 0
        #     for layer in self.layers:
        #         var_layers_out += layer.ks * layer_wise(layer, i, self.task)
        #         var_layers_in += layer.ks * layer_wise(layer, self.task, i).sum()
        #     var_new_neurons.append(var_layers_out)

        # var_layers_out = 0
        # for layer in self.layers:
        #     var = layer.ks * layer_wise(layer, self.task, self.task)
        #     var_layers_out += var

        # var_new_neurons.append(var_layers_out)
        # var_new_neurons = torch.stack(var_new_neurons, dim=0)
        # var_new_neurons /= self.gain # shape (num task, num new neurons)
        
        # const = sum([layer.ks * layer.shape_out[-2] * getattr(layer, f'bound_std_{self.task}') ** 2 for layer in self.layers]) / self.gain
        # var_tasks = var_new_neurons.sum(1) # shape (num task)
        # var_tasks[self.task] += (var_layers_in / self.gain)
        # if self.task > 0:
        #     var_tasks[:self.task] += const
        #     # getattr(self, f'old_var_{self.task}').data /= var_tasks[:self.task]

        # std_new_neurons = var_tasks.sqrt()
        # std_old_neurons = var_tasks[self.task].sqrt()
        # for layer in self.layers:
        #     # layer.register_buffer(f'old_var_{self.task}', getattr(self, f'old_var_{self.task}').data)
        #     # layer.register_buffer(f'bound_std_{self.task}', std_old_neurons)
        #     for i in range(self.task):
        #         getattr(layer, f'weight_{i}_{self.task}').data /= std_new_neurons[i].view(layer.view_in)
        #         getattr(layer, f'weight_{self.task}_{i}').data /= std_old_neurons
        #     getattr(layer, f'weight_{self.task}_{self.task}').data /= std_new_neurons[self.task].view(layer.view_in)

        # if self.norm_type is not None and 'scale' not in self.args.ablation:
        #     out_scale = (std_new_neurons**2).sum(0).sqrt() # shape (num new neurons)
        #     if self.ets_norm_layers[-1].track_running_stats:
        #         self.ets_norm_layers[-1].running_mean[layer.shape_out[-2]:] /= out_scale
        #         self.ets_norm_layers[-1].running_var[layer.shape_out[-2]:] /= (out_scale ** 2)

        #     if self.ets_norm_layers[-1].affine:
        #         self.ets_norm_layers[-1].weight.data[layer.shape_out[-2]:] /= out_scale
        #         self.ets_norm_layers[-1].bias.data[layer.shape_out[-2]:] /= out_scale
            

    def proximal_gradient_descent(self, lr=0, lamb=0, total_strength=1):

        var_old_neurons = 0
        var_new_neurons = 0
        for layer in self.layers:
            mean = layer.bwt_weight[-1].data.mean(layer.dim_in)
            layer.bwt_weight[-1].data -= mean.view(layer.view_in)
            bwt_var = (layer.bwt_weight[-1].data ** 2).mean(layer.dim_in)
            layer.register_buffer(f'std_neurons_{self.task}', bwt_var.sqrt().view(layer.view_in).clone())

            fwt_weight = torch.cat([layer.fwt_weight[-1], layer.weight[-1]], dim=1)
            mean = fwt_weight.data.mean(layer.dim_in)
            layer.fwt_weight[-1].data -= mean.view(layer.view_in)
            layer.weight[-1].data -= mean.view(layer.view_in)
            fwt_weight = torch.cat([layer.fwt_weight[-1], layer.weight[-1]], dim=1)
            fwt_var = (fwt_weight.data ** 2).mean(layer.dim_in)

            var_old_neurons += layer.ks * bwt_var
            var_new_neurons += layer.ks * fwt_var
        
        var_old_neurons /= self.gain
        var_new_neurons /= self.gain
        strength = self.strength / total_strength
        aux = 1 - lamb * lr * strength / var_new_neurons.sqrt()
        aux = F.threshold(aux, 0, 0, False)
        self.mask_out = (aux > 0).clone().detach()

        # Normalize the new weights so it will not vanishing during pruning
        sum_std_old = var_new_neurons.sum().sqrt()
        sum_std_new = (var_new_neurons * (aux ** 2)).sum().sqrt()
        aux = aux * sum_std_old / sum_std_new

        std = torch.cat([var_old_neurons, var_new_neurons], dim=0).sum().sqrt()

        for layer in self.layers:
            layer.bwt_weight[-1].data /= std
            getattr(layer, f'std_neurons_{self.task}').data /= std
            layer.fwt_weight[-1].data /= (std / aux).view(layer.view_in)
            layer.weight[-1].data /= (std / aux).view(layer.view_in)

        self.check_var()

        # def layer_wise(layer, i, j):
        #     # if getattr(layer, f'weight_{i}_{j}').numel() <= getattr(layer, f'weight_{i}_{j}').size(0):
        #     if getattr(layer, f'weight_{i}_{j}').numel() == 0:
        #         return torch.zeros(getattr(layer, f'weight_{i}_{j}').shape[0]).to(self.device)
        #     mean = getattr(layer, f'weight_{i}_{j}').data.mean(layer.dim_in)
        #     getattr(layer, f'weight_{i}_{j}').data -= mean.view(layer.view_in)
        #     var = (getattr(layer, f'weight_{i}_{j}').data ** 2).mean(layer.dim_in)
        #     return var
        
        # var_new_neurons = []
        # var_layers_in = 0
        # for i in range(self.task):
        #     var_layers_out = 0
        #     for layer in self.layers:
        #         var_layers_out += layer.ks * layer_wise(layer, i, self.task)
        #         var_layers_in += layer.ks * layer_wise(layer, self.task, i).sum()
        #     var_new_neurons.append(var_layers_out)

        # var_layers_out = 0
        # for layer in self.layers:
        #     var = layer.ks * layer_wise(layer, self.task, self.task)
        #     var_layers_out += var

        # var_new_neurons.append(var_layers_out)
        # var_new_neurons = torch.stack(var_new_neurons, dim=0)
        # var_new_neurons /= self.gain # shape (num task, num new neurons)
        # strength = self.strength / total_strength
        # aux = 1 - lamb * lr * strength / var_new_neurons.sum(0).sqrt()
        # aux = F.threshold(aux, 0, 0, False) # shape (num new neurons)
        # self.mask_out = (aux > 0).clone().detach() # shape (num new neurons)
        
        # const = sum([layer.ks * layer.shape_out[-2] * getattr(layer, f'bound_std_{self.task}') ** 2 for layer in self.layers]) / self.gain
        # var_tasks = (var_new_neurons * (aux**2).view(1, -1)).sum(1) # shape (num task)
        # var_tasks[self.task] += (var_layers_in / self.gain)
        # if self.task > 0:
        #     var_tasks[:self.task] += const
        #     # getattr(self, f'old_var_{self.task}').data /= var_tasks[:self.task]

        # std_new_neurons = var_tasks.sqrt().view(-1, 1) / aux.view(1, -1) # shape (num task, num new neurons)
        # std_old_neurons = var_tasks[self.task].sqrt() # shape (0)
        # for layer in self.layers:
        #     # layer.register_buffer(f'old_var_{self.task}', getattr(self, f'old_var_{self.task}').data)
        #     # layer.register_buffer(f'bound_std_{self.task}', std_old_neurons)
        #     for i in range(self.task):
        #         getattr(layer, f'weight_{i}_{self.task}').data /= std_new_neurons[i].view(layer.view_in)
        #         getattr(layer, f'weight_{self.task}_{i}').data /= std_old_neurons
        #     getattr(layer, f'weight_{self.task}_{self.task}').data /= std_new_neurons[self.task].view(layer.view_in)

        # if self.norm_type is not None and 'scale' not in self.args.ablation:
        #     out_scale = (std_new_neurons**2).sum(0).sqrt() # shape (num new neurons)
        #     if self.ets_norm_layers[-1].track_running_stats:
        #         self.ets_norm_layers[-1].running_mean[layer.shape_out[-2]:] /= out_scale
        #         self.ets_norm_layers[-1].running_var[layer.shape_out[-2]:] /= (out_scale ** 2)

        #     if self.ets_norm_layers[-1].affine:
        #         self.ets_norm_layers[-1].weight.data[layer.shape_out[-2]:] /= out_scale
        #         self.ets_norm_layers[-1].bias.data[layer.shape_out[-2]:] /= out_scale
        
        # if self.task > 0:
        #     self.check_var()
        
    def check_var(self):
        mean = 0
        var = 0
        for layer in self.layers:
            bwt_mean = layer.bwt_weight[-1].data.mean(layer.dim_in)
            bwt_var = ((layer.bwt_weight[-1].data - bwt_mean.view(layer.view_in)) ** 2).mean(layer.dim_in)

            fwt_weight = torch.cat([layer.fwt_weight[-1], layer.weight[-1]], dim=1)
            fwt_mean = fwt_weight.data.mean(layer.dim_in)
            fwt_var = ((fwt_weight.data - fwt_mean.view(layer.view_in)) ** 2).mean(layer.dim_in)

            var += layer.ks * torch.cat([bwt_var, fwt_var], dim=0)
            mean += torch.cat([bwt_mean, fwt_mean], dim=0)
        
        var /= self.gain
        for l, layer in enumerate(self.layers):
            print(l, layer.shape_in[-1].item(), layer.shape_out[-1].item(), layer.ks, end=' - ')
        print()
        print(f'mean: {mean.sum()}, var: {var.sum()}')

        # def layer_wise(layer, i, j):
        #     w = getattr(layer, f'weight_{i}_{j}')
        #     # if w.numel() <= w.size(0):
        #     if w.numel() == 0:
        #         return torch.zeros(w.shape[0]).to(self.device)
            
        #     if hasattr(layer, f'scale_{i}_{j}'):
        #         w = w * getattr(layer, f'bound_std_{self.task}') / getattr(layer, f'scale_{i}_{j}')
        #     var = (w.data ** 2).mean(layer.dim_in)

        #     return var
        
        # for l, layer in enumerate(self.layers):
        #     print(l, layer.shape_in[-1].item(), layer.shape_out[-1].item(), layer.ks, end=' - ')
        # print()

        # for i in range(self.task+1):
        #     var_layers_in = 0
        #     for j in range(self.task+1):
        #         for layer in self.layers:
        #             var_layers_in += layer.ks * layer_wise(layer, i, j).sum()
        #     print(i, var_layers_in.item()/self.gain)

    def get_optim_ets_params(self):
        params = []
        for layer in self.layers:
            params += [getattr(layer, f'weight_{self.task}_{self.task}')]
            for i in range(self.task):
                params += [getattr(layer, f'weight_{i}_{self.task}'), getattr(layer, f'weight_{self.task}_{i}')]
    
        if self.norm_type is not None and 'affine' in self.norm_type:
            params += [self.ets_norm_layers[-1].weight, self.ets_norm_layers[-1].bias]
        return params
    
    def get_optim_kbts_params(self):
        params = []
        if self.norm_type is not None and 'affine' in self.norm_type:
            params += [self.kbts_norm_layers[-1].weight, self.kbts_norm_layers[-1].bias]
        return params, [layer.score for layer in self.layers]
    
    def freeze(self, state=False):
        for layer in self.layers:
            for i in range(self.task):
                for j in range(self.task):
                    getattr(layer, f'weight_{i}_{j}').requires_grad = state

            if layer.score is not None:
                t = len(layer.kbts_sparsities) - 1
                mask = GetSubnet.apply(layer.score.abs(), 1-layer.kbts_sparsities[t])
                layer.register_buffer('kbts_mask'+f'_{t}', mask.detach().bool().clone())
                layer.score = None
        if self.norm_type is not None:
            if 'affine' in self.norm_type:
                self.ets_norm_layers[-1].weight.requires_grad = state
                self.kbts_norm_layers[-1].weight.requires_grad = state
                self.ets_norm_layers[-1].bias.requires_grad = state
                self.kbts_norm_layers[-1].bias.requires_grad = state
            self.ets_norm_layers[-1].track_running_stats = state
            self.kbts_norm_layers[-1].track_running_stats = state


class DynamicClassifier(DynamicLinear):

    def __init__(self, in_features, out_features, bias=True, norm_type=None, args=None, s=1):
        super(DynamicClassifier, self).__init__(in_features, out_features, bias, norm_type, args, s)
        self.weight_ets = nn.ParameterList([])
        self.weight_kbts = nn.ParameterList([])
        self.use_bias = bias
        if bias:
            self.bias_ets = nn.ParameterList([])
            self.bias_kbts = nn.ParameterList([])

    def ets_forward(self, x, t): 
        weight = self.weight_ets[t]
        bias = self.bias_ets[t] if self.use_bias else None
        out = F.linear(x, weight, bias)
        return out
    
    def kbts_forward(self, x, t):
        weight = self.weight_kbts[t]
        bias = self.bias_kbts[t] if self.use_bias else None
        out = F.linear(x, weight, bias)
        return out
    
    def expand(self, add_in, add_out):
        fan_in = add_in[0] + self.shape_in[-1]
        fan_in_kbts = add_in[1] + self.shape_in[-1]
        fan_out = add_out[0] + self.shape_out[-1]
        fan_out_kbts = add_out[1] + self.shape_out[-1]

        add_in = add_in[0]
        add_out = add_out[0]

        self.num_out = torch.cat([self.num_out, torch.IntTensor([add_out]).to(self.device)])
        self.num_in = torch.cat([self.num_in, torch.IntTensor([add_in]).to(self.device)])
        self.shape_out = torch.cat([self.shape_out, torch.IntTensor([self.shape_out[-1] + add_out]).to(self.device)])
        self.shape_in = torch.cat([self.shape_in, torch.IntTensor([self.shape_in[-1] + add_in]).to(self.device)])

        # self.num_out.append(add_out)
        # self.num_in.append(add_in)
        # self.shape_out.append(fan_out)
        # self.shape_in.append(fan_in)

        bound_std = self.gain / math.sqrt(self.shape_in[-1])
        # bound_std = self.gain / math.sqrt(self.num_out[-1])
        self.weight_ets.append(nn.Parameter(torch.Tensor(self.num_out[-1], self.shape_in[-1]).normal_(0, bound_std).to(self.device)))

        bound_std = self.gain / math.sqrt(fan_in_kbts)
        # bound_std = self.gain / math.sqrt(self.num_out[-1])
        self.weight_kbts.append(nn.Parameter(torch.Tensor(self.num_out[-1], fan_in_kbts).normal_(0, bound_std).to(self.device)))

        if self.use_bias:
            self.bias_ets.append(nn.Parameter(torch.zeros(self.num_out[-1]).to(self.device))) 
            self.bias_kbts.append(nn.Parameter(torch.zeros(self.num_out[-1]).to(self.device)))
        
    
    def freeze(self, state=False):
        self.weight_ets[-1].requires_grad = state
        self.weight_kbts[-1].requires_grad = state
        if self.use_bias:
            self.bias_ets[-1].requires_grad = state
            self.bias_kbts[-1].requires_grad = state

    def get_optim_ets_params(self):
        if self.use_bias:
            return [self.weight_ets[-1], self.bias_ets[-1]]
        else:
            return [self.weight_ets[-1]]

    def get_optim_kbts_params(self):
        if self.use_bias:
            return [self.weight_kbts[-1], self.bias_kbts[-1]]
        else:
            return [self.weight_kbts[-1]]

    def count_params(self, t):
        count = 0
        for i in range(t+1):
            count += self.weight_ets[i].numel()
            count += self.weight_kbts[i].numel()
            if self.use_bias:
                count += self.bias_ets[i].numel()
                count += self.bias_kbts[i].numel()
        return count

    def squeeze(self, optim_state, mask_in=None, mask_out=None):
        # prune_out = mask_out is not None and mask_out.sum() != self.num_out[-1]
        prune_in = mask_in is not None and mask_in.sum() != self.num_in[-1]
        if prune_in:
            if self.s != 1:
                mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0), self.s, self.s).contiguous().view(-1)
            
            mask = torch.ones(self.shape_in[-2], dtype=bool, device=self.device)
            mask = torch.cat([mask, mask_in])
            apply_mask_in(self.weight_ets[-1], mask, optim_state)
            self.shape_in[-1] = self.weight_ets[-1].shape[1]
        
    def initialize(self):
        nn.init.normal_(self.weight_ets[-1], 0, 1 / math.sqrt(self.weight_ets[-1].shape[0]))
        nn.init.normal_(self.weight_kbts[-1], 0, 1 / math.sqrt(self.weight_kbts[-1].shape[0]))
    
    def normalize(self):
        mean = self.weight_ets[-1].data.mean(self.dim_in)
        self.weight_ets[-1].data -= mean.view(self.view_in)
        var = (self.weight_ets[-1].data ** 2).mean(self.dim_in)
        std = var.sum(0).sqrt()
        self.weight_ets[-1].data /= std

        mean = self.weight_kbts[-1].data.mean(self.dim_in)
        self.weight_kbts[-1].data -= mean.view(self.view_in)
        var = (self.weight_kbts[-1].data ** 2).mean(self.dim_in)
        std = var.sum(0).sqrt()
        self.weight_kbts[-1].data /= std

class DynamicNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(DynamicNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.num_features = num_features
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float).to(self.device))
            self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float).to(self.device))

        if self.track_running_stats:
            self.register_buffer(f'running_mean', torch.zeros(num_features).to(self.device))
            self.register_buffer(f'running_var', torch.ones(num_features).to(self.device))
            self.register_buffer(f'num_batches_tracked', torch.tensor(0, dtype=torch.long).to(self.device))
        else:
            self.register_buffer(f'running_mean', None)
            self.register_buffer(f'running_var', None)
            self.register_buffer(f'num_batches_tracked', None)


    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked += 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        if len(input.shape) == 4:
            mean = input.mean([0, 2, 3])
            shape = (1, -1, 1, 1)
            var = ((input - mean.view(shape)) ** 2).mean([0, 2, 3])
        else:
            mean = input.mean([0])
            shape = (1, -1)
            var = ((input - mean.view(shape)) ** 2).mean([0])

        # calculate running estimates
        if bn_training:
            if self.track_running_stats:
                n = input.numel() / input.size(1)
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        # var = var.sum() / (2*3)
        # var = var.mean()
        output = (input - mean.view(shape)) / (torch.sqrt(var.view(shape) + self.eps))
    
        if self.affine:
            output = output * self.weight.view(shape) + self.bias.view(shape)

        return output
    
    def squeeze(self, mask, optim_state):
        if self.affine:
            apply_mask_out(self.weight, mask, optim_state)
            apply_mask_out(self.bias, mask, optim_state)
            self.num_features = self.weight.shape[0]
        
        if self.track_running_stats:
            self.running_mean = self.running_mean[mask]
            self.running_var = self.running_var[mask]
            self.num_features = self.running_mean.shape[0]


class DynamicNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True, device='cuda'):
        super(DynamicNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.num_features = num_features
        self.device = device
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features).to(device))
            self.bias = nn.Parameter(torch.zeros(num_features).to(device))

        if self.track_running_stats:
            self.register_buffer(f'running_mean', torch.zeros(num_features).to(device))
            self.register_buffer(f'running_var', torch.ones(num_features).to(device))
            self.register_buffer(f'num_batches_tracked', torch.tensor(0, dtype=torch.long).to(device))
        else:
            self.register_buffer(f'running_mean', None)
            self.register_buffer(f'running_var', None)
            self.register_buffer(f'num_batches_tracked', None)


    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked += 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        if len(input.shape) == 4:
            mean = input.mean([0, 2, 3])
            shape = (1, -1, 1, 1)
            var = ((input - mean.view(shape)) ** 2).mean([0, 2, 3])
        else:
            mean = input.mean([0])
            shape = (1, -1)
            var = ((input - mean.view(shape)) ** 2).mean([0])

        # calculate running estimates
        if bn_training:
            if self.track_running_stats:
                n = input.numel() / input.size(1)
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        var = var.sum()
        # var = var.sum()
        output = (input - mean.view(shape)) / (torch.sqrt(var + self.eps))
    
        if self.affine:
            output = output * self.weight.view(shape) + self.bias.view(shape)

        return output
    
    def squeeze(self, mask, optim_state):
        if self.affine:
            apply_mask_out(self.weight, mask, optim_state)
            apply_mask_out(self.bias, mask, optim_state)
            self.num_features = self.weight.shape[0]
        
        if self.track_running_stats:
            self.running_mean = self.running_mean[mask]
            self.running_var = self.running_var[mask]
            self.num_features = self.running_mean.shape[0]

    
            
