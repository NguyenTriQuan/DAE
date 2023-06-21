import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor, device, isin, seed
from typing import Optional, Any
from torch.nn import init
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from utils import *
from typing import Optional, List, Tuple, Union
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.dropout = args.dropout
        self.sparsity = args.sparsity
        self.s = s
        self.base_in_features = in_features
        self.base_out_features = out_features
        self.base_params = self.base_out_features * self.base_in_features * self.ks

        self.weight = nn.ParameterList([])
        self.fwt_weight = nn.ParameterList([])
        self.bwt_weight = nn.ParameterList([])

        self.mask_in = None
        self.mask_out = None

        self.register_buffer('bias', None)

        self.register_buffer('shape_out', torch.IntTensor([0]).to(device))
        self.register_buffer('shape_in', torch.IntTensor([0]).to(device))
        self.register_buffer('num_out', torch.IntTensor([]).to(device))
        self.register_buffer('num_in', torch.IntTensor([]).to(device))
        # self.register_buffer('kbts_sparsities', torch.IntTensor([]).to(device))

        # self.shape_out = [0]
        # self.shape_in = [0]
        # self.num_out = []
        # self.num_in = []
        self.kbts_sparsities = []
        self.old_std = []
        self.total_masked_kb = []

        self.jr_sparsity = 0

        # self.gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
        self.gain = torch.nn.init.calculate_gain('leaky_relu', 0)
        self.gen_dummy()

    def gen_dummy(self):
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        self.dummy_weight = torch.Tensor(self.base_params).to(device)
        nn.init.normal_(self.dummy_weight, 0, 1)
    
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
        if len(self.weight) > 0:
            self.update_scale()
        fan_in = add_in[0] + self.shape_in[-1]
        fan_in_kbts = add_in[1] + self.shape_in[-1]
        fan_out = add_out[0] + self.shape_out[-1]
        fan_out_kbts = add_out[1] + self.shape_out[-1]

        add_in = add_in[0]
        add_out = add_out[0]

        self.num_out = torch.cat([self.num_out, torch.IntTensor([add_out]).to(device)])
        self.num_in = torch.cat([self.num_in, torch.IntTensor([add_in]).to(device)])

        self.shape_out = torch.cat([self.shape_out, torch.IntTensor([fan_out]).to(device)])
        self.shape_in = torch.cat([self.shape_in, torch.IntTensor([fan_in]).to(device)])

        # self.num_out.append(add_out)
        # self.num_in.append(add_in)
        # self.shape_out.append(fan_out)
        # self.shape_in.append(fan_in)
        
        bound_std = self.gain / math.sqrt(fan_in * self.ks)
        # bound_std = self.gain / math.sqrt(fan_out * self.ks)
        self.old_std.append(bound_std)
        if isinstance(self, DynamicConv2D):
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.shape_in[-2] // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.shape_out[-2], add_in // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))
            self.score = nn.Parameter(torch.Tensor(fan_out_kbts, fan_in_kbts // self.groups, *self.kernel_size).to(device))
        else:
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in).normal_(0, bound_std).to(device)))
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.shape_in[-2]).normal_(0, bound_std).to(device)))
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.shape_out[-2], add_in).normal_(0, bound_std).to(device)))
            self.score = nn.Parameter(torch.Tensor(fan_out_kbts, fan_in_kbts).to(device))

        nn.init.kaiming_uniform_(self.score, a=math.sqrt(5))
        self.register_buffer('kbts_mask'+f'_{len(self.num_out)-1}', torch.ones_like(self.score).to(device).bool())
        
        self.set_reg_strength()

    def get_kb_params(self, t):
        # get knowledge base parameters for task t
        # kb weight std = 1
        
        self.kb_weight = torch.empty(0).to(device)

        for i in range(t):
            weight_scale = getattr(self, f'weight_scale_{i}')
            fwt_weight_scale = getattr(self, f'fwt_weight_scale_{i}')
            bwt_weight_scale = getattr(self, f'bwt_weight_scale_{i}')
            self.kb_weight = torch.cat([torch.cat([self.kb_weight, self.bwt_weight[i] / bwt_weight_scale], dim=1), 
                                torch.cat([self.fwt_weight[i] / fwt_weight_scale, self.weight[i] / weight_scale], dim=1)], dim=0)
        #     self.kb_weight = torch.cat([torch.cat([self.kb_weight, self.bwt_weight[i]], dim=1), 
        #                         torch.cat([self.fwt_weight[i], self.weight[i]], dim=1)], dim=0)
        # if self.kb_weight.numel() != 0:
        #     old_bound_std = self.gain / math.sqrt(self.kb_weight.shape[1] * self.ks)
        #     self.kb_weight = self.kb_weight / old_bound_std

    
    def get_masked_kb_params(self, t, add_in, add_out=None):
        # kb weight std = bound of the model size
        fan_in, fan_out, add_in, add_out = self.get_expand_shape(t, add_in, add_out, kbts=True)

        n_0 = add_out * (fan_in-add_in) * self.ks
        n_1 = fan_out * add_in * self.ks

        if self.dummy_weight is None:
            self.gen_dummy()
        num = (n_0 + n_1) // self.dummy_weight.numel() + 1
        dummy_weight = torch.cat([self.dummy_weight for _ in range(num)])

        if isinstance(self, DynamicConv2D):
            dummy_weight_0 = dummy_weight[:n_0].view(add_out, (fan_in-add_in) // self.groups, *self.kernel_size)
            dummy_weight_1 = dummy_weight[n_0:n_0+n_1].view(fan_out, add_in // self.groups, *self.kernel_size)
        else:
            dummy_weight_0 = dummy_weight[:n_0].view(add_out, (fan_in-add_in))
            dummy_weight_1 = dummy_weight[n_0:n_0+n_1].view(fan_out, add_in)
        self.masked_kb_weight = torch.cat([torch.cat([self.kb_weight, dummy_weight_0], dim=0), dummy_weight_1], dim=1)
        
        bound_std = self.gain / math.sqrt(fan_in * self.ks)
        # bound_std = self.gain / math.sqrt(fan_out * self.ks)
        self.masked_kb_weight = self.masked_kb_weight * bound_std
        return add_out * self.s * self.s

    def ets_forward(self, x, t):
        # get expanded task specific model
        # bound_std = self.gain / math.sqrt(self.shape_in[t+1] * self.ks)
        # bound_std = self.gain / math.sqrt(self.shape_out[t+1] * self.ks)
        weight = self.kb_weight
        weight = weight * self.old_std[t]
        weight = F.dropout(weight, self.dropout, self.training)
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
            mask = getattr(self, 'kbts_mask'+f'_{t}')
            weight = self.masked_kb_weight * mask / (1-self.kbts_sparsities[t])
        
        if isinstance(self, DynamicConv2D):
            output = F.conv2d(x, weight, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.linear(x, weight, None)
        
        return output    

    def clear_memory(self):
        # if self.score is not None:
        #     t = len(self.kbts_sparsities) - 1
        #     mask = GetSubnet.apply(self.score.abs(), 1-self.kbts_sparsities[t])
        #     self.register_buffer('kbts_mask'+f'_{t}', mask.detach().bool().clone())
        #     self.score = None
        # self.dummy_weight = None
        self.kb_weight = None
        self.masked_kb_weight = None
        
    def update_scale(self):
        with torch.no_grad():
            i = len(self.weight)-1
            if self.weight[i].numel() > 1:
                w_std = self.weight[i].std(dim=self.dim_in, unbiased=False)
                # w_std = self.weight[i].std(unbiased=False)
                self.register_buffer(f'weight_scale_{i}', w_std.view(self.view_in))
            else:
                self.register_buffer(f'weight_scale_{i}', torch.ones(1).to(device).view(self.view_in))

            if self.fwt_weight[i].numel() > 1:
                # w_std = self.fwt_weight[i].std(unbiased=False)
                w_std = self.fwt_weight[i].std(dim=self.dim_in, unbiased=False)
                self.register_buffer(f'fwt_weight_scale_{i}', w_std.view(self.view_in))
            else:
                self.register_buffer(f'fwt_weight_scale_{i}', torch.ones(1).to(device).view(self.view_in))

            if self.bwt_weight[i].numel() > 1:
                # w_std = self.bwt_weight[i].std(unbiased=False)
                w_std = self.bwt_weight[i].std(dim=self.dim_in, unbiased=False)
                self.register_buffer(f'bwt_weight_scale_{i}', w_std.view(self.view_in))
            else:
                self.register_buffer(f'bwt_weight_scale_{i}', torch.ones(1).to(device).view(self.view_in))

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
        self.strength = (self.shape_in[-1] * self.num_out[-1] * self.ks)

    def squeeze(self, optim_state, mask_in=None, mask_out=None):
        prune_out = mask_out is not None and mask_out.sum() != self.num_out[-1]
        prune_in = mask_in is not None and mask_in.sum() != self.num_in[-1]
        if prune_out:
            apply_mask_out(self.weight[-1], mask_out, optim_state)
            apply_mask_out(self.fwt_weight[-1], mask_out, optim_state)

            self.num_out[-1] = self.weight[-1].shape[0]
            self.shape_out[-1] = sum(self.num_out)

            mask = torch.ones(self.shape_out[-2], dtype=bool, device=device)
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
        self.norm_type = norm_type
        self.ets_norm_layers = nn.ModuleList([])
        self.kbts_norm_layers = nn.ModuleList([])
        if act == 'relu':
            self.activation = nn.LeakyReLU(args.negative_slope)
            self.gain = torch.nn.init.calculate_gain('leaky_relu', args.negative_slope) ** 2
        else:
            self.activation = nn.Identity()
            self.gain = 1
        self.register_buffer('task', torch.tensor(-1, dtype=torch.int).to(device))
        self.mask_out = None
        self.last = False

    def ets_forward(self, inputs, t):
        out = 0
        for x, layer in zip(inputs, self.layers):
            out = out + layer.ets_forward(x, t)
            
        out = self.activation(self.ets_norm_layers[t](out))
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
            # self.ets_norm_layers.append(DynamicNorm(layer.shape_out[-1] + add_out, affine=affine, track_running_stats=track_running_stats))
            # self.kbts_norm_layers.append(DynamicNorm(layer.shape_out[-1] + add_out_kbts, affine=affine, track_running_stats=track_running_stats))
            self.kbts_norm_layers.append(nn.BatchNorm2d(layer.shape_out[-1] + add_out_kbts, affine=affine, track_running_stats=track_running_stats).to(device))
            self.ets_norm_layers.append(nn.BatchNorm2d(layer.shape_out[-1] + add_out, affine=affine, track_running_stats=track_running_stats).to(device))

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
            mask = torch.ones(self.layers[0].shape_out[-2], dtype=bool, device=device)
            mask = torch.cat([mask, self.mask_out])
            if self.ets_norm_layers[-1].affine:
                apply_mask_out(self.ets_norm_layers[-1].weight, mask, optim_state)
                apply_mask_out(self.ets_norm_layers[-1].bias, mask, optim_state)
                self.ets_norm_layers[-1].num_features = self.ets_norm_layers[-1].weight.shape[0]
            
            if self.ets_norm_layers[-1].track_running_stats:
                self.ets_norm_layers[-1].running_mean = self.ets_norm_layers[-1].running_mean[mask]
                self.ets_norm_layers[-1].running_var = self.ets_norm_layers[-1].running_var[mask]
                self.ets_norm_layers[-1].num_features = self.ets_norm_layers[-1].running_mean.shape[0]

    def proximal_gradient_descent(self, lr=0, lamb=0, total_strength=1):
        strength = self.strength / total_strength
        norm = 0
        for layer in self.layers:
            norm += (torch.cat([layer.fwt_weight[-1], layer.weight[-1]], dim=1) ** 2).sum(layer.dim_in)
        norm = norm.sqrt()
        aux = 1 - lamb * lr * strength / norm
        aux = F.threshold(aux, 0, 0, False)
        self.mask_out = (aux > 0).clone().detach()
        if self.norm_type is not None:
            std_new = (self.ets_norm_layers[-1].running_var[layer.shape_out[-2]:] * (aux ** 2)).sum().sqrt()
            std_old = self.ets_norm_layers[-1].running_var[layer.shape_out[-2]:].sum().sqrt()
            aux = aux * std_old / std_new
        for layer in self.layers:
            layer.weight[-1].data *= aux.view(layer.view_in)
            layer.fwt_weight[-1].data *= aux.view(layer.view_in)
        
        if self.norm_type is not None:
            norm_layer = self.ets_norm_layers[-1]
            temp = torch.ones(layer.shape_out[-2], dtype=float, device=device)
            temp = torch.cat([temp, aux], dim=0)
            if norm_layer.affine:
                norm_layer.weight.data *= temp
                norm_layer.bias.data *= temp

    def get_optim_ets_params(self):
        params = []
        for layer in self.layers:
            params += [layer.weight[-1], layer.fwt_weight[-1], layer.bwt_weight[-1]]
        # if self.norm_type is not None and 'affine' in self.norm_type:
        #     params += [self.ets_norm_layers[-1].weight, self.ets_norm_layers[-1].bias]
        return params
    
    def get_optim_kbts_params(self):
        params = []
        if self.norm_type is not None and 'affine' in self.norm_type:
            params += [self.kbts_norm_layers[-1].weight, self.kbts_norm_layers[-1].bias]
        return params, [layer.score for layer in self.layers]
    
    def freeze(self):
        for layer in self.layers:
            layer.weight[-1].requires_grad = False
            layer.fwt_weight[-1].requires_grad = False
            layer.bwt_weight[-1].requires_grad = False
        if self.norm_type is not None:
            if 'affine' in self.norm_type:
                self.ets_norm_layers[-1].weight.requires_grad = False
                self.kbts_norm_layers[-1].weight.requires_grad = False
                self.ets_norm_layers[-1].bias.requires_grad = False
                self.kbts_norm_layers[-1].bias.requires_grad = False
            self.ets_norm_layers[-1].track_running_stats = False
            self.kbts_norm_layers[-1].track_running_stats = False

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

        self.num_out = torch.cat([self.num_out, torch.IntTensor([add_out]).to(device)])
        self.num_in = torch.cat([self.num_in, torch.IntTensor([add_in]).to(device)])
        self.shape_out = torch.cat([self.shape_out, torch.IntTensor([self.shape_out[-1] + add_out]).to(device)])
        self.shape_in = torch.cat([self.shape_in, torch.IntTensor([self.shape_in[-1] + add_in]).to(device)])

        # self.num_out.append(add_out)
        # self.num_in.append(add_in)
        # self.shape_out.append(fan_out)
        # self.shape_in.append(fan_in)

        bound_std = self.gain / math.sqrt(self.shape_in[-1])
        # bound_std = self.gain / math.sqrt(self.num_out[-1])
        self.weight_ets.append(nn.Parameter(torch.Tensor(self.num_out[-1], self.shape_in[-1]).normal_(0, bound_std).to(device)))

        bound_std = self.gain / math.sqrt(fan_in_kbts)
        # bound_std = self.gain / math.sqrt(self.num_out[-1])
        self.weight_kbts.append(nn.Parameter(torch.Tensor(self.num_out[-1], fan_in_kbts).normal_(0, bound_std).to(device)))

        if self.use_bias:
            self.bias_ets.append(nn.Parameter(torch.zeros(self.num_out[-1]).to(device))) 
            self.bias_kbts.append(nn.Parameter(torch.zeros(self.num_out[-1]).to(device)))
        
    
    def freeze(self):
        self.weight_ets[-1].requires_grad = False
        self.weight_kbts[-1].requires_grad = False
        if self.use_bias:
            self.bias_ets[-1].requires_grad = False
            self.bias_kbts[-1].requires_grad = False
        

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
            
            mask = torch.ones(self.shape_in[-2], dtype=bool, device=device)
            mask = torch.cat([mask, mask_in])
            apply_mask_in(self.weight_ets[-1], mask, optim_state)
            self.shape_in[-1] = self.weight_ets[-1].shape[1]

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
            self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float).to(device))
            self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float).to(device))

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


# class DynamicNorm(nn.Module):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1,
#                  affine=False, track_running_stats=True, norm_type=None):
#         super(DynamicNorm, self).__init__()
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats

#         self.base_num_features = num_features
#         self.norm_type = norm_type
#         if 'affine' in norm_type:
#             self.affine = True
#         else:
#             self.affine = False

#         if 'track' in norm_type:
#             self.track_running_stats = True
#         else:
#             self.track_running_stats = False

#         if self.affine:
#             self.weight = nn.ParameterList([])
#             self.bias = nn.ParameterList([])
#         else:
#             self.weight = None
#             self.bias = None

#         self.register_buffer('shape_out', torch.IntTensor([0]).to(device))


#     def expand(self, add_num=None):
#         if add_num is None:
#             add_num = self.base_num_features

#         self.shape_out = torch.cat([self.shape_out, torch.IntTensor([add_num]).to(device)])
#         if self.affine:
#             self.weight.append(nn.Parameter(torch.ones(add_num).to(device)))
#             self.bias.append(nn.Parameter(torch.zeros(add_num).to(device)))

#         if self.track_running_stats:
#             self.register_buffer(f'running_mean_{self.shape_out.shape[0]-2}', torch.zeros(add_num).to(device))
#             self.register_buffer(f'running_var_{self.shape_out.shape[0]-2}', torch.ones(add_num).to(device))
#             self.num_batches_tracked = 0
#         else:
#             self.register_buffer(f'running_mean_{self.shape_out.shape[0]-2}', None)
#             self.register_buffer(f'running_var_{self.shape_out.shape[0]-2}', None)
#             self.num_batches_tracked = None
    
#     def freeze(self):
#         if self.affine:
#             self.weight[-1].requires_grad = False
#             self.bias[-1].requires_grad = False
    
#     def squeeze(self, mask, optim_state):
#         if self.weight is not None:
#             apply_mask_out(self.weight[-1], mask, optim_state)
#             apply_mask_out(self.bias[-1], mask, optim_state)
#             self.shape_out[-1] = self.weight[-1].shape[0]

#         running_mean = getattr(self, f'running_mean_{self.shape_out.shape[0]-2}')
#         running_var = getattr(self, f'running_var_{self.shape_out.shape[0]-2}')
#         if running_mean is not None:
#             running_mean = running_mean[mask]
#             running_var = running_var[mask]
#             self.register_buffer(f'running_mean_{self.shape_out.shape[0]-2}', running_mean)
#             self.register_buffer(f'running_var_{self.shape_out.shape[0]-2}', running_var)
#             self.shape_out[-1] = running_mean.shape[0]
    
#     def proximal_gradient_descent(self, aux_, lr, lamb, strength):
#         t = self.shape_out.shape[0]-2
#         if self.track_running_stats:
#             running_mean = getattr(self, f'running_mean_{t}')
#             running_var = getattr(self, f'running_var_{t}')
#             running_mean[self.shape_out[t]:] *= aux_
#             running_var[self.shape_out[t]:] *= aux_
#             self.register_buffer(f'running_mean_{t}', running_mean)
#             self.register_buffer(f'running_var_{t}', running_var)

#         if self.affine:
#             # norm = (self.weight[t][self.shape_out[t]:]**2 + self.bias[t][self.shape_out[t]:]**2) ** 0.5
#             # aux = 1 - lamb * lr * strength / norm
#             # aux = F.threshold(aux, 0, 0, False)
#             # mask_out = (aux > 0)
#             self.weight[t].data[self.shape_out[t]:] *= aux_
#             self.bias[t].data[self.shape_out[t]:] *= aux_
#         else:
#             mask_out = True
#         # return mask_out
#         return True
    
#     def count_params(self, t):
#         count = 0
#         for i in range(t+1):
#             if self.affine:
#                 count += self.weight[i].numel() + self.bias[i].numel()
#         return count

#     def batch_norm(self, input, t):
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum

#         if self.training and self.track_running_stats:
#             # TODO: if statement only here to tell the jit to skip emitting this when it is None
#             if self.num_batches_tracked is not None:  # type: ignore[has-type]
#                 self.num_batches_tracked += 1  # type: ignore[has-type]
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum

#         r"""
#         Decide whether the mini-batch stats should be used for normalization rather than the buffers.
#         Mini-batch stats are used in training mode, and in eval mode when buffers are None.
#         """
#         running_mean = getattr(self, f'running_mean_{t}')
#         running_var = getattr(self, f'running_var_{t}')

#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (running_mean is None) and (running_var is None)

#         if len(input.shape) == 4:
#             mean = input.mean([0, 2, 3])
#             # var = input.var([0, 2, 3], unbiased=False)
#             shape = (1, -1, 1, 1)
#             var = ((input - mean.view(shape)) ** 2).mean([0, 2, 3])
#             var = var
#         else:
#             mean = input.mean([0])
#             # var = input.var([0], unbiased=False)
#             shape = (1, -1)
#             var = ((input - mean.view(shape)) ** 2).mean([0])

#         # calculate running estimates
#         if bn_training:
#             if self.track_running_stats:
#                 n = input.numel() / input.size(1)
#                 with torch.no_grad():
#                     running_mean.copy_(exponential_average_factor * mean + (1 - exponential_average_factor) * running_mean)
#                     # update running_var with unbiased var
#                     running_var.copy_(exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * running_var)
#         else:
#             if self.track_running_stats:
#                 mean = running_mean
#                 var = running_var

#         return (input - mean.view(shape)) / (torch.sqrt(var.view(shape) + self.eps))


#     def layer_norm(self, input):
#         if len(input.shape) == 4:
#             mean = input.mean([1, 2, 3])
#             var = input.var([1, 2, 3], unbiased=False)
#             shape = (-1, 1, 1, 1)
#         else:
#             mean = input.mean([1])
#             var = input.var([1], unbiased=False)
#             shape = (-1, 1)

#         return (input - mean.view(shape)) / (torch.sqrt(var.view(shape) + self.eps))

#     def L2_norm(self, input):
#         if len(input.shape) == 4:
#             norm = input.norm(2, dim=(1,2,3)).view(-1,1,1,1)
#         else:
#             norm = input.norm(2, dim=(1)).view(-1,1)

#         return input / norm

#     def forward(self, input, t=-1):
#         if 'bn' in self.norm_type:
#             output = self.batch_norm(input, t)

#         if self.affine:
#             weight = self.weight[t]
#             bias = self.bias[t]
#             if len(input.shape) == 4:
#                 output = output * weight.view(1,-1,1,1) + bias.view(1,-1,1,1)
#             else:
#                 output = output * weight.view(1,-1) + bias.view(1,-1)

#         return output

    
            
