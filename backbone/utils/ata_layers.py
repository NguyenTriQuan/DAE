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

        self.activation = nn.LeakyReLU(args.negative_slope)
        self.gain = torch.nn.init.calculate_gain('leaky_relu', args.negative_slope)

        self.mask_in = None
        self.mask_out = None

        self.register_buffer('bias', None)
        self.register_buffer('shape_out', torch.IntTensor([0]).to(device))
        self.register_buffer('shape_in', torch.IntTensor([0]).to(device))
        self.register_buffer('num_out', torch.IntTensor([]).to(device))
        self.register_buffer('num_in', torch.IntTensor([]).to(device))
        self.register_buffer('task', torch.tensor(-1, dtype=torch.int).to(device))
        self.kbts_sparsities = []
        self.jr_sparsity = 0

        self.kb_weight = torch.empty(0).to(device)
        self.masked_kb_weight = torch.empty(0).to(device)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        self.over_mul = 1 # prevent errors
        self.dummy_weight = torch.Tensor(self.base_out_features * self.base_in_features * self.ks * self.over_mul).to(device)
        nn.init.normal_(self.dummy_weight, 0, 1)
        self.last = False

    def forward(self, x, t, mode):    
        if 'ets' == mode:
            weight, bias = self.get_ets_params(t)
        elif 'kbts' == mode:
            weight, bias = self.get_kbts_params(t)
        elif 'jr' == mode:
            weight, bias = self.get_jr_params()
    
        if isinstance(self, DynamicConv2D):
            output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.linear(x, weight, bias)
        return output
    
    def get_expand_shape(self, t, add_in=None, add_out=None, fix=False):
        # expand from knowledge base weights of task t
        if add_in is None:
            if 'fix' in self.args.ablation:
                add_in = self.base_in_features - self.shape_in[t]
            else:
                add_in = self.base_in_features
        fan_in = self.shape_in[t] + add_in
        if add_out is None:
            # compute add_out
            if 'fix' in self.args.ablation:
                add_out = self.base_out_features - self.shape_out[t]
            elif 'op' in self.args.ablation:
                add_out = self.base_out_features
            else:
                total_params = self.shape_out[t] * self.shape_in[t] * self.ks + (self.dummy_weight.numel() / self.over_mul)
                fan_out = total_params // (fan_in * self.ks)
                add_out = max(fan_out - self.shape_out[t], 0)

        fan_out = self.shape_out[t] + add_out
        return int(fan_in), int(fan_out), int(add_in), int(add_out)

    def expand(self, add_in=None, add_out=None):
        self.task += 1
        fan_in, fan_out, add_in, add_out = self.get_expand_shape(-1, add_in, add_out)
        self.num_out = torch.cat([self.num_out, torch.IntTensor([add_out]).to(device)])
        self.num_in = torch.cat([self.num_in, torch.IntTensor([add_in]).to(device)])

        self.shape_out = torch.cat([self.shape_out, torch.IntTensor([fan_out]).to(device)])
        self.shape_in = torch.cat([self.shape_in, torch.IntTensor([fan_in]).to(device)])
        
        bound_std = self.gain / math.sqrt(fan_in * self.ks)
        if isinstance(self, DynamicConv2D):
            for i in range(self.task):
                self.register_buffer(f'weight_{i}_{self.task}', 
                    nn.Parameter(torch.Tensor(self.num_out[self.task], self.num_in[i] // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))
                self.register_buffer(f'weight_{self.task}_{i}', 
                    nn.Parameter(torch.Tensor(self.num_out[i], self.num_in[self.task] // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))
            self.register_buffer(f'weight_{self.task}_{self.task}', 
                nn.Parameter(torch.Tensor(self.num_out[self.task], self.num_in[self.task] // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))

            self.score = nn.Parameter(torch.Tensor(fan_out, fan_in // self.groups, *self.kernel_size).to(device))
        else:
            for i in range(self.task):
                self.register_buffer(f'weight_{i}_{self.task}', 
                    nn.Parameter(torch.Tensor(self.num_out[self.task], self.num_in[i]).normal_(0, bound_std).to(device)))
                self.register_buffer(f'weight_{self.task}_{i}', 
                    nn.Parameter(torch.Tensor(self.num_out[i], self.num_in[self.task]).normal_(0, bound_std).to(device)))
            self.register_buffer(f'weight_{self.task}_{self.task}', 
                nn.Parameter(torch.Tensor(self.num_out[self.task], self.num_in[self.task]).normal_(0, bound_std).to(device)))

            self.score = nn.Parameter(torch.Tensor(fan_out, fan_in).to(device))

        nn.init.kaiming_uniform_(self.score, a=math.sqrt(5))
        # self.register_buffer(f'kbts_mask_{self.task}', torch.ones_like(self.score).to(device))

        self.set_reg_strength()
        return add_out * self.s * self.s

    def get_kb_params(self, t):
        # get knowledge base parameters for task t
        # if len(self.kb_weight.shape) > 1 and self.kb_weight.shape[0] == self.shape_out[t] and self.kb_weight.shape[1] == self.shape_in[t]:
        #     return
        
        self.kb_weight = torch.empty(0).to(device)
        self.masked_kb_weight = torch.empty(0).to(device)

        for i in range(t):
            row = torch.empty(0).to(device)
            old_std = getattr(self, f'old_var_{t}')[i].sqrt()
            for j in range(t):
                row = torch.cat([row, old_std * getattr(self, f'weight_{i}_{j}').data / getattr(self, f'scale_{i}_{j}')], dim=0)
                # row = torch.cat([row, getattr(self, f'weight_{i}_{j}')], dim=0)
            self.kb_weight = torch.cat([self.kb_weight, row], dim=1)

    
    def get_masked_kb_params(self, t, add_in, add_out=None):
        fan_in, fan_out, add_in, add_out = self.get_expand_shape(t, add_in, add_out)

        # if len(self.masked_kb_weight.shape) > 1 and self.masked_kb_weight.shape[0] == fan_out and self.masked_kb_weight.shape[1] == fan_in:
        #     return add_out * self.s * self.s
        
        self.get_kb_params(t)
        n_0 = add_out * (fan_in-add_in) * self.ks
        n_1 = fan_out * add_in * self.ks

        if isinstance(self, DynamicConv2D):
            dummy_weight_0 = self.dummy_weight[:n_0].view(add_out, (fan_in-add_in) // self.groups, *self.kernel_size)
            dummy_weight_1 = self.dummy_weight[n_0:n_0+n_1].view(fan_out, add_in // self.groups, *self.kernel_size)
        else:
            dummy_weight_0 = self.dummy_weight[:n_0].view(add_out, (fan_in-add_in))
            dummy_weight_1 = self.dummy_weight[n_0:n_0+n_1].view(fan_out, add_in)
        self.masked_kb_weight = torch.cat([torch.cat([self.kb_weight, dummy_weight_0], dim=0), dummy_weight_1], dim=1)
        
        bound_std = self.gain / math.sqrt(fan_in * self.ks)
        self.masked_kb_weight = self.masked_kb_weight * bound_std
        return add_out * self.s * self.s

    def get_ets_params(self, t):
        # get expanded task specific model
        if self.last:
            fwt_weight = torch.empty(0).to(device)
            for i in range(t):
                fwt_weight = torch.cat([fwt_weight, getattr(self, f'weight_{i}_{t}')], dim=1)
            weight = torch.cat([fwt_weight, getattr(self, f'weight_{t}_{t}')], dim=1)
        else:
            weight = self.kb_weight
            weight = F.dropout(weight, self.dropout, self.training)
            fwt_weight = torch.empty(0).to(device)
            bwt_weight = torch.empty(0).to(device)
            for i in range(t):
                fwt_weight = torch.cat([fwt_weight, getattr(self, f'weight_{i}_{t}')], dim=1)
                bwt_weight = torch.cat([bwt_weight, getattr(self, f'weight_{t}_{i}')], dim=0)
            weight = torch.cat([torch.cat([weight, bwt_weight], dim=1), 
                                torch.cat([fwt_weight, getattr(self, f'weight_{t}_{t}')], dim=1)], dim=0)
        self.sh = weight.norm(2).item()
        return weight, None
    
    def get_kbts_params(self, t):
        if self.training:
            mask = GetSubnet.apply(self.score.abs(), 1-self.kbts_sparsities[t])
            weight = self.masked_kb_weight * mask / (1-self.kbts_sparsities[t])
            self.register_buffer(f'kbts_mask_{t}', mask.detach().bool().clone())
        else:
            mask = getattr(self, f'kbts_mask_{t}')
            weight = self.masked_kb_weight * mask / (1-self.kbts_sparsities[t])
        
        return weight, None
    
    def get_jr_params(self):
        if self.training:
            mask = GetSubnet.apply(self.score.abs(), 1-self.jr_sparsity)
            weight = self.masked_kb_weight * mask / (1-self.jr_sparsity)
            self.register_buffer('jr_mask', mask.detach().bool().clone())
        else:
            mask = getattr(self, 'jr_mask')
            weight = self.masked_kb_weight * mask / (1-self.jr_sparsity)
        
        return weight, None

    def freeze(self):
        for i in range(self.task):
            for j in range(self.task):
                getattr(self, f'weight_{i}_{j}').requires_grad = False

    def get_optim_params(self):
        params = [self.score, getattr(self, f'weight_{self.task}_{self.task}')]
        for i in range(self.task):
            params += [getattr(self, f'weight_{i}_{self.task}'), getattr(self, f'weight_{self.task}_{i}')]
        return params

    def clear_memory(self):
        self.score = None

    def count_params(self, t):
        count = 0
        for i in range(t+1):
            for j in range(t+1):
                count += getattr(self, f'weight_{i}_{j}').numel()
        return count
    
    def set_reg_strength(self):
        self.strength = 1 - ((self.shape_in[-1] + self.shape_out[-1] + self.kernel_size[0] + self.kernel_size[1]) / 
                                (self.shape_in[-1] * self.shape_out[-1] * self.kernel_size[0] * self.kernel_size[1])) 

    def squeeze(self, optim_state, mask_in=None, mask_out=None):
        prune_out = mask_out is not None and mask_out.sum() != self.num_out[-1]
        prune_in = mask_in is not None and mask_in.sum() != self.num_in[-1]
        if prune_out:
            apply_mask_out(getattr(self, f'weight_{self.task}_{self.task}'), mask_out, optim_state)
            for i in range(self.task):
                apply_mask_out(getattr(self, f'weight_{i}_{self.task}'), mask_out, optim_state)

            self.num_out[-1] = getattr(self, f'weight_{self.task}_{self.task}').shape[0]
            self.shape_out[-1] = self.num_out.sum()
        
        if prune_in:
            if self.s != 1:
                mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0), self.s, self.s).contiguous().view(-1)
            apply_mask_in(getattr(self, f'weight_{self.task}_{self.task}'), mask_in, optim_state)
            for i in range(self.task):
                apply_mask_in(getattr(self, f'weight_{self.task}_{i}'), mask_in, optim_state)

            self.num_in[-1] = getattr(self, f'weight_{self.task}_{self.task}').shape[1]
            self.shape_in[-1] = self.num_in.sum()

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
        self.name = 'linear'
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
        self.name = 'conv'
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
        self.name = 'conv2d'

        super(DynamicConv2D, self).__init__(in_features, out_features, kernel_size, 
                                            stride, padding, dilation, False, _pair(0), groups, bias, norm_type, args, s)


class DynamicBlock(nn.Module):
    # A block of dynamic layers, normalization, and activation. All layers are share the number of out features.
    def __init__(self, layers, norm_type=None, args=None, act='relu'):
        super(DynamicBlock, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.args = args
        self.norm_type = norm_type
        self.norm_layers = nn.ModuleList([])
        if act == 'relu':
            self.activation = nn.LeakyReLU(args.negative_slope)
            self.gain = torch.nn.init.calculate_gain('leaky_relu', args.negative_slope) ** 2
        else:
            self.activation = nn.Identity()
            self.gain = 1
        self.register_buffer('task', torch.tensor(-1, dtype=torch.int).to(device))
        self.mask_out = None
        self.last = False

    def forward(self, inputs, t, mode):
        out = 0
        for x, layer in zip(inputs, self.layers):
            out += layer(x, t, mode)

        out = self.activation(self.norm_layers[t](out))
        # out = self.activation(out)
        return out
    
    def expand(self, add_ins, add_outs):
        self.task += 1
        add_outs_ = []
        add_ins_ = []
        for add_in, add_out, layer in zip(add_ins, add_outs, self.layers):
            _, _, add_in_, add_out_ = layer.get_expand_shape(-1, add_in, add_out)
            add_outs_.append(add_out_)
            add_ins_.append(add_in_)

        add_out = max(add_outs_)
        for add_in, layer in zip(add_ins_, self.layers):
            layer.expand(add_in, add_out)

        self.strength = max([layer.strength for layer in self.layers])

        if self.norm_type is None:
            self.norm_layers.append(nn.Identity())
        else:
            if 'affine' in self.norm_type:
                affine = True
            else:
                affine = False
            if 'track' in self.norm_type:
                track_running_stats = True
            else:
                track_running_stats = False
            self.norm_layers.append(DynamicNorm(layer.shape_out[-1], affine=affine, track_running_stats=track_running_stats))

        return add_out

    def initialize(self):
        def compute_scale(layer, i, j):
            if getattr(layer, f'weight_{i}_{j}').numel() == 0:
                return torch.tensor(1).to(device)
            else:
                var = (getattr(layer, f'weight_{i}_{j}').data ** 2).mean(layer.dim_in).detach()
                # print(var)
                return var.sqrt().view(layer.view_in)
        # Initialize new weights and rescale old weights to have the same variance:
        sum_var = 0
        for layer in self.layers:
            sum_var += layer.ks * layer.shape_out[-1]
        
        std = math.sqrt(self.gain / sum_var)
        # initial equal var for old neurons
        self.register_buffer(f'old_var_{self.task}', (std ** 2) * torch.ones(self.task).to(device))

        for layer in self.layers:
            # rescale old weights
            layer.register_buffer(f'old_var_{self.task}', getattr(self, f'old_var_{self.task}').data)
            if self.task > 0:
                for i in range(self.task-1):
                    layer.register_buffer(f'scale_{i}_{self.task-1}', compute_scale(layer, i, self.task-1))
                    layer.register_buffer(f'scale_{self.task-1}_{i}', compute_scale(layer, self.task-1, i))
                layer.register_buffer(f'scale_{self.task-1}_{self.task-1}', compute_scale(layer, self.task-1, self.task-1))

            # initialize new weights
            for i in range(self.task):
                nn.init.normal_(getattr(layer, f'weight_{i}_{self.task}'), 0, std)
                nn.init.normal_(getattr(layer, f'weight_{self.task}_{i}'), 0, std)
            nn.init.normal_(getattr(layer, f'weight_{self.task}_{self.task}'), 0, std)

        # self.check_var()
        self.normalize()
        self.check_var()

    def squeeze(self, optim_state, mask_ins):
        for mask_in, layer in zip(mask_ins, self.layers):
            layer.squeeze(optim_state, mask_in, self.mask_out)
        self.strength = max([layer.strength for layer in self.layers])
        if self.norm_type is not None and self.mask_out is not None:
            mask = torch.ones(self.layers[0].shape_out[-2], dtype=bool, device=device)
            mask = torch.cat([mask, self.mask_out])
            self.norm_layers[-1].squeeze(mask, optim_state)
        self.normalize()

    def normalize(self):
        def layer_wise(layer, i, j):
            if getattr(layer, f'weight_{i}_{j}').numel() == 0:
                return torch.zeros(getattr(layer, f'weight_{i}_{j}').shape[0]).to(device)
            mean = getattr(layer, f'weight_{i}_{j}').data.mean(layer.dim_in)
            getattr(layer, f'weight_{i}_{j}').data -= mean.view(layer.view_in)
            var = (getattr(layer, f'weight_{i}_{j}').data ** 2).mean(layer.dim_in)
            return var
        
        var_new_neurons = []
        var_layers_in = 0
        for i in range(self.task):
            var_layers_out = 0
            for layer in self.layers:
                var_layers_out += layer.ks * layer_wise(layer, i, self.task)
                var_layers_in += layer.ks * layer_wise(layer, self.task, i).sum()
            var_new_neurons.append(var_layers_out)

        var_layers_out = 0
        for layer in self.layers:
            var = layer.ks * layer_wise(layer, self.task, self.task)
            var_layers_out += var

        var_new_neurons.append(var_layers_out)
        var_new_neurons = torch.stack(var_new_neurons, dim=0)
        var_new_neurons /= self.gain # shape (num task, num new neurons)
        
        const = sum([layer.ks * layer.shape_out[-2] for layer in self.layers]) / self.gain
        var_tasks = var_new_neurons.sum(1) # shape (num task)
        var_tasks[self.task] += (var_layers_in / self.gain)
        if self.task > 0:
            var_tasks[:self.task] += const * getattr(self, f'old_var_{self.task}')
            getattr(self, f'old_var_{self.task}').data /= var_tasks[:self.task]

        std_new_neurons = var_tasks.sqrt()
        std_old_neurons = var_tasks[self.task].sqrt()
        for layer in self.layers:
            layer.register_buffer(f'old_var_{self.task}', getattr(self, f'old_var_{self.task}').data)
            for i in range(self.task):
                getattr(layer, f'weight_{i}_{self.task}').data /= std_new_neurons[i].view(layer.view_in)
                getattr(layer, f'weight_{self.task}_{i}').data /= std_old_neurons
            getattr(layer, f'weight_{self.task}_{self.task}').data /= std_new_neurons[self.task].view(layer.view_in)

        # if self.norm_type is not None and 'scale' not in self.args.ablation:
        #     out_scale = (std_new_neurons**2).sum(0).sqrt() # shape (num new neurons)
        #     if self.norm_layers[-1].track_running_stats:
        #         self.norm_layers[-1].running_mean[layer.shape_out[-2]:] /= out_scale
        #         self.norm_layers[-1].running_var[layer.shape_out[-2]:] /= (out_scale ** 2)

        #     if self.norm_layers[-1].affine:
        #         self.norm_layers[-1].weight.data[layer.shape_out[-2]:] /= out_scale
        #         self.norm_layers[-1].bias.data[layer.shape_out[-2]:] /= out_scale
            

    def proximal_gradient_descent(self, lr=0, lamb=0, total_strength=1):
        def layer_wise(layer, i, j):
            if getattr(layer, f'weight_{i}_{j}').numel() == 0:
                return torch.zeros(getattr(layer, f'weight_{i}_{j}').shape[0]).to(device)
            mean = getattr(layer, f'weight_{i}_{j}').data.mean(layer.dim_in)
            getattr(layer, f'weight_{i}_{j}').data -= mean.view(layer.view_in)
            var = (getattr(layer, f'weight_{i}_{j}').data ** 2).mean(layer.dim_in)
            return var
        
        var_new_neurons = []
        var_layers_in = 0
        for i in range(self.task):
            var_layers_out = 0
            for layer in self.layers:
                var_layers_out += layer.ks * layer_wise(layer, i, self.task)
                var_layers_in += layer.ks * layer_wise(layer, self.task, i).sum()
            var_new_neurons.append(var_layers_out)

        var_layers_out = 0
        for layer in self.layers:
            var = layer.ks * layer_wise(layer, self.task, self.task)
            var_layers_out += var

        var_new_neurons.append(var_layers_out)
        var_new_neurons = torch.stack(var_new_neurons, dim=0)
        var_new_neurons /= self.gain # shape (num task, num new neurons)
        strength = self.strength / total_strength
        aux = 1 - lamb * lr * strength / var_new_neurons.sum(0).sqrt()
        aux = F.threshold(aux, 0, 0, False) # shape (num new neurons)
        self.mask_out = (aux > 0).clone().detach() # shape (num new neurons)
        
        const = sum([layer.ks * layer.shape_out[-2] for layer in self.layers]) / self.gain
        var_tasks = (var_new_neurons * (aux**2).view(1, -1)).sum(1) # shape (num task)
        var_tasks[self.task] += (var_layers_in / self.gain)
        if self.task > 0:
            var_tasks[:self.task] += const * getattr(self, f'old_var_{self.task}')
            getattr(self, f'old_var_{self.task}').data /= var_tasks[:self.task]

        std_new_neurons = var_tasks.sqrt().view(-1, 1) / aux.view(1, -1) # shape (num task, num new neurons)
        std_old_neurons = var_tasks[self.task].sqrt() # shape (0)
        for layer in self.layers:
            layer.register_buffer(f'old_var_{self.task}', getattr(self, f'old_var_{self.task}').data)
            for i in range(self.task):
                getattr(layer, f'weight_{i}_{self.task}').data /= std_new_neurons[i].view(layer.view_in)
                getattr(layer, f'weight_{self.task}_{i}').data /= std_old_neurons
            getattr(layer, f'weight_{self.task}_{self.task}').data /= std_new_neurons[self.task].view(layer.view_in)

        # if self.norm_type is not None and 'scale' not in self.args.ablation:
        #     out_scale = (std_new_neurons**2).sum(0).sqrt() # shape (num new neurons)
        #     if self.norm_layers[-1].track_running_stats:
        #         self.norm_layers[-1].running_mean[layer.shape_out[-2]:] /= out_scale
        #         self.norm_layers[-1].running_var[layer.shape_out[-2]:] /= (out_scale ** 2)

        #     if self.norm_layers[-1].affine:
        #         self.norm_layers[-1].weight.data[layer.shape_out[-2]:] /= out_scale
        #         self.norm_layers[-1].bias.data[layer.shape_out[-2]:] /= out_scale
        
        # if self.task > 0:
        #     self.check_var()
        
    def check_var(self):
        def layer_wise(layer, i, j):
            w = getattr(layer, f'weight_{i}_{j}')
            if w.numel() == 0:
                return torch.zeros(w.shape[0]).to(device)
            
            if hasattr(layer, f'scale_{i}_{j}'):
                w = w * getattr(self, f'old_var_{self.task}')[i].sqrt() / getattr(layer, f'scale_{i}_{j}')
            var = (w.data ** 2).mean(layer.dim_in)

            return var
        
        for layer in self.layers:
            print(layer.name, end=' ')
        print()
        if hasattr(self, f'old_var_{self.task}'):
            print(getattr(self, f'old_var_{self.task}'))

        if self.last:
            var_layers_in = 0
            for j in range(self.task+1):
                for layer in self.layers:
                    var_layers_in += layer.ks * layer_wise(layer, self.task, j).sum()
            print(self.task, var_layers_in.item()/self.gain)
        else:
            for i in range(self.task+1):
                var_layers_in = 0
                for j in range(self.task+1):
                    for layer in self.layers:
                        var_layers_in += layer.ks * layer_wise(layer, i, j).sum()
                print(i, var_layers_in.item()/self.gain)

class DynamicClassifier(DynamicLinear):

    def __init__(self, in_features, out_features, bias=False, norm_type=None, args=None, s=1):
        super(DynamicClassifier, self).__init__()

    def forward(self, x, t, mode):    
        if 'ets' == mode:
            weight, bias = self.get_ets_params(t)
        elif 'kbts' == mode:
            weight, bias = self.get_kbts_params(t)
        elif 'jr' == mode:
            weight, bias = self.get_jr_params()
    
        if isinstance(self, DynamicConv2D):
            output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.linear(x, weight, bias)
        return output
    
    def get_expand_shape(self, t, add_in=None, add_out=None, fix=False):
        # expand from knowledge base weights of task t
        if add_in is None:
            if 'fix' in self.args.ablation:
                add_in = self.base_in_features - self.shape_in[t]
            else:
                add_in = self.base_in_features
        fan_in = self.shape_in[t] + add_in
        if add_out is None:
            # compute add_out
            if 'fix' in self.args.ablation:
                add_out = self.base_out_features - self.shape_out[t]
            elif 'op' in self.args.ablation:
                add_out = self.base_out_features
            else:
                total_params = self.shape_out[t] * self.shape_in[t] * self.ks + (self.dummy_weight.numel() / self.over_mul)
                fan_out = total_params // (fan_in * self.ks)
                add_out = max(fan_out - self.shape_out[t], 0)

        fan_out = self.shape_out[t] + add_out
        return int(fan_in), int(fan_out), int(add_in), int(add_out)

    def expand(self, add_in=None, add_out=None):
        self.task += 1
        fan_in, fan_out, add_in, add_out = self.get_expand_shape(-1, add_in, add_out)
        self.num_out = torch.cat([self.num_out, torch.IntTensor([add_out]).to(device)])
        self.num_in = torch.cat([self.num_in, torch.IntTensor([add_in]).to(device)])

        self.shape_out = torch.cat([self.shape_out, torch.IntTensor([fan_out]).to(device)])
        self.shape_in = torch.cat([self.shape_in, torch.IntTensor([fan_in]).to(device)])
        
        bound_std = self.gain / math.sqrt(fan_in * self.ks)
        if isinstance(self, DynamicConv2D):
            for i in range(self.task):
                self.register_buffer(f'weight_{i}_{self.task}', 
                    nn.Parameter(torch.Tensor(self.num_out[self.task], self.num_in[i] // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))
                self.register_buffer(f'weight_{self.task}_{i}', 
                    nn.Parameter(torch.Tensor(self.num_out[i], self.num_in[self.task] // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))
            self.register_buffer(f'weight_{self.task}_{self.task}', 
                nn.Parameter(torch.Tensor(self.num_out[self.task], self.num_in[self.task] // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))

            self.score = nn.Parameter(torch.Tensor(fan_out, fan_in // self.groups, *self.kernel_size).to(device))
        else:
            for i in range(self.task):
                self.register_buffer(f'weight_{i}_{self.task}', 
                    nn.Parameter(torch.Tensor(self.num_out[self.task], self.num_in[i]).normal_(0, bound_std).to(device)))
                self.register_buffer(f'weight_{self.task}_{i}', 
                    nn.Parameter(torch.Tensor(self.num_out[i], self.num_in[self.task]).normal_(0, bound_std).to(device)))
            self.register_buffer(f'weight_{self.task}_{self.task}', 
                nn.Parameter(torch.Tensor(self.num_out[self.task], self.num_in[self.task]).normal_(0, bound_std).to(device)))

            self.score = nn.Parameter(torch.Tensor(fan_out, fan_in).to(device))

        nn.init.kaiming_uniform_(self.score, a=math.sqrt(5))
        # self.register_buffer(f'kbts_mask_{self.task}', torch.ones_like(self.score).to(device))

        self.set_reg_strength()
        return add_out * self.s * self.s

    def get_kb_params(self, t):
        # get knowledge base parameters for task t
        # if len(self.kb_weight.shape) > 1 and self.kb_weight.shape[0] == self.shape_out[t] and self.kb_weight.shape[1] == self.shape_in[t]:
        #     return
        
        self.kb_weight = torch.empty(0).to(device)
        self.masked_kb_weight = torch.empty(0).to(device)

        for i in range(t):
            row = torch.empty(0).to(device)
            old_std = getattr(self, f'old_var_{t}')[i].sqrt()
            for j in range(t):
                row = torch.cat([row, old_std * getattr(self, f'weight_{i}_{j}').data / getattr(self, f'scale_{i}_{j}')], dim=0)
                # row = torch.cat([row, getattr(self, f'weight_{i}_{j}')], dim=0)
            self.kb_weight = torch.cat([self.kb_weight, row], dim=1)

    
    def get_masked_kb_params(self, t, add_in, add_out=None):
        fan_in, fan_out, add_in, add_out = self.get_expand_shape(t, add_in, add_out)

        # if len(self.masked_kb_weight.shape) > 1 and self.masked_kb_weight.shape[0] == fan_out and self.masked_kb_weight.shape[1] == fan_in:
        #     return add_out * self.s * self.s
        
        self.get_kb_params(t)
        n_0 = add_out * (fan_in-add_in) * self.ks
        n_1 = fan_out * add_in * self.ks

        if isinstance(self, DynamicConv2D):
            dummy_weight_0 = self.dummy_weight[:n_0].view(add_out, (fan_in-add_in) // self.groups, *self.kernel_size)
            dummy_weight_1 = self.dummy_weight[n_0:n_0+n_1].view(fan_out, add_in // self.groups, *self.kernel_size)
        else:
            dummy_weight_0 = self.dummy_weight[:n_0].view(add_out, (fan_in-add_in))
            dummy_weight_1 = self.dummy_weight[n_0:n_0+n_1].view(fan_out, add_in)
        self.masked_kb_weight = torch.cat([torch.cat([self.kb_weight, dummy_weight_0], dim=0), dummy_weight_1], dim=1)
        
        bound_std = self.gain / math.sqrt(fan_in * self.ks)
        self.masked_kb_weight = self.masked_kb_weight * bound_std
        return add_out * self.s * self.s

    def get_ets_params(self, t):
        # get expanded task specific model
        if self.last:
            fwt_weight = torch.empty(0).to(device)
            for i in range(t):
                fwt_weight = torch.cat([fwt_weight, getattr(self, f'weight_{i}_{t}')], dim=1)
            weight = torch.cat([fwt_weight, getattr(self, f'weight_{t}_{t}')], dim=1)
        else:
            weight = self.kb_weight
            weight = F.dropout(weight, self.dropout, self.training)
            fwt_weight = torch.empty(0).to(device)
            bwt_weight = torch.empty(0).to(device)
            for i in range(t):
                fwt_weight = torch.cat([fwt_weight, getattr(self, f'weight_{i}_{t}')], dim=1)
                bwt_weight = torch.cat([bwt_weight, getattr(self, f'weight_{t}_{i}')], dim=0)
            weight = torch.cat([torch.cat([weight, bwt_weight], dim=1), 
                                torch.cat([fwt_weight, getattr(self, f'weight_{t}_{t}')], dim=1)], dim=0)
        self.sh = weight.norm(2).item()
        return weight, None
    
    def get_kbts_params(self, t):
        if self.training:
            mask = GetSubnet.apply(self.score.abs(), 1-self.kbts_sparsities[t])
            weight = self.masked_kb_weight * mask / (1-self.kbts_sparsities[t])
            self.register_buffer(f'kbts_mask_{t}', mask.detach().bool().clone())
        else:
            mask = getattr(self, f'kbts_mask_{t}')
            weight = self.masked_kb_weight * mask / (1-self.kbts_sparsities[t])
        
        return weight, None
    
    def get_jr_params(self):
        if self.training:
            mask = GetSubnet.apply(self.score.abs(), 1-self.jr_sparsity)
            weight = self.masked_kb_weight * mask / (1-self.jr_sparsity)
            self.register_buffer('jr_mask', mask.detach().bool().clone())
        else:
            mask = getattr(self, 'jr_mask')
            weight = self.masked_kb_weight * mask / (1-self.jr_sparsity)
        
        return weight, None

    def freeze(self):
        for i in range(self.task):
            for j in range(self.task):
                getattr(self, f'weight_{i}_{j}').requires_grad = False

    def get_optim_params(self):
        params = [self.score, getattr(self, f'weight_{self.task}_{self.task}')]
        for i in range(self.task):
            params += [getattr(self, f'weight_{i}_{self.task}'), getattr(self, f'weight_{self.task}_{i}')]
        return params

    def clear_memory(self):
        self.score = None

    def count_params(self, t):
        count = 0
        for i in range(t+1):
            for j in range(t+1):
                count += getattr(self, f'weight_{i}_{j}').numel()
        return count
    
    def set_reg_strength(self):
        self.strength = 1 - ((self.shape_in[-1] + self.shape_out[-1] + self.kernel_size[0] + self.kernel_size[1]) / 
                                (self.shape_in[-1] * self.shape_out[-1] * self.kernel_size[0] * self.kernel_size[1])) 

    def squeeze(self, optim_state, mask_in=None, mask_out=None):
        prune_out = mask_out is not None and mask_out.sum() != self.num_out[-1]
        prune_in = mask_in is not None and mask_in.sum() != self.num_in[-1]
        if prune_out:
            apply_mask_out(getattr(self, f'weight_{self.task}_{self.task}'), mask_out, optim_state)
            for i in range(self.task):
                apply_mask_out(getattr(self, f'weight_{i}_{self.task}'), mask_out, optim_state)

            self.num_out[-1] = getattr(self, f'weight_{self.task}_{self.task}').shape[0]
            self.shape_out[-1] = self.num_out.sum()
        
        if prune_in:
            if self.s != 1:
                mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0), self.s, self.s).contiguous().view(-1)
            apply_mask_in(getattr(self, f'weight_{self.task}_{self.task}'), mask_in, optim_state)
            for i in range(self.task):
                apply_mask_in(getattr(self, f'weight_{self.task}_{i}'), mask_in, optim_state)

            self.num_in[-1] = getattr(self, f'weight_{self.task}_{self.task}').shape[1]
            self.shape_in[-1] = self.num_in.sum()

        self.mask_out = None
        self.set_reg_strength()

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

        var = var.sum() / (2*3)
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

    
            
