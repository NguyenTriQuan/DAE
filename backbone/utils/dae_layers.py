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
        flat_out[idx[:j]] = False
        flat_out[idx[j:]] = True

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

        self.weight = nn.ParameterList([])
        self.fwt_weight = nn.ParameterList([])
        self.bwt_weight = nn.ParameterList([])

        if norm_type:
            self.norm_layer = DynamicNorm(out_features, affine=True, track_running_stats=True, norm_type=norm_type)
        else:
            self.norm_layer = None

        self.mask_in = None
        self.mask_out = None

        self.register_buffer('bias', None)
        self.register_buffer('shape_out', torch.IntTensor([0]).to(device))
        self.register_buffer('shape_in', torch.IntTensor([0]).to(device))
        self.register_buffer('num_out', torch.IntTensor([]).to(device))
        self.register_buffer('num_in', torch.IntTensor([]).to(device))

        self.kb_weight = torch.empty(0).to(device)
        self.gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        self.dummy_weight = torch.Tensor(self.base_out_features * self.base_in_features * self.ks).to(device)
        nn.init.normal_(self.dummy_weight, 0, 1)


    def expand(self, add_in=None, add_out=None):
        if add_in is None:
            if self.args.fix:
                add_in = self.base_in_features - self.shape_in[-1]
            else:
                add_in = self.base_in_features
        if add_out is None:
            if self.args.fix:
                add_out = self.base_out_features - self.shape_out[-1]
            else:
                add_out = self.base_out_features

        self.num_out = torch.cat([self.num_out, torch.IntTensor([add_out]).to(device)])
        self.num_in = torch.cat([self.num_in, torch.IntTensor([add_in]).to(device)])

        self.shape_out = torch.cat([self.shape_out, torch.IntTensor([self.shape_out[-1] + add_out]).to(device)])
        self.shape_in = torch.cat([self.shape_in, torch.IntTensor([self.shape_in[-1] + add_in]).to(device)])

        fan_out_kbts = max(self.base_out_features, self.shape_out[-2])
        fan_in_kbts = max(self.base_in_features, self.shape_in[-2])

        fan_out_jr = max(self.base_out_features, self.shape_out[-1])
        fan_in_jr = max(self.base_in_features, self.shape_in[-1])
        
        if isinstance(self, DynamicConv2D):
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in // self.groups, *self.kernel_size).normal_(0, 1).to(device)))
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.shape_in[-2] // self.groups, *self.kernel_size).normal_(0, 1).to(device)))
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.shape_out[-2], add_in // self.groups, *self.kernel_size).normal_(0, 1).to(device)))
            self.score_kbts = nn.Parameter(torch.Tensor(fan_out_kbts, fan_in_kbts // self.groups, *self.kernel_size).to(device))
            self.score_jr = nn.Parameter(torch.Tensor(fan_out_jr, fan_in_jr // self.groups, *self.kernel_size).to(device))
        else:
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in).normal_(0, 1).to(device)))
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.shape_in[-2]).normal_(0, 1).to(device)))
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.shape_out[-2], add_in).normal_(0, 1).to(device)))
            self.score_kbts = nn.Parameter(torch.Tensor(fan_out_kbts, fan_in_kbts).to(device))
            self.score_jr = nn.Parameter(torch.Tensor(fan_out_jr, fan_in_jr).to(device))

        nn.init.kaiming_uniform_(self.score_kbts, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.score_jr, a=math.sqrt(5))

        mask = GetSubnet.apply(self.score_kbts.abs(), self.sparsity)
        self.register_buffer('kbts_mask'+f'_{self.num_out.shape[0]-1}', mask.detach().bool())

        mask = GetSubnet.apply(self.score_jr.abs(), self.sparsity)
        self.register_buffer('jr_mask', mask.detach().bool())

        self.reg_strength = (
            (self.shape_in[-1] + self.num_out[-1] + self.kernel_size[0] + self.kernel_size[1]) 
            / (self.shape_in[-1] * self.num_out[-1] * self.kernel_size[0] * self.kernel_size[1])
        )

        if self.norm_layer:
            self.norm_layer.expand(add_out) 


    def forward(self, x, t, mode='ets'):    
        if x.numel() == 0:
            return torch.empty(0)
        
        if mode == 'ets':
            weight, bias = self.get_ets_params(t)
        else:
            weight, bias = self.get_masked_kb_params(t, mode)

        if weight.numel() == 0:
            return torch.empty(0)
    
        if isinstance(self, DynamicConv2D):
            output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.linear(x, weight, bias)

        if self.norm_layer is not None:
            output = self.norm_layer(output, t)
        
        return output

    def get_kb_params(self, t):
        # get knowledge base parameters for task t
        self.old_weight = torch.empty(0).to(device)
        for i in range(t):
            self.kb_weight = torch.cat([torch.cat([self.old_weight, self.bwt_weight[i]], dim=1), 
                                torch.cat([self.fwt_weight[i], self.weight[i]], dim=1)], dim=0)

    def get_ets_params(self, t):
        # get expanded task specific model
        weight = self.old_weight
        weight = F.dropout(weight, self.dropout, self.training)
        weight = torch.cat([torch.cat([weight, self.bwt_weight[t]], dim=1), 
                                torch.cat([self.fwt_weight[t], self.weight[t]], dim=1)], dim=0)

        bound_std = self.gain / math.sqrt(weight.shape[1] * self.ks)
        weight = weight * bound_std
        return weight, None
    
    def get_masked_kb_params(self, t, mode):
        # select parameters from knowledge base to build: knowledge base task specific model and join rehearsal model
        weight = self.old_weight
        fan_out = max(self.base_out_features, self.shape_out[t])
        fan_in = max(self.base_in_features, self.shape_in[t])
        add_out = max(self.base_out_features - self.shape_out[t], 0)
        add_in = max(self.base_in_features - self.shape_in[t], 0)
        n_0 = add_out * (fan_in-add_in) * self.ks
        n_1 = fan_out * add_in * self.ks
        if isinstance(self, DynamicConv2D):
            dummy_weight_0 = self.dummy_weight[:n_0].view(add_out, (fan_in-add_in) // self.groups, *self.kernel_size)
            dummy_weight_1 = self.dummy_weight[n_0:n_0+n_1].view(fan_out, add_in // self.groups, *self.kernel_size)
        else:
            dummy_weight_0 = self.dummy_weight[:n_0].view(add_out, (fan_in-add_in))
            dummy_weight_1 = self.dummy_weight[n_0:n_0+n_1].view(fan_out, add_in)
        weight = torch.cat([torch.cat([weight, dummy_weight_0], dim=0), dummy_weight_1], dim=1)

        bound_std = self.gain / math.sqrt(weight.shape[1] * self.ks)
        weight = weight * bound_std
        if self.training:
            if mode == 'kbts':
                mask = GetSubnet.apply(self.score_kbts.abs(), self.sparsity)
                weight = weight * mask / self.sparsity
                self.register_buffer('kbts_mask'+f'_{t}', mask.detach().bool())
            else:
                mask = GetSubnet.apply(self.score_jr.abs(), self.sparsity)
                weight = weight * mask / self.sparsity
                self.register_buffer('jr_mask', mask.detach().bool())
        else:
            if mode == 'kbts':
                weight = weight * getattr(self, 'kbts_mask'+f'_{t}') / self.sparsity
            else:
                weight = weight * getattr(self, 'jr_mask') / self.sparsity

        return weight, None
            

    def freeze(self):
        self.weight[-2].requires_grad = False
        self.fwt_weight[-2].requires_grad = False
        self.bwt_weight[-2].requires_grad = False
        if self.norm_layer:
            if self.norm_layer.affine:
                self.norm_layer.weight[-2].requires_grad = False
                self.norm_layer.bias[-2].requires_grad = False

    def clear_memory(self):
        self.score_jr = None
        self.score_kbts = None
        self.old_weight = None
        
    def update_scale(self):
        for i in range(self.cur_task):
            w = self.weight[i][-1]
            if w.numel() != 0:
                w_std = w.std(unbiased=False).item()
                self.scale[i][-1] = w_std

            w = self.weight[-1][i]
            if w.numel() != 0:
                w_std = w.std(unbiased=False).item()
                self.scale[-1][i] = w_std

        w = self.weight[-1][-1]
        if w.numel() != 0:
            w_std = w.std(unbiased=False).item()
            self.scale[-1][-1] = w_std

    def get_optim_params(self):
        params = [self.weight[-1], self.fwt_weight[-1], self.bwt_weight[-1]]
        params += [self.score_kbts, self.score_jr]
        if self.norm_layer:
            if self.norm_layer.affine:
                params += [self.norm_layer.weight[-1], self.norm_layer.bias[-1]]
        return params

    def count_params(self, t):
        count = 0
        for i in range(t+1):
            count += self.weight[i].numel() + self.fwt_weight[i].numel() + self.bwt_weight[i].numel()
            if self.norm_layer:
                if self.norm_layer.affine:
                    count += self.norm_layer.weight[i].numel() + self.norm_layer.bias[i].numel()
        return count

    def norm_in(self):
        weight = torch.cat([self.fwt_weight[-1], self.weight[-1]], dim=1)
        norm = weight.norm(2, dim=self.dim_in)
        # norm = (weight ** 2).mean(dim=self.dim_in) ** 0.5
        return norm

    def squeeze(self, optim_state, mask_in=None, mask_out=None):
        prune_out = mask_out is not None and mask_out.sum() != self.num_out[-1]
        prune_in = mask_in is not None and mask_in.sum() != self.num_in[-1]
        if prune_out:
            apply_mask_out(self.weight[-1], mask_out, optim_state)
            apply_mask_out(self.fwt_weight[-1], mask_out, optim_state)

            self.num_out[-1] = self.weight[-1].shape[0]
            self.shape_out[-1] = self.num_out.sum()

            mask = torch.ones(self.shape_out[-2], dtype=bool, device=device)
            mask = torch.cat([mask, mask_out])
            if self.bias is not None:
                apply_mask_out(self.bias[-1], mask, optim_state)

            if self.norm_layer:
                if self.norm_layer.affine:
                    apply_mask_out(self.norm_layer.weight[-1], mask, optim_state)
                    apply_mask_out(self.norm_layer.bias[-1], mask, optim_state)

                if self.norm_layer.track_running_stats:
                    self.norm_layer.running_mean[-1] = self.norm_layer.running_mean[-1][mask]
                    self.norm_layer.running_var[-1] = self.norm_layer.running_var[-1][mask]

                self.norm_layer.shape[-1] = self.shape_out[-1]
        
        if prune_in:
            if self.s != 1:
                mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0), self.s, self.s).contiguous().view(-1)
            apply_mask_in(self.weight[-1], mask_in, optim_state)
            apply_mask_in(self.bwt_weight[-1], mask_in, optim_state)

            self.num_in[-1] = self.weight[-1].shape[1]
            self.shape_in[-1] = self.num_in.sum()


    def proximal_gradient_descent(self, lr, lamb):
        eps = 0
        with torch.no_grad():
            strength = self.reg_strength

            # regularize std
            norm = self.norm_in()
            aux = 1 - lamb * lr * strength / norm
            aux = F.threshold(aux, 0, eps, False)
            self.mask_out = (aux > eps)
            self.weight[-1].data *= aux.view(self.view_in)
            self.fwt_weight[-1].data *= aux.view(self.view_in)
                
            # group lasso affine weights
            # if self.norm_layer:
            #     if self.norm_layer.affine:
            #         norm = self.norm_layer.norm()
            #         aux = 1 - lamb * lr * strength / norm
            #         aux = F.threshold(aux, 0, eps, False)
            #         self.mask_out *= (aux > eps)
            #         self.norm_layer.weight[-1].data[self.norm_layer.shape[-2]:] *= aux
            #         self.norm_layer.bias[-1].data[self.norm_layer.shape[-2]:] *= aux


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
        super(_DynamicConvNd, self).__init__(in_features, out_features, bias, norm_type, args, s)
        if in_features % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_features % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups


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


class DynamicClassifier(DynamicLinear):

    def __init__(self, in_features, out_features, bias=True, norm_type=None, args=None, s=1):
        super(DynamicClassifier, self).__init__(in_features, out_features, bias, norm_type, args, s)
        self.weight_ets = nn.ParameterList([])
        self.weight_kbts = nn.ParameterList([])
        self.bias_ets = nn.ParameterList([])
        self.bias_kbts = nn.ParameterList([])

    def expand(self, add_in=None, add_out=None):
        if add_in is None:
            if self.args.fix:
                add_in = self.base_in_features - self.shape_in[-1]
            else:
                add_in = self.base_in_features
        if add_out is None:
            add_out = self.base_out_features


        self.num_out = torch.cat([self.num_out, torch.IntTensor([add_out]).to(device)])
        self.num_in = torch.cat([self.num_in, torch.IntTensor([add_in]).to(device)])

        self.shape_out = torch.cat([self.shape_out, torch.IntTensor([self.shape_out[-1] + add_out]).to(device)])
        self.shape_in = torch.cat([self.shape_in, torch.IntTensor([self.shape_in[-1] + add_in]).to(device)])

        fan_in_kbts = max(self.base_in_features, self.shape_in[-2])
        fan_in_jr = max(self.base_in_features, self.shape_in[-1])

        bound_std = self.gain / math.sqrt(self.shape_in[-1])
        self.weight_ets.append(nn.Parameter(torch.Tensor(self.num_out[-1], self.shape_in[-1]).normal_(0, bound_std).to(device)))
        self.bias_ets.append(nn.Parameter(torch.Tensor(self.num_out[-1]).zero_().to(device))) 

        bound_std = self.gain / math.sqrt(fan_in_kbts)
        self.weight_kbts.append(nn.Parameter(torch.Tensor(self.num_out[-1], fan_in_kbts).normal_(0, bound_std).to(device)))
        self.bias_kbts.append(nn.Parameter(torch.Tensor(self.num_out[-1]).zero_().to(device)))

        bound_std = self.gain / math.sqrt(fan_in_jr)
        self.weight_jr = nn.Parameter(torch.Tensor(self.shape_out[-1], fan_in_jr).normal_(0, bound_std).to(device))
        self.bias_jr = nn.Parameter(torch.Tensor(self.shape_out[-1]).zero_().to(device))

    def forward(self, x, t, mode='ets'):
        if mode == 'kbts':
            weight = self.weight_kbts[t]
            bias = self.bias_kbts[t]
        elif mode == 'jr':
            weight = self.weight_jr
            bias = self.bias_jr
            if t is not None:
                weight = weight[self.shape_out[t]:self.shape_out[t+1]]
                bias = bias[self.shape_out[t]:self.shape_out[t+1]]
        else:
            weight = self.weight_ets[t]
            bias = self.bias_ets[t]
        x = F.linear(x, weight, bias)
        return x
    
    def freeze(self):
        self.weight_ets[-2].requires_grad = False
        self.weight_kbts[-2].requires_grad = False
        self.bias_ets[-2].requires_grad = False
        self.bias_kbts[-2].requires_grad = False
        

    def get_optim_params(self):
        params = [self.weight_ets[-1], self.weight_kbts[-1], self.weight_jr]
        params += [self.bias_ets[-1], self.bias_kbts[-1], self.bias_jr]
        return params

    def count_params(self, t):
        count = 0
        for i in range(t+1):
            count += self.weight_ets[i].numel() + self.weight_kbts[-1].numel()
            count += self.bias_ets[i].numel() + self.bias_kbts[-1].numel()
        count += self.weight_jr.numel() + self.bias_jr.numel()
        return count

    def squeeze(self, optim_state, mask_in=None, mask_out=None):
        if mask_in is not None:
            if self.s != 1:
                mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0), self.s, self.s).contiguous().view(-1)
            
            apply_mask_in(self.weight_ets[-1], mask_in, optim_state)
            self.num_in[-1] = self.weight_ets[-1].shape[1]
            self.shape_in[-1] = self.num_in.sum()
  

class DynamicNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True, norm_type=None):
        super(DynamicNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.base_num_features = num_features
        self.num_features = 0
        self.shape = [0]
        self.norm_type = norm_type
        if 'affine' in norm_type:
            self.affine = True
        else:
            self.affine = False

        if 'track' in norm_type:
            self.track_running_stats = True
        else:
            self.track_running_stats = False

        if self.affine:
            self.weight = []
            self.bias = []

        if self.track_running_stats:
            self.running_mean = []
            self.running_var = []
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def expand(self, add_num=None):
        if add_num is None:
            add_num = self.base_num_features

        self.num_features += add_num
        self.shape.append(self.num_features)

        if self.affine:
            self.weight.append(nn.Parameter(torch.Tensor(self.num_features).uniform_(1,1).to(device)))
            self.bias.append(nn.Parameter(torch.Tensor(self.num_features).uniform_(0,0).to(device)))
            if len(self.weight) > 2:
                self.weight[-2].requires_grad = False
                self.bias[-2].requires_grad = False

        if self.track_running_stats:
            self.running_mean.append(torch.zeros(self.num_features).to(device))
            self.running_var.append(torch.ones(self.num_features).to(device))
            self.num_batches_tracked = 0
    
    def norm(self, t=-1):
        return (self.weight[t][self.shape[t-1]:]**2 + self.bias[t][self.shape[t-1]:]**2) ** 0.5

    def batch_norm(self, input, t=-1):

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
            # var = input.var([0, 2, 3], unbiased=False)
            shape = (1, -1, 1, 1)
            var = ((input - mean.view(shape)) ** 2).mean([0, 2, 3])
        else:
            mean = input.mean([0])
            # var = input.var([0], unbiased=False)
            shape = (1, -1)
            var = ((input - mean.view(shape)) ** 2).mean([0])

        # calculate running estimates
        if bn_training:
            if self.track_running_stats:
                n = input.numel() / input.size(1)
                with torch.no_grad():
                    self.running_mean[t] = exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * self.running_mean[t]
                    # update running_var with unbiased var
                    self.running_var[t] = exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_var[t]
        else:
            mean = self.running_mean[t]
            var = self.running_var[t]

        if 'res' in self.norm_type:
            return input / (torch.sqrt(var.view(shape) + self.eps))
        else:
            return (input - mean.view(shape)) / (torch.sqrt(var.view(shape) + self.eps))


    def layer_norm(self, input):
        if len(input.shape) == 4:
            mean = input.mean([1, 2, 3])
            var = input.var([1, 2, 3], unbiased=False)
            shape = (-1, 1, 1, 1)
        else:
            mean = input.mean([1])
            var = input.var([1], unbiased=False)
            shape = (-1, 1)

        return (input - mean.view(shape)) / (torch.sqrt(var.view(shape) + self.eps))

    def L2_norm(self, input):
        if len(input.shape) == 4:
            norm = input.norm(2, dim=(1,2,3)).view(-1,1,1,1)
        else:
            norm = input.norm(2, dim=(1)).view(-1,1)

        return input / norm

    def forward(self, input, t=-1):
        output = self.batch_norm(input, t)

        if self.affine:
            weight = self.weight[t]
            bias = self.bias[t]
            if len(input.shape) == 4:
                output = output * weight.view(1,-1,1,1) + bias.view(1,-1,1,1)
            else:
                output = output * weight.view(1,-1) + bias.view(1,-1)

        return output

    
            