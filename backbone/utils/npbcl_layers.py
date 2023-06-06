import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

from functools import partial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TopK(torch.autograd.Function):
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
    
class MaksedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, args=None):
        super(MaksedLinear, self).__init__(in_features, out_features, bias)  
        self.stable_score = nn.Parameter(torch.empty_like(self.weight), requires_grad=True) 
        self.plastic_score = nn.Parameter(torch.empty_like(self.weight), requires_grad=True)
        nn.init.kaiming_normal_(self.stable_score)   
        nn.init.kaiming_normal_(self.plastic_score)
        self.stable_masks = [torch.zeros_like(self.weight).to(device) for _ in range(args.total_tasks)]
        self.plastic_masks = [torch.zeros_like(self.weight).to(device) for _ in range(args.total_tasks)]
        self.unused_weight = torch.ones_like(self.weight).to(device)
        self.args = args
        self.sparsity = 0.5
        self.mode = 'ensemble'
        self.dim_in = (0)
        self.dim_out = (1)
        self.view_in = (1, -1)
        self.view_out = (-1, 1)

    def update_unused_weights(self, t): 
        # zero if used, one if unused
        self.unused_weight = torch.ones_like(self.weight).to(device)
        for mask in self.stable_masks[:t+1] + self.plastic_masks[:t+1]:
            self.unused_weight *= (~mask.bool())
        # print(self.unused_weight.sum(), self.unused_weight.numel())

    def freeze_used_weights(self):
        self.weight.grad *= self.unused_weight

    def forward(self, inputs, t): 
        if self.mode == 'ensemble':
            N = inputs.shape[0] // 2
            inputs = inputs.split(N, dim=0)

            if self.training:
                self.stable_masks[t] = TopK.apply(self.stable_score.abs(), 1-self.sparsity) / (1-self.sparsity)
                self.plastic_masks[t] = TopK.apply(self.plastic_score.abs(), 1-self.sparsity) / (1-self.sparsity)

            stable_out = F.linear(inputs[0], self.stable_masks[t] * self.weight, self.bias)
            plastic_out = F.linear(inputs[1], self.plastic_masks[t] * self.weight, self.bias)

            return torch.cat([stable_out, plastic_out], dim=0)
        elif self.mode == 'stable':
            if self.training:
                self.stable_masks[t] = TopK.apply(self.stable_score.abs(), 1-self.sparsity) / (1-self.sparsity)
            stable_out = F.linear(inputs, self.stable_masks[t] * self.weight, self.bias)
            return stable_out
        elif self.mode == 'plastic':
            if self.training:
                self.plastic_masks[t] = TopK.apply(self.plastic_score.abs(), 1-self.sparsity) / (1-self.sparsity)
            plastic_out = F.linear(inputs, self.plastic_masks[t] * self.weight, self.bias)
            return plastic_out


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', args=None):
        super(MaskedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode)
        self.stable_score = nn.Parameter(torch.empty_like(self.weight), requires_grad=True) 
        self.plastic_score = nn.Parameter(torch.empty_like(self.weight), requires_grad=True)
        nn.init.kaiming_normal_(self.stable_score)   
        nn.init.kaiming_normal_(self.plastic_score)
        self.stable_masks = [torch.zeros_like(self.weight).to(device) for _ in range(args.total_tasks)]
        self.plastic_masks = [torch.zeros_like(self.weight).to(device) for _ in range(args.total_tasks)]
        self.unused_weight = torch.ones_like(self.weight).to(device)
        self.args = args
        self.sparsity = 0.5
        self.mode = 'ensemble'
        self.dim_in = (0,2,3)
        self.dim_out = (1,2,3)
        self.view_in = (1, -1, 1, 1)
        self.view_out = (-1, 1, 1, 1)

    def _conv_forward(self, inputs, weight, bias, groups):
        return F.conv2d(inputs, weight, bias, self.stride,
                        self.padding, self.dilation, groups)
    
    def update_unused_weights(self, t): 
        # zero if used, one if unused
        self.unused_weight = torch.ones_like(self.weight).to(device)
        for mask in self.stable_masks[:t+1] + self.plastic_masks[:t+1]:
            self.unused_weight *= (~mask.bool())
        # print(self.unused_weight.sum(), self.unused_weight.numel())

    def freeze_used_weights(self):
        self.weight.grad *= self.unused_weight

    def forward(self, inputs, t):
        if self.mode == 'ensemble':
            N, C, H, W = inputs.shape
            H = H//self.stride[0]
            W = W//self.stride[0]

            E = 2
            N = N//E
            ## input [N//num_member, num_member*chn_in, H, W]
            inputs = inputs.view(E, N, *inputs.shape[1:]).permute(1,0,2,3,4).reshape(N, E*self.in_channels, *inputs.shape[2:])
        
            if self.training:
                self.stable_masks[t] = TopK.apply(self.stable_score.abs(), 1-self.sparsity) / (1-self.sparsity)
                self.plastic_masks[t] = TopK.apply(self.plastic_score.abs(), 1-self.sparsity) / (1-self.sparsity)

            ## filters with shape [num_member*chn_out, chn_in, k, k]
            weight = torch.cat([self.stable_masks[t] * self.weight, self.plastic_masks[t] * self.weight], dim=0)
            ## out with shape [N//num_member, num_member*chn_out, H', W']
            out = self._conv_forward(inputs, weight, None, E)
            out = out.view(-1, E, self.out_channels, H, W).permute(1,0,2,3,4).reshape(-1, self.out_channels, H, W)

            return out
        
        elif self.mode == 'stable':
            if self.training:
                self.stable_masks[t] = TopK.apply(self.stable_score.abs(), 1-self.sparsity) / (1-self.sparsity)
            return self._conv_forward(inputs, self.stable_masks[t] * self.weight, None, 1)
        elif self.mode == 'plastic':
            if self.training:
                self.plastic_masks[t] = TopK.apply(self.plastic_score.abs(), 1-self.sparsity) / (1-self.sparsity)
            return self._conv_forward(inputs, self.plastic_masks[t] * self.weight, None, 1)
        
        
class EsmBatchNorm2d(nn.Module):
    def __init__(self, num_channels, args=None):
        super(EsmBatchNorm2d, self).__init__()
        self.stable_bns = nn.ModuleList([nn.BatchNorm2d(num_channels) for _ in range(args.total_tasks)])
        self.plastic_bns = nn.ModuleList([nn.BatchNorm2d(num_channels) for _ in range(args.total_tasks)])
        self.mode = 'ensemble'
        self.args = args

    def forward(self, inputs, t):
        if self.mode == 'ensemble':
            N = inputs.shape[0] // 2
            inputs = inputs.split(N, dim=0)
            return torch.cat([self.stable_bns[t](inputs[0]), self.plastic_bns[t](inputs[1])], dim=0)
        elif self.mode == 'stable':
            return self.stable_bns[t](inputs)
        elif self.mode == 'plastic':
            return self.plastic_bns[t](inputs)

class EsmLinear(nn.Module):
    def __init__(self, in_channels, out_channels, args=None):
        super(EsmLinear, self).__init__()
        self.stable_linears = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(args.total_tasks)])
        self.plastic_linears = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(args.total_tasks)])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = 'ensemble'
        self.args = args

    def forward(self, inputs, t):
        if self.mode == 'ensemble':
            N = inputs.shape[0] // 2
            inputs = inputs.split(N, dim=0)
            return torch.cat([self.stable_linears[t](inputs[0]), self.plastic_linears[t](inputs[1])], dim=0)
        elif self.mode == 'stable':
            return self.stable_linears[t](inputs)
        elif self.mode == 'plastic':
            return self.plastic_linears[t](inputs)