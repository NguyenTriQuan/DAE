# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from backbone.utils.dae_layers import DynamicLinear, DynamicConv2D, DynamicClassifier, _DynamicLayer

def xavier(m: nn.Module) -> None:
    """
    Applies Xavier initialization to linear modules.

    :param m: the module to be initialized

    Example::
        >>> net = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        >>> net.apply(xavier)
    """
    if m.__class__.__name__ == 'Linear':
        fan_in = m.weight.data.size(1)
        fan_out = m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def num_flat_features(x: torch.Tensor) -> int:
    """
    Computes the total number of items except the first dimension.

    :param x: input tensor
    :return: number of item from the second dimension onward
    """
    size = x.size()[1:]
    num_features = 1
    for ff in size:
        num_features *= ff
    return num_features

class MammothBackbone(nn.Module):

    def __init__(self, **kwargs) -> None:
        super(MammothBackbone, self).__init__()

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :return: output tensor (output_classes)
        """
        raise NotImplementedError

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, returnt='features')

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        return torch.cat(self.get_grads_list())

    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads

class _DynamicModel(nn.Module):
    def __init__(self):
        super(_DynamicModel, self).__init__()
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]

    def get_optim_params(self):
        params = []
        for m in self.DM:
            params += m.get_optim_params()
        return params

    def count_params(self, t=-1):
        if t == -1:
            t = len(self.DM[-1].num_out)-1
        num_params = []
        num_neurons = []
        for m in self.DM:
            num_params.append(m.count_params(t))
            num_neurons.append(m.shape_out[t+1].item())
        return num_params, num_neurons

    def freeze(self):
        for m in self.DM:
            m.freeze()
    
    def clear_memory(self):
        for m in self.DM[:-1]:
            m.clear_memory()
    
    def update_scale(self):
        for m in self.DM[:-1]:
            m.update_scale()
    
    def get_kb_params(self, t):
        for m in self.DM[:-1]:
            m.get_kb_params(t)
    
    def get_masked_kb_params(self, t):
        if t == 0:
            add_in = self.DM[0].base_in_features
        else:
            add_in = 0

        for m in self.DM[:-1]:
            add_in = m.get_masked_kb_params(t, add_in)

    def set_jr_params(self):
        add_in = 0
        for m in self.DM:
            add_in = m.set_jr_params(add_in)
    
    def set_squeeze_state(self, squeeze):
        for m in self.DM[:-1]:
            m.set_squeeze_state(squeeze)

    def initialize(self):
        for layers in self.prev_layers:
            for layer in layers:
                for i in range(layer.task):
                    std = layer.gain / math.sqrt((layer.task+1) * len(layers) * layer.ks * layer.num_out[layer.task])
                    nn.init.normal_(getattr(layer, f'weight_{i}_{layer.task}'), 0, std)

                    std = layer.gain / math.sqrt((layer.task+1) * len(layers) * layer.ks * layer.num_out[i])
                    nn.init.normal_(getattr(layer, f'weight_{layer.task}_{i}'), 0, std)

                std = layer.gain / math.sqrt((layer.task+1) * len(layers) * layer.ks * layer.num_out[layer.task])
                nn.init.normal_(getattr(layer, f'weight_{layer.task}_{layer.task}'), 0, std)
        
        std = 1 / math.sqrt(self.DM[-1].num_out[-1])
        nn.init.normal_(self.DM[-1].weight_ets[-1], 0, std)
        self.normalize()

    def normalize(self):
        num_tasks = self.DM[-1].task + 1
        for layers in self.prev_layers:
            var_layers_in = 0
            # for i in range(num_tasks-1):
            #     var_layers_out = 0
            #     for layer in layers:
            #         mean = getattr(layer, f'weight_{i}_{layer.task}').mean(layer.dim_in)
            #         getattr(layer, f'weight_{i}_{layer.task}').data -= mean.view(layer.view_in)
            #         var = (getattr(layer, f'weight_{i}_{layer.task}') ** 2).mean(layer.dim_in)
            #         var_layers_out += var * layer.ks / (layer.gain ** 2)

            #         mean = getattr(layer, f'weight_{layer.task}_{i}').mean(layer.dim_in)
            #         getattr(layer, f'weight_{layer.task}_{i}').data -= mean.view(layer.view_in)
            #         var = (getattr(layer, f'weight_{layer.task}_{i}') ** 2).mean(layer.dim_in)
            #         var_layers_in += var * layer.ks / (layer.gain ** 2)

            #     for layer in layers:
            #         getattr(layer, f'weight_{i}_{layer.task}').data /= math.sqrt(num_tasks * var_layers_out.sum())

            for layer in layers:
                mean = getattr(layer, f'weight_{layer.task}_{layer.task}').mean(layer.dim_in)
                getattr(layer, f'weight_{layer.task}_{layer.task}').data -= mean.view(layer.view_in)
                var = (getattr(layer, f'weight_{layer.task}_{layer.task}') ** 2).mean(layer.dim_in)
                var_layers_in += var * layer.ks / (layer.gain ** 2)
            
            sum_std_layers_in = var_layers_in.sqrt().sum()
            for layer in layers:
                # for i in range(layer.task):
                #     getattr(layer, f'weight_{layer.task}_{i}').data /= sum_std_layers_in
                getattr(layer, f'weight_{layer.task}_{layer.task}').data /= sum_std_layers_in

        mean = self.DM[-1].weight_ets[-1].mean(self.DM[-1].dim_in)
        self.DM[-1].weight_ets[-1].data -= mean.view(self.DM[-1].view_in)
        var = (self.DM[-1].weight_ets[-1] ** 2).mean(self.DM[-1].dim_in)
        self.DM[-1].weight_ets[-1].data /= math.sqrt(var.sum())
        # self.check()

    def proximal_gradient_descent(self, lr, lamb):
        num_tasks = self.DM[-1].task + 1
        for layers in self.prev_layers:
            var_layers_in = 0
            # for i in range(num_tasks-1):
            #     var_layers_out = 0
            #     for layer in layers:
            #         mean = getattr(layer, f'weight_{i}_{layer.task}').mean(layer.dim_in)
            #         getattr(layer, f'weight_{i}_{layer.task}').data -= mean.view(layer.view_in)
            #         var = (getattr(layer, f'weight_{i}_{layer.task}') ** 2).mean(layer.dim_in)
            #         var_layers_out += var * layer.ks / (layer.gain ** 2)

            #         mean = getattr(layer, f'weight_{layer.task}_{i}').mean(layer.dim_in)
            #         getattr(layer, f'weight_{layer.task}_{i}').data -= mean.view(layer.view_in)
            #         var = (getattr(layer, f'weight_{layer.task}_{i}') ** 2).mean(layer.dim_in)
            #         var_layers_in += var * layer.ks / (layer.gain ** 2)

            #     for layer in layers:
            #         getattr(layer, f'weight_{i}_{layer.task}').data /= math.sqrt(num_tasks * var_layers_out.sum())

            for layer in layers:
                mean = getattr(layer, f'weight_{layer.task}_{layer.task}').mean(layer.dim_in)
                getattr(layer, f'weight_{layer.task}_{layer.task}').data -= mean.view(layer.view_in)
                var = (getattr(layer, f'weight_{layer.task}_{layer.task}') ** 2).mean(layer.dim_in)
                var_layers_in += var * layer.ks / (layer.gain ** 2)

            strength = layer.strength_in / self.total_strength
            std_layers_in = var_layers_in.sqrt()
            aux = 1 - lamb * lr * strength / std_layers_in
            aux = F.threshold(aux, 0, 0, False)
            
            sum_std_layers_in = (var_layers_in * aux**2).sqrt().sum()
            for layer in layers:
                layer.mask_out = (aux > 0).clone().detach()
                getattr(layer, f'weight_{layer.task}_{layer.task}').data *= aux.view(layer.view_in)
                # for i in range(self.task):
                #     f'weight_{i}_{self.task}'.data *= aux.view(self.view_in)
                # for i in range(layer.task):
                #     getattr(layer, f'weight_{layer.task}_{i}').data /= sum_std_layers_in
                getattr(layer, f'weight_{layer.task}_{layer.task}').data /= sum_std_layers_in

        mean = self.DM[-1].weight_ets[-1].mean(self.DM[-1].dim_in)
        self.DM[-1].weight_ets[-1].data -= mean.view(self.DM[-1].view_in)
        var = (self.DM[-1].weight_ets[-1] ** 2).mean(self.DM[-1].dim_in)
        self.DM[-1].weight_ets[-1].data /= math.sqrt(var.sum())
        # self.check()

    def check(self):
        with torch.no_grad():
            for layers in self.prev_layers:
                var_layers_in = 0
                for layer in layers:
                    print(layer.name, layer.ks, end=' ')
                    for i in range(layer.task):
                        mean = getattr(layer, f'weight_{layer.task}_{i}').mean(layer.dim_in)
                        var = (getattr(layer, f'weight_{layer.task}_{i}') ** 2).mean(layer.dim_in)
                        var_layers_in += var * layer.ks / (layer.gain ** 2)

                    mean = getattr(layer, f'weight_{layer.task}_{layer.task}').mean(layer.dim_in)
                    var = (getattr(layer, f'weight_{layer.task}_{layer.task}') ** 2).mean(layer.dim_in)
                    var_layers_in += var * layer.ks / (layer.gain ** 2)
                    # print(mean.sum().item(), end=' ')
                
                print('|Var|', var_layers_in.sum().item())

    def ERK_sparsify(self, sparsity=0.9):
        # print('initialize by ERK')
        density = 1 - sparsity
        erk_power_scale = 1

        total_params = 0
        for m in self.DM[:-1]:
            total_params += m.score.numel()
        is_epsilon_valid = False

        dense_layers = set()
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            for m in self.DM[:-1]:
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
            max_prob = np.max([m.raw_probability for m in self.DM[:-1]])
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for m in self.DM[:-1]:
                    if m.raw_probability == max_prob:
                        # print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(m)
            else:
                is_epsilon_valid = True

        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        min_sparsity = 0.5
        for i, m in enumerate(self.DM[:-1]):
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
