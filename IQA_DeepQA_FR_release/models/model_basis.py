from __future__ import absolute_import, division, print_function

import pickle
from collections import OrderedDict

import numpy as np
import theano.tensor as T
from functools import reduce

from .. import optimizer
from ..layers import layers


class ModelBasis(object):
    """
    Arguments
        model_config: model configuration dictionary

    Attributes of model_config
        input_size: input image size, (height, width).
        num_ch: number of input channels
        lr: initial learning rate
    """

    def __init__(self, model_config={}, rng=None):
        # Make input_shape
        self.input_size = tuple(model_config.get('input_size', None))
        assert len(self.input_size) == 2
        self.num_ch = model_config.get('num_ch', None)
        assert self.num_ch is not None
        self.input_shape = (None, self.num_ch) + self.input_size

        # Get optimizer
        self.opt = optimizer.Optimizer()
        self.set_opt_configs(model_config)

        # Initialize variables
        self.layers = OrderedDict()
        self.params = OrderedDict()

    def set_opt_configs(self, model_config=None, opt_scheme=None, lr=None):
        if model_config is None:
            assert lr is not None and opt_scheme is not None
        else:
            lr = float(model_config.get('lr', 1e-3))
            opt_scheme = model_config.get('opt_scheme', 'adam')
        self.lr = lr
        self.opt_scheme = opt_scheme
        self.opt.set_learning_rate(self.lr)

    ###########################################################################
    # Functions for cost calculation

    def get_l2_regularization(self, layer_keys=None, mode='sum',
                              attr_list=['W', 'gamma']):
        if layer_keys is None:
            layer_keys = list(self.layers.keys())
        l2 = []
        if mode == 'sum':
            for key in layer_keys:
                for layer in self.layers[key]:
                    for attr in attr_list:
                        if hasattr(layer, attr):
                            l2.append(T.sum(getattr(layer, attr) ** 2))
            return T.sum(l2)
        elif mode == 'mean':
            for key in layer_keys:
                for layer in self.layers[key]:
                    for attr in attr_list:
                        if hasattr(layer, attr):
                            l2.append(T.sum(getattr(layer, attr) ** 2))
            return T.mean(l2)
        else:
            raise NotImplementedError

    def get_mse(self, x, y, return_map=False):
        if return_map:
            return (x - y) ** 2
        else:
            # return T.mean(((x - y) ** 2).flatten(2), axis=1)
            return T.mean((x - y) ** 2)

    def add_all_weighted_losses(self, losses, weights):
        """Add the losses with the weights multiplied.
        If the weight is 0, the corresponding loss is ignored.
        """
        assert len(losses) == len(weights)
        loss_list = []
        for loss, weight in zip(losses, weights):
            if weight != 0:
                loss_list.append(weight * loss)
        return reduce(lambda x, y: x + y, loss_list)

    ###########################################################################
    # Functions to help build layers

    def last_sh(self, key, nth=-1):
        """Get the `nth` output shape in the `key` layers
        """
        assert len(self.layers[key]) > 0, "No layers in the key: %s" % key
        idx = len(self.layers[key]) + nth if nth < 0 else nth
        out_sh = None
        while out_sh is None:
            if idx < 0:
                out_sh = self.input_shape
            out_sh = self.layers[key][idx].get_out_shape()
            idx = idx - 1
        return out_sh

    def get_concat_shape(self, key0, key1):
        """Get the concatenated shape of the outputs of
        `key0` and `key1` layers
        """
        prev_sh0 = self.last_sh(key0)
        prev_sh1 = self.last_sh(key1)
        if isinstance(prev_sh0, (list, tuple)):
            assert prev_sh0[0] == prev_sh1[0]
            assert prev_sh0[2:] == prev_sh1[2:]
            return (prev_sh0[0], prev_sh0[1] + prev_sh1[1]) + prev_sh0[2:]
        else:
            return prev_sh0 + prev_sh1

    def image_vec_to_tensor(self, input):
        """Reshape input into 4D tensor.
        """
        # im_sh = (-1, self.input_size[0],
        #          self.input_size[1], self.num_ch)
        # return input.reshape(im_sh).dimshuffle(0, 3, 1, 2)
        return input.dimshuffle(0, 3, 1, 2)


    ###########################################################################

    def get_key_layers_output(self, input, key, var_shape=False):
        """Put `input` to the `key` layers and return the final output.
        """
        prev_out = input
        for layer in self.layers[key]:
            prev_out = layer.get_output(prev_out, var_shape=var_shape)
        return prev_out

    def get_updates(self, cost, wrt_params):
        return self.opt.get_updates_cost(cost, wrt_params, self.opt_scheme)

    def get_updates_keys(self, cost, keys=[], params=[],
                         params_lr_factors=None):
        wrt_params = []
        for key in keys:
            wrt_params += self.params[key]
        if params:
            wrt_params += params

        lr_factors = None
        if params_lr_factors:
            lr_factors = []
            for key in keys:
                lr_factors += params_lr_factors[key]
            assert len(wrt_params) == len(lr_factors)

            # remove factors of 0
            new_wrt_params = []
            new_lr_factors = []
            for idx in range(len(wrt_params)):
                if lr_factors[idx] > 0.0:
                    new_wrt_params.append(wrt_params[idx])
                    new_lr_factors.append(lr_factors[idx])
            wrt_params = new_wrt_params
            lr_factors = new_lr_factors

        print(' - Update w.r.t.: %s' % ', '.join(keys))
        return self.opt.get_updates_cost(cost, wrt_params, self.opt_scheme,
                                         lr_factors)

    ###########################################################################
    # Functions to control batch normalization and dropout layers

    def get_batch_norm_layers(self, keys=[]):
        # For the first call, generate bn_layers
        if not hasattr(self, 'bn_layers'):
            self.bn_layers = {}
            for key in list(self.layers.keys()):
                self.bn_layers[key] = []
                for layer in self.layers[key]:
                    if layer.__class__.__name__ == 'BatchNormLayer':
                        self.bn_layers[key].append(layer)

        layers = []
        for key in keys:
            layers += self.bn_layers[key]
        return layers

    def set_batch_norm_update_averages(self, update_averages, keys=[]):
        # if update_averages:
        #     print(' - Batch norm: update the stored averages')
        # else:
        #     print(' - Batch norm: not update the stored averages')
        layers = self.get_batch_norm_layers(keys)
        for layer in layers:
            layer.update_averages = update_averages

    def set_batch_norm_training(self, training, keys=[]):
        # if training:
        #     print(' - Batch norm: use mini-batch statistics')
        # else:
        #     print(' - Batch norm: use the stored statistics')
        layers = self.get_batch_norm_layers(keys)
        for layer in layers:
            layer.deterministic = not training

    def set_dropout_on(self, training):
        layers.DropoutLayer.set_dropout_training(training)

    def set_training_mode(self, training):
        """Decide the behavior of batch normalization and dropout.
        Parameters
        ----------
        training: boolean
            if True, training mode / False: testing mode.
        """
        # Decide behaviors of the model during training
        # Batch normalization
        l_keys = [key for key in list(self.layers.keys())]
        self.set_batch_norm_update_averages(training, l_keys)
        self.set_batch_norm_training(training, l_keys)

        # Dropout
        self.set_dropout_on(training)

    ###########################################################################
    # Functions to help deal with parameters of the model

    def make_param_list(self):
        """collect all the parameters from `self.layers`
        """
        self.params, self.bn_layers = {}, {}

        for key in list(self.layers.keys()):
            self.params[key] = []
            self.bn_layers[key] = []
            for layer in self.layers[key]:
                if layer.get_params():
                    self.params[key] += layer.get_params()
                if layer.__class__.__name__ == 'BatchNormLayer':
                    self.bn_layers[key].append(layer)

    def get_lr_factors_of_params(self, lr_factors_dict):
        """collect all the parameters from `self.layers`
        """
        params_lr_factors = {}
        for key in list(self.layers.keys()):
            params_lr_factors[key] = []
            for layer in self.layers[key]:
                for p in layer.get_params():
                    params_lr_factors[key].append(
                        lr_factors_dict.get(layer.name, 1.0))
        return params_lr_factors

    def show_num_params(self):
        """Dislay the number of parameters for each layer_key.
        """
        paramscnt = {}
        for key in list(self.layers.keys()):
            paramscnt[key] = 0
            for p in self.params[key]:
                paramscnt[key] += np.prod(p.get_value(borrow=True).shape)
            if paramscnt[key] > 0:
                print(' - Num params %s:' % key, '{:,}'.format(paramscnt[key]))

    def get_params(self, layer_keys=None):
        """Get concatenated parameter list
        from layers belonging to layer_keys"""
        if layer_keys is None:
            layer_keys = list(self.layers.keys())

        params = []
        bn_mean_std = []
        for key in layer_keys:
            params += self.params[key]

        for key in layer_keys:
            for layer in self.bn_layers[key]:
                bn_mean_std += layer.statistics
        params += bn_mean_std
        return params

    def save(self, filename):
        """Save parameters to file.
        """
        params = self.get_params()
        with open(filename, 'wb') as f:
            pickle.dump(params, f, protocol=2)
            # pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(' = Save params: %s' % (filename))

    def load(self, filename):
        """Load parameters from file.
        """
        params = self.get_params()
        with open(filename, 'rb') as f:
            newparams = pickle.load(f)

        assert len(newparams) == len(params)
        for p, new_p in zip(params, newparams):
            if p.name != new_p.name:
                print((' @ WARNING: Different name - (loaded) %s != %s'
                      % (new_p.name, p.name)))
            new_p_sh = new_p.get_value(borrow=True).shape
            p_sh = p.get_value(borrow=True).shape
            if p_sh != new_p_sh:
                # print(new_p.name, p_sh, new_p_sh)
                print(' @ WARNING: Different shape %s - (loaded)' % new_p.name,
                      new_p_sh, end='')
                print(' !=', p_sh)
                continue
            p.set_value(new_p.get_value())
        print(' = Load all params: %s ' % (filename))

    def load_params_keys(self, layer_keys, filename):
        """Load the selected parameters from file.
        Parameters from layers belong to layer_keys.
        """
        print(' = Load params: %s (keys = %s)' % (
            filename, ', '.join(layer_keys)))
        to_params = self.get_params(layer_keys)
        with open(filename, 'rb') as f:
            from_params = pickle.load(f)

        # Copy the params having same shape and name
        copied_idx = []
        for fidx, f_param in enumerate(from_params):
            f_val = f_param.get_value(borrow=True)
            for tidx, t_param in enumerate(to_params):
                t_val = t_param.get_value(borrow=True)
                if f_val.shape == t_val.shape and f_param.name == t_param.name:
                    t_param.set_value(f_val)
                    del to_params[tidx]
                    copied_idx.append(fidx)
                    break
        # print(' = Copied from_param: ', [
        #     from_params[idx] for idx in copied_idx])
        if to_params:
            print(' = Not existing to_param: ', to_params)
