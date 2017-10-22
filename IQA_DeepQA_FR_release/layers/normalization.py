from __future__ import absolute_import, division, print_function

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import bn
from .layers import Layer, linear


class BatchNormLayer(Layer):
    """
    Batch normalization layer.
    (theano.tensor.nnet.bn.batch_normalization_{train, test})

    Parameters
    ----------
    input_shape: int or a tuple of ints
        Input feature dimension or (batch_size, input feature dimension)
    activation: function
        Activation function.
    axes: {``'spatial'``, ``'per-activation'``}
    epsilon: float
    alpha: float
    """
    layers = []

    def __init__(self, input_shape, activation=linear, axis=1, axes='spatial',
                 epsilon=1e-4, alpha=0.1, name=None):
        super(BatchNormLayer, self).__init__()

        self.input_shape = input_shape
        self.activation = activation
        self.axis = axis
        self.axes_org = axes
        self.epsilon = epsilon
        self.alpha = alpha
        self.name = 'BN' if name is None else name
        self.act_name = activation.__name__
        self.deterministic = False

        shape = [input_shape[self.axis]]
        ndim = len(input_shape)
        if axes == 'per-activation':
            self.axes = (0,)
        elif axes == 'spatial':
            self.axes = (0,) + tuple(range(2, ndim))
        self.non_bc_axes = tuple(i for i in range(ndim) if i not in self.axes)

        self.gamma = theano.shared(np.ones(shape, dtype=theano.config.floatX),
                                   name=name + '_G', borrow=True)
        self.beta = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  name=name + '_B', borrow=True)

        self.mean = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  name=name + '_mean', borrow=True)
        self.var = theano.shared(np.ones(shape, dtype=theano.config.floatX),
                                 name=name + '_var', borrow=True)

        self.params = [self.gamma, self.beta]
        self.statistics = [self.mean, self.var]
        BatchNormLayer.layers.append(self)

        # Show information
        print('  # %s (BN) ' % (self.name), end='')
        print('act.: %s,' % self.act_name)

    def get_output(self, input, **kwargs):
        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(list(range(input.ndim - len(self.axes))))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = self.beta.dimshuffle(pattern)
        gamma = self.gamma.dimshuffle(pattern)
        mean = self.mean.dimshuffle(pattern)
        var = self.var.dimshuffle(pattern)

        if not self.deterministic:
            normalized, _, _, mean_, var_ = bn.batch_normalization_train(
                input, gamma, beta, self.axes_org,
                self.epsilon, self.alpha, mean, var)

            # Update running mean and variance
            # Tricks adopted from Lasagne implementation
            # http://lasagne.readthedocs.io/en/latest/modules/layers/normalization.html
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_var = theano.clone(self.var, share_inputs=False)
            running_mean.default_update = mean_.dimshuffle(self.non_bc_axes)
            running_var.default_update = var_.dimshuffle(self.non_bc_axes)
            self.mean += 0 * running_mean
            self.var += 0 * running_var
        else:
            normalized = bn.batch_normalization_test(
                input, gamma, beta, mean, var, self.axes_org, self.epsilon)
            # normalized, _, _, _, _ = bn.batch_normalization_train(
            #     input, gamma, beta, self.axes_org, self.epsilon, 0, mean, var)
            # normalized = (input - mean) * (gamma / T.sqrt(var)) + beta

        return self.activation(normalized)

    def get_out_shape(self):
        return self.input_shape

    def reset_stats(self):
        # reset mean and var
        self.mean.set_value(np.zeros(self.mean.get_value().shape,
                                     dtype=theano.config.floatX))
        self.var.set_value(np.ones(self.var.get_value().shape,
                                   dtype=theano.config.floatX))

    def get_stats(self):
        return (self.mean, self.var)

    @staticmethod
    def set_batch_norms_training(training):
        deterministic = False if training else True
        print(' - Batch norm layres: deterministic =', deterministic)
        for layer in BatchNormLayer.layers:
            layer.deterministic = deterministic
            layer.update_averages = not deterministic

    @staticmethod
    def reset_batch_norms_stats():
        print(' - Batch norm layres: reset mean and var')
        for layer in BatchNormLayer.layers:
            layer.reset_stats()


class BatchNormLayer_old(Layer):
    """
    Batch normalization layer
    (theano.tensor.nnet.bn.batch_normalization)
    """
    layers = []

    def __init__(self, input_shape, activation=linear,
                 epsilon=1e-4, alpha=0.1, name=None):
        super(BatchNormLayer, self).__init__()

        if len(input_shape) == 2:
            self.axes = (0,)
            shape = [input_shape[0]]
        elif len(input_shape) == 4:
            self.axes = (0, 2, 3)
            shape = [input_shape[1]]
        else:
            raise NotImplementedError

        self.name = 'BN' if name is None else name
        self.epsilon = epsilon
        self.alpha = alpha
        self.deterministic = False
        self.update_averages = True
        self.activation = activation
        self.act_name = activation.__name__
        self.input_shape = input_shape

        self.gamma = theano.shared(np.ones(shape, dtype=theano.config.floatX),
                                   name=name + '_G', borrow=True)
        self.beta = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  name=name + '_B', borrow=True)

        self.mean = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  name=name + '_mean', borrow=True)
        self.std = theano.shared(np.ones(shape, dtype=theano.config.floatX),
                                 name=name + '_std', borrow=True)

        self.params = [self.gamma, self.beta]
        self.statistics = [self.mean, self.std]
        BatchNormLayer.layers.append(self)

        # Show information
        print('  # %s (BN_T) ' % (self.name), end='')
        print('act.: %s,' % self.act_name)

    def get_output(self, input, **kwargs):
        input_mean = input.mean(self.axes)
        input_std = T.sqrt(input.var(self.axes) + self.epsilon)

        # Decide whether to use the stored averages or mini-batch statistics
        use_averages = self.deterministic
        if use_averages:
            mean = self.mean
            std = self.std
        else:
            mean = input_mean
            std = input_std

        # Decide whether to update the stored averages
        update_averages = self.update_averages and not use_averages
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * input_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            std += 0 * running_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(list(range(input.ndim - len(self.axes))))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        std = std.dimshuffle(pattern)

        # normalize
        normalized = bn.batch_normalization(input, gamma, beta, mean, std)
        return self.activation(normalized)

    def get_out_shape(self):
        return self.input_shape

    def reset_stats(self):
        # reset mean and std
        self.mean.set_value(np.zeros(self.mean.get_value().shape,
                                     dtype=theano.config.floatX))
        self.std.set_value(np.ones(self.std.get_value().shape,
                                   dtype=theano.config.floatX))

    def get_stats(self):
        return (self.mean, self.std)

    @staticmethod
    def set_batch_norms_training(training):
        deterministic = False if training else True
        print(' - Batch norm layres: deterministic =', deterministic)
        for layer in BatchNormLayer.layers:
            layer.deterministic = deterministic
            layer.update_averages = not deterministic

    @staticmethod
    def reset_batch_norms_stats():
        print(' - Batch norm layres: reset mean and std')
        for layer in BatchNormLayer.layers:
            layer.reset_stats()


class BatchNormLayer_L(Layer):
    """
    Batch normalization layer.
    Core algorithm is brought from Lasagne.
    http://lasagne.readthedocs.io/en/latest/modules/layers/normalization.html
    """
    layers = []

    def __init__(self, input_shape, activation=linear,
                 epsilon=1e-4, alpha=0.1, name=None):
        super(BatchNormLayer, self).__init__()

        if len(input_shape) == 2:
            self.axes = (0,)
            shape = [input_shape[0]]
        elif len(input_shape) == 4:
            self.axes = (0, 2, 3)
            shape = [input_shape[1]]
        else:
            raise NotImplementedError

        self.name = 'BN' if name is None else name
        self.epsilon = epsilon
        self.alpha = alpha
        self.deterministic = False
        self.update_averages = True
        self.activation = activation
        self.act_name = activation.__name__
        self.input_shape = input_shape

        self.gamma = theano.shared(np.ones(shape, dtype=theano.config.floatX),
                                   name=name + '_G', borrow=True)
        self.beta = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  name=name + '_B', borrow=True)

        self.mean = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  name=name + '_mean', borrow=True)
        self.invstd = theano.shared(np.ones(shape, dtype=theano.config.floatX),
                                    name=name + '_invstd', borrow=True)

        self.params = [self.gamma, self.beta]
        self.statistics = [self.mean, self.invstd]
        BatchNormLayer.layers.append(self)

        # Show information
        print('  # %s (BN_L) ' % (self.name), end='')
        print('act.: %s,' % self.act_name)

    def get_output(self, input, **kwargs):
        input_mean = input.mean(self.axes)
        input_invstd = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        use_averages = self.deterministic
        if use_averages:
            mean = self.mean
            invstd = self.invstd
        else:
            mean = input_mean
            invstd = input_invstd

        # Decide whether to update the stored averages
        update_averages = self.update_averages and not use_averages
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_invstd = theano.clone(self.invstd, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = (
                (1 - self.alpha) * running_mean + self.alpha * input_mean)
            running_invstd.default_update = (
                (1 - self.alpha) * running_invstd + self.alpha * input_invstd)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            invstd += 0 * running_invstd

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(list(range(input.ndim - len(self.axes))))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        invstd = invstd.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * invstd) + beta
        return self.activation(normalized)

    def get_out_shape(self):
        return self.input_shape

    def reset_stats(self):
        # reset mean and invstd
        self.mean.set_value(np.zeros(self.mean.get_value().shape,
                                     dtype=theano.config.floatX))
        self.invstd.set_value(np.ones(self.invstd.get_value().shape,
                                      dtype=theano.config.floatX))

    def get_stats(self):
        return (self.mean, self.invstd)

    @staticmethod
    def set_batch_norms_training(training):
        deterministic = False if training else True
        print(' - Batch norm layres: deterministic =', deterministic)
        for layer in BatchNormLayer.layers:
            layer.deterministic = deterministic
            layer.update_averages = not deterministic

    @staticmethod
    def reset_batch_norms_stats():
        print(' - Batch norm layres: reset mean and invstd')
        for layer in BatchNormLayer.layers:
            layer.reset_stats()
