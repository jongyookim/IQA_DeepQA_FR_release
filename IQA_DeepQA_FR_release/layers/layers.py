from __future__ import absolute_import, division, print_function

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


# Activation function wrappers
def linear(x):
    return x


def tanh(x):
    """ Hyperbolic tangent """
    return T.tanh(x)


def sigm(x):
    """ Sigmoid """
    return T.nnet.sigmoid(x)


def relu(x, alpha=0.0):
    """ Rectified linear unit """
    return T.nnet.relu(x, alpha)


def lrelu(x, alpha=0.1):
    """ Leaky ReLU """
    return T.nnet.relu(x, alpha)


def elu(x, alpha=1.0):
    """ Exponential LU """
    return T.nnet.elu(x, alpha)


##############################################################################
class Layer(object):
    """
    Base class for layers
    """
    # init_rng = np.random.RandomState(1235)
    init_rng = np.random.RandomState()

    def __init__(self):
        self.params = []
        self.rng = Layer.init_rng

    def get_params(self):
        return self.params

    def get_output(self, input, **kwargs):
        raise NotImplementedError("get_output")

    def get_out_shape(self):
        return None

    def init_he(self, shape, activation, sampling='uniform', lrelu_alpha=0.1):
        # He et al. 2015
        if activation in [T.nnet.relu, relu, elu]:  # relu or elu
            gain = np.sqrt(2)
        elif activation == lrelu:  # lrelu
            gain = np.sqrt(2 / (1 + lrelu_alpha ** 2))
        else:
            gain = 1.0

        # len(shape) == 2 -> fully-connected layers
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])

        if sampling == 'normal':
            std = gain * np.sqrt(1. / fan_in)
            return np.asarray(self.rng.normal(0., std, shape),
                              dtype=theano.config.floatX)
        elif sampling == 'uniform':
            bound = gain * np.sqrt(3. / fan_in)
            return np.asarray(self.rng.uniform(-bound, bound, shape),
                              dtype=theano.config.floatX)
        else:
            raise NotImplementedError


##############################################################################
# Neural Network Layers
class FCLayer(Layer):
    """
    Fully connected layer.

    Parameters
    ----------
    input_shape: int or a tuple of ints
        Input feature dimension or (batch_size, input feature dimension)
    n_out: int
        Output feature dimension.
    activation: function
        Activation function.
    W: tensor, numpy or None
        Filter weights. If this is not given, the weight is initialized by
        random values.
    b: tensor, numpy or None
        Biases. If this is not given, the weight is initialized by
        random values.
    """
    def __init__(self, input_shape, n_out, activation=linear,
                 W=None, b=None, no_bias=False, name=None):
        super(FCLayer, self).__init__()

        if isinstance(input_shape, (list, tuple)):
            self.input_shape = input_shape
        else:
            self.input_shape = (None, input_shape)
        self.n_in = self.input_shape[1]
        self.n_out = n_out
        self.activation = activation
        self.act_name = activation.__name__
        self.no_bias = no_bias
        self.name = 'FC' if name is None else name

        self.params = []
        if isinstance(W, T.sharedvar.TensorSharedVariable):
            self.W = W
        else:
            if W is None:
                W_values = self.init_he(
                    (self.n_in, self.n_out), self.activation)
            else:
                W_values = W
            self.W = theano.shared(W_values, self.name + '_W', borrow=True)
            self.params += [self.W]

        if self.no_bias:
            self.b = None
        elif isinstance(b, T.sharedvar.TensorSharedVariable):
            self.b = b
        else:
            if b is None:
                b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            else:
                b_values = b
            self.b = theano.shared(b_values, self.name + '_b', borrow=True)
            self.params += [self.b]

        # Show information
        print('  # %s (FC): in = %d -> out = %d,' % (
            self.name, self.n_in, self.n_out), end=' ')
        print('act.: %s,' % self.act_name, end=' ')
        if self.no_bias:
            print('No bias')
        else:
            print('')

    def get_output(self, input, **kwargs):
        lin_output = T.dot(input, self.W)
        if not self.no_bias:
            lin_output += self.b
        return self.activation(lin_output)

    def get_out_shape(self):
        return (self.input_shape[0], self.n_out)


class ConvLayer(Layer):
    """
    Convolutional layer.

    Parameters
    ----------
    input_shape: a tuple of ints
        (batch size, num input feature maps, image height, image width)
    num_filts int
        Number of output channels.
    filt_size: a tuple of ints
        (filter rows, filter columns)
    activation: function
        Activation function.
    mode: ``'half'``, ``'valid'`` or ``'full'``
        Border mode of convolution.
    subsample: a tuple of ints (len = 2)
        Stide of convolution.
    filter_dilation: a tuple of ints (len = 2)
        Dilation of convolution.
    W: tensor, numpy or None
        Filter weights. If this is not given, the weight is initialized by
        random values.
    b: tensor, numpy or None
        Biases. If this is not given, the weight is initialized by
        random values.
    no_bias: bool
        If True, bias is not used in this layer.
    """
    def __init__(self, input_shape, num_filts, filt_size, activation=linear,
                 mode='half', subsample=(1, 1), filter_dilation=(1, 1),
                 W=None, b=None, no_bias=False, name=None):
        super(ConvLayer, self).__init__()

        # Make filter shape
        assert len(filt_size) == 2
        filter_shape = (num_filts, input_shape[1]) + filt_size

        # Calculate output shape and validate
        if isinstance(mode, tuple):
            self.mode = mode
            self.out_size = [
                input_shape[i] - filter_shape[i] + 2 * self.mode[i - 2] + 1
                for i in range(2, len(input_shape))]
        else:
            self.mode = mode.lower()
            if self.mode == 'valid':
                self.out_size = [input_shape[i] - filter_shape[i] + 1
                                 for i in range(2, len(input_shape))]
            elif self.mode == 'half':
                self.out_size = input_shape[2:]
            elif self.mode == 'full':
                self.out_size = [input_shape[i] - filter_shape[i] - 1
                                 for i in range(2, len(input_shape))]
            else:
                raise ValueError('Invalid mode: %s' % self.mode)
        self.out_size = tuple(self.out_size)
        for sz in self.out_size:
            if sz < 1:
                raise ValueError('Invalid feature size: (%s).' %
                                 ', '.join([str(i) for i in self.out_size]))

        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.activation = activation
        self.act_name = activation.__name__
        self.no_bias = no_bias
        self.name = 'Conv' if name is None else name
        self.subsample = subsample
        self.filter_dilation = filter_dilation

        # Initialize parameters
        self.params = []
        if isinstance(W, T.sharedvar.TensorSharedVariable):
            self.W = W
        else:
            if W is None:
                W_values = self.init_he(filter_shape, self.activation)
            else:
                W_values = W
            self.W = theano.shared(W_values, self.name + '_W',
                                   borrow=True)
            self.params += [self.W]

        if self.no_bias:
            self.b = None
        elif isinstance(b, T.sharedvar.TensorSharedVariable):
            self.b = b
        else:
            if b is None:
                b_values = np.zeros((filter_shape[0],),
                                    dtype=theano.config.floatX)
            else:
                b_values = b
            self.b = theano.shared(b_values, self.name + '_b',
                                   borrow=True)
            self.params += [self.b]

        # Show information
        print('  # %s (Conv-%s):' % (name, mode), end=' ')
        print('flt.(%s),' % ', '.join(
            [str(i) for i in self.filter_shape]), end=' ')
        print('in.(%s),' % ', '.join(
            [str(i) for i in self.input_shape[1:]]), end=' ')
        print('act.: %s,' % self.act_name, end=' ')
        if self.no_bias:
            print('No bias')
        else:
            print('')
        if self.subsample != (1, 1):
            print('    subsample (%s) -> (%s)' % (
                ', '.join([str(i) for i in self.input_shape[1:]]),
                ', '.join([str(i) for i in self.get_out_shape()[1:]])))

    def get_output(self, input, **kwargs):
        var_shape = kwargs.get('var_shape', False)

        lin_output = conv2d(
            input=input,
            filters=self.W,
            input_shape=None if var_shape else self.input_shape,
            filter_shape=self.filter_shape,
            border_mode=self.mode,
            subsample=self.subsample,
            filter_dilation=self.filter_dilation
        )

        if not self.no_bias:
            lin_output += self.b.dimshuffle('x', 0, 'x', 'x')

        return self.activation(lin_output)

    def get_out_shape(self, after_ss=True):
        out_size = self.out_size
        if after_ss:
            out_size = [(out_size[i] + self.subsample[i] - 1) //
                        self.subsample[i] for i in range(len(out_size))]

        return (self.input_shape[0], self.filter_shape[0]) + tuple(out_size)


class ConvGradLayer(Layer):
    """
    Transposed convolutional layer.

    Parameters
    ----------
    out_shape: a tuple of ints
        (batch size, num output feature maps, image height, image width)
    num_in_feat: int
        Number input feature maps.
    filt_size: a tuple of ints
        (filter rows, filter columns)
    activation: function
        Activation function.
    mode: ``'half'``, ``'valid'`` or ``'full'``
        Border mode of convolution in the forward path.
    subsample: a tuple of ints (len = 2)
        Stide of convolution in the forward path.
    filter_dilation: a tuple of ints (len = 2)
        Dilation of convolution in the forward path.
    W: tensor, numpy or None
        Filter weights. If this is not given, the weight is initialized by
        random values.
    b: tensor, numpy or None
        Biases. If this is not given, the weight is initialized by
        random values.
    no_bias: bool
        If True, bias is not used in this layer.
    """
    def __init__(self, out_shape, num_in_feat, filt_size, activation=linear,
                 mode='half', subsample=(1, 1), filter_dilation=(1, 1),
                 W=None, b=None, no_bias=False, name=None):
        super(ConvGradLayer, self).__init__()

        # Make filter shape
        assert len(filt_size) == 2
        filter_shape = (out_shape[1], num_in_feat) + filt_size

        self.mode = mode.lower()
        self.filter_shape = filter_shape
        self.out_shape = out_shape
        self.activation = activation
        self.act_name = activation.__name__
        self.no_bias = no_bias
        self.name = 'ConvGr' if name is None else name
        self.subsample = subsample
        self.filter_dilation = filter_dilation

        # Initialize parameters
        self.params = []
        if isinstance(W, T.sharedvar.TensorSharedVariable):
            self.W = W
        else:
            if W is None:
                W_values = self.init_he(filter_shape, self.activation)
            else:
                W_values = W
            self.W = theano.shared(W_values, self.name + '_W',
                                   borrow=True)
            self.params += [self.W]

        if self.no_bias:
            self.b = None
        elif isinstance(b, T.sharedvar.TensorSharedVariable):
            self.b = b
        else:
            if b is None:
                b_values = np.zeros((filter_shape[0],),
                                    dtype=theano.config.floatX)
            else:
                b_values = b
            self.b = theano.shared(b_values, self.name + '_b',
                                   borrow=True)
            self.params += [self.b]

        # Show information
        print('  # %s (ConvGr-%s):' % (name, mode), end=' ')
        print('flt.(%s),' % ', '.join(
            [str(i) for i in self.filter_shape]), end=' ')
        print('out.(%s),' % ', '.join(
            [str(i) for i in self.out_shape[1:]]), end=' ')
        print('act.: %s,' % self.act_name, end=' ')
        if self.no_bias:
            print('No bias')
        else:
            print('')
        if self.subsample != (1, 1):
            print('    upsample -> (%s)' % (
                ', '.join([str(i) for i in self.out_shape[1:]])))

    def get_output(self, input, **kwargs):
        lin_output = conv2d_grad_wrt_inputs(
            output_grad=input,
            filters=self.W,
            input_shape=self.out_shape,
            filter_shape=self.filter_shape,
            border_mode=self.mode,
            subsample=self.subsample,
            # filter_flip=True,
            filter_dilation=self.filter_dilation
        )
        if not self.no_bias:
            lin_output += self.b.dimshuffle('x', 0, 'x', 'x')

        return self.activation(lin_output)

    def get_out_shape(self, **kwargs):
        return self.out_shape


class ActivationLayer(Layer):
    """
    Activation layer (no weights and bias).

    Parameters
    ----------
    activation: function
        Activation function.
    """
    def __init__(self, activation=linear, name=None):
        super(ActivationLayer, self).__init__()

        self.activation = activation
        self.act_name = activation.__name__
        self.name = 'Act' if name is None else name

        # Show information
        print('  # %s (Act.)' % (self.name), end=' ')
        print('act.: %s,' % self.act_name)

    def get_output(self, input, **kwargs):
        return self.activation(input)


class BiasLayer(Layer):
    """
    Bias layer (no weights).

    Parameters
    ----------
    input_shape: int or a tuple of ints
        Input feature dimension or (batch_size, input feature dimension)
    axis: int
        Axis of input to add the bias.
    activation: function
        Activation function.
    b: tensor, numpy or None
        Biases. If this is not given, the weight is initialized by
        random values.
    """
    def __init__(self, input_shape, axis=1, activation=linear,
                 b=None, name=None):
        super(BiasLayer, self).__init__()

        self.input_shape = input_shape
        self.axis = axis
        self.activation = activation
        self.name = 'Bias' if name is None else name
        self.act_name = activation.__name__

        if isinstance(input_shape, (list, tuple)):
            self.bias_sh = (input_shape[self.axis],)
            self.in_dim = len(input_shape)
        else:
            self.bias_sh = (input_shape,)
            self.in_dim = 2

        if isinstance(b, T.sharedvar.TensorSharedVariable):
            self.b = b
        else:
            if b is None:
                b_values = np.zeros(self.bias_sh, dtype=theano.config.floatX)
            else:
                b_values = b
            self.b = theano.shared(b_values, self.name + '_b', borrow=True)
            self.params += [self.b]

        # Show information
        print('  # %s (Bias)' % (self.name), end=' ')
        if self.in_dim > 2:
            print('in.(%s),' % ', '.join(
                [str(i) for i in self.input_shape[1:]]), end=' ')
        else:
            print('in.(%d),' % self.input_shape, end=' ')
        print('bias dim:%d,' % self.axis, end=' ')
        print('act.: %s,' % self.act_name)

    def get_output(self, input, **kwargs):
        if self.in_dim > 2:
            pattern = [0 if ii == self.axis else 'x'
                       for ii in range(self.in_dim)]
            lin_output = input + self.b.dimshuffle(pattern)
        else:
            lin_output = input + self.b
        return self.activation(lin_output)

    def get_out_shape(self):
        return self.input_shape


class TensorToVectorLayer(Layer):
    """
    Converts 4D tensor to 2D tensor.

    Parameters
    ----------
    input_shape: a tuple of ints
        (batch size, num input feature maps, image height, image width)
    """
    def __init__(self, input_shape):
        super(TensorToVectorLayer, self).__init__()

        self.input_shape = input_shape
        print('  # tensor to vector: (%s) -> %d' % (
            ', '.join([str(i) for i in self.input_shape[1:]]),
            np.prod(self.input_shape[1:])))

    def get_output(self, input, **kwargs):
        return input.flatten(2)

    def get_out_shape(self):
        return (self.input_shape[0], np.prod(self.input_shape[1:]))


class VectorToTensorLayer(Layer):
    """
    Converts 2D tensor to 4D tensor.

    Parameters
    ----------
    output_shape: a tuple of ints
        (batch size, num output feature maps, image height, image width)
    """
    def __init__(self, output_shape):
        super(VectorToTensorLayer, self).__init__()

        self.output_shape = output_shape
        print('  # vector to tensor: (%s)' % ', '.join(
            [str(i) for i in self.output_shape[1:]]))

    def get_output(self, input, **kwargs):
        # output_shape = (T.shape(input)[0], ) + self.output_shape[1:]
        output_shape = (-1,) + self.output_shape[1:]
        return input.reshape(output_shape)

    def get_out_shape(self):
        return self.output_shape


##############################################################################


class UpsampleLayer(Layer):
    """
    Upscale the input by a specified factor.

    Parameters
    ----------
    mode: {``'zero'``, ``'NN'``}
        Put zeros or nearest neigbor pixels between original pixels.
    """
    def __init__(self, input_shape, us=(2, 2), out_shape=None, mode='zero'):
        super(UpsampleLayer, self).__init__()

        self.input_shape = input_shape
        self.us = us
        self.mode = mode
        self.out_shape = out_shape
        print('  # upsample-(%s)-%s (%s) -> (%s)' % (
            mode,
            ', '.join([str(i) for i in self.us]),
            ', '.join([str(i) for i in self.input_shape[1:]]),
            ', '.join([str(i) for i in self.get_out_shape()[1:]])))

    def get_output(self, input, **kwargs):
        us = self.us
        if self.mode == 'zero':
            sh = input.shape
            upsample = T.zeros((sh[0], sh[1], sh[2] * us[0], sh[3] * us[1]),
                               dtype=input.dtype)
            out = T.set_subtensor(upsample[:, :, ::us[0], ::us[1]], input)

        elif self.mode == 'NN':
            out = input.repeat(us[0], axis=2).repeat(us[1], axis=3)

        else:
            raise ValueError('Select the proper mode: zero / NN')

        return out

    def get_out_shape(self):
        in_sh = self.input_shape
        out_len0 = in_sh[2] * self.us[0]
        out_len1 = in_sh[3] * self.us[1]
        return (in_sh[0], in_sh[1], out_len0, out_len1)


class Pool2DLayer(Layer):
    """
    Downscale the input by a specified factor.

    Parameters
    ----------
    input_shape: a tuple of ints
        (batch size, num input feature maps, image height, image width)
    pool_size: tuple of length 2 or theano vector of ints of size 2.
        Factor by which to downscale (vertical ws, horizontal ws).
        (2,2) will halve the image in each dimension.
    pad: tuple of two ints - (pad_h, pad_w),
        pad zeros to extend beyond four borders of the images,
        pad_h is the size of the top and bottom margins,
        and pad_w is the size of the left and right margins.
    ignore_border: bool
        (default None, will print a warning and set to False)
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    mode: {``'max'``, ``'sum'``, ``'average_inc_pad'``, ``'average_exc_pad'``}
    """
    def __init__(self, input_shape, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='max'):
        super(Pool2DLayer, self).__init__()

        self.input_shape = input_shape
        self.pool_size = pool_size

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = stride

        self.pad = pad

        self.ignore_border = ignore_border
        self.mode = mode
        print('  # Pool-%s (%s) -> (%s)' % (
            mode,
            ', '.join([str(i) for i in self.input_shape[1:]]),
            ', '.join([str(i) for i in self.get_out_shape()[1:]])))

    def get_output(self, input, **kwargs):
        pooled = pool_2d(input,
                         ws=self.pool_size,
                         stride=self.stride,
                         ignore_border=self.ignore_border,
                         pad=self.pad,
                         mode=self.mode,
                         )
        return pooled

    def get_out_shape(self):
        output_shape = list(self.input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(self.input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border)

        output_shape[3] = pool_output_length(self.input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border)

        return tuple(output_shape)


def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length


##############################################################################
# Dropout
class DropoutLayer(Layer):
    """
    Conducts Dropout.
    """
    layers = []

    def __init__(self, p=0.5, rescale=True):
        super(DropoutLayer, self).__init__()

        self._srng = RandomStreams(self.rng.randint(1, 2147462579))
        self.p = p
        self.rescale = rescale
        self.deterministic = False
        DropoutLayer.layers.append(self)
        print('  # Dropout: p = %.2f' % (self.p))

    def get_output(self, input, **kwargs):
        if self.deterministic or self.p == 0:
            return input
        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1)
            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            return input * self._srng.binomial(input.shape, p=retain_prob,
                                               dtype=input.dtype)

    @staticmethod
    def set_dropout_training(training):
        deterministic = False if training else True
        # print(' - Dropout layres: deterministic =', deterministic)
        for layer in DropoutLayer.layers:
            layer.deterministic = deterministic
