from __future__ import absolute_import, division, print_function

# import theano.tensor as T
import numpy as np
import theano
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = (k / k.sum()).reshape((1, 1, 5, 5))
kern = theano.shared(k5x5, borrow=True)

k5x5_3ch = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
k5x5_3ch = k5x5_3ch.transpose([2, 3, 0, 1])
kern_3ch = theano.shared(k5x5_3ch, borrow=True)


def downsample_img(img, n_ch=1):
    if n_ch == 1:
        kernel = kern
        filter_shape = [1, 1, 5, 5]
    elif n_ch == 3:
        kernel = kern_3ch
        filter_shape = [3, 3, 5, 5]
    else:
        raise NotImplementedError
    return conv2d(img, kernel, filter_shape=filter_shape,
                  border_mode='half', subsample=(2, 2))


def upsample_img(img, out_shape, n_ch=1):
    if n_ch == 1:
        kernel = kern * 4
        filter_shape = [1, 1, 5, 5]
    elif n_ch == 3:
        kernel = kern_3ch * 4
        filter_shape = [3, 3, 5, 5]
    else:
        raise NotImplementedError
    return conv2d_tr_half(img, kernel, filter_shape=filter_shape,
                          input_shape=out_shape, subsample=(2, 2))


def conv2d_tr_half(output, filters, filter_shape, input_shape,
                   subsample=(1, 1)):
    input = conv2d_grad_wrt_inputs(
        output, filters,
        input_shape=(None, filter_shape[0], input_shape[2], input_shape[3]),
        filter_shape=filter_shape, border_mode='half', subsample=subsample)
    return input


def lap_split(img, n_ch=1):
    '''Split the image into lo and hi frequency components'''
    lo = downsample_img(img, n_ch)
    lo2 = upsample_img(lo, img.shape, n_ch)
    hi = img - lo2
    return lo, hi


def gen_lpyr(img, n_level, n_ch=1):
    '''Build Laplacian pyramid with n_level splits'''
    l_pyr = []
    for i in range(n_level - 1):
        img, hi = lap_split(img, n_ch)
        l_pyr.append(hi)
    l_pyr.append(img)
    return l_pyr


def gen_gpyr(img, n_level, n_ch=1):
    """Generate a Gaussian pyramid."""
    g_pyr = []
    g_pyr.append(img)
    for idx in range(n_level - 1):
        g_pyr.append(downsample_img(g_pyr[idx], n_ch))
    return g_pyr


def merge_lpyr(l_pyr, n_ch=1):
    '''Merge Laplacian pyramid'''
    l_pyr = l_pyr[::-1]
    img = l_pyr[0]
    for hi in l_pyr[1:]:
        img = upsample_img(img, hi.shape, n_ch) + hi
    return img


def normalize_lowpass_subt(img, n_level, n_ch=1):
    '''Normalize image by subtracting the low-pass-filtered image'''
    # Downsample
    img_ = img
    pyr_sh = []
    for i in range(n_level - 1):
        pyr_sh.append(img_.shape)
        img_ = downsample_img(img_, n_ch)

    # Upsample
    for i in range(n_level - 1):
        img_ = upsample_img(img_, pyr_sh[n_level - 2 - i], n_ch)
    return img - img_


def get_hi_lo_lap(img, n_level, n_ch=1):
    '''Normalize image by subtracting the low-pass-filtered image'''
    # Downsample
    img_ = img
    pyr_sh = []
    for i in range(n_level - 1):
        pyr_sh.append(img_.shape)
        img_ = downsample_img(img_, n_ch)
    lo = img_

    # Upsample
    for i in range(n_level - 1):
        img_ = upsample_img(img_, pyr_sh[n_level - 2 - i], n_ch)
    return img - img_, lo


def get_lowfreq_upscale(l_pyr, n_ch=1):
    n_level = len(l_pyr)
    lf = l_pyr[-1]

    # Upsample
    for i in range(n_level - 1):
        lf = upsample_img(lf, l_pyr[n_level - 2 - i].shape, n_ch)
    return lf


# def downsample_img(img, n_ch=1):
#     """Downsample an image by 2 by 2"""
#     if n_ch == 1:
#         output = conv2d(img, kern, filter_shape=(1, 1, 5, 5),
#                         border_mode='half')
#     elif n_ch > 1:
#         conv_outs = []
#         for ch in range(n_ch):
#             cur_ch = img[:, ch, :, :].dimshuffle(0, 'x', 1, 2)
#             conv_outs.append(conv2d(cur_ch, kern, filter_shape=(1, 1, 5, 5),
#                                     border_mode='half'))
#         output = T.concatenate(conv_outs, axis=1)
#     else:
#         raise NotImplementedError
#     return output[:, :, ::2, ::2]


# def upsample_img(img, out_shape, n_ch=1):
#     """Upsample an image by 2 by 2"""
#     img_up = img.repeat(2, axis=2).repeat(2, axis=3)
#     if n_ch == 1:
#         output = conv2d(img_up, kern, filter_shape=(1, 1, 5, 5),
#                         border_mode='half')
#     elif n_ch > 1:
#         conv_outs = []
#         for ch in range(n_ch):
#             cur_ch = img_up[:, ch, :, :].dimshuffle(0, 'x', 1, 2)
#             conv_outs.append(conv2d(cur_ch, kern, filter_shape=(1, 1, 5, 5),
#                                     border_mode='half'))
#         output = T.concatenate(conv_outs, axis=1)
#     else:
#         raise NotImplementedError
#     return output[:, :, :out_shape[2], :out_shape[3]]
