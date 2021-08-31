# -*- coding: utf-8 -*-
#
# File : echotorch/timetensor.py
# Description : A special tensor with a time dimension
# Date : 25th of January, 2021
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

# Imports
from typing import Optional, Tuple, Union, List, Callable, Any
import torch
import numpy as np
import warnings

# EchoTorch imports
import echotorch
from .base_tensors import BaseTensor


# Error
ERROR_TENSOR_TO_SMALL = "Time dimension does not exists in the data tensor " \
                        "(time dim at {}, {} dimension in tensor). The minimum tensor size " \
                        "is {}"
ERROR_TIME_LENGTHS_TOO_BIG = "There is time lengths which are bigger than the actual tensor data"
ERROR_WRONG_TIME_LENGTHS_SIZES = "The sizes of the time lengths tensor should be {}"
ERROR_TIME_DIM_NEGATIVE = "The index of the time-dimension cannot be negative"

# Torch overridable methods
# TORCH_OPS = [
#     'abs', 'absolute', 'adaptive_avg_pool1d', 'adaptive_max_pool1d', 'acos', 'arccos', 'acosh', 'arccosh', 'add',
#     'addbmm', 'addcdiv', 'addcmul', 'addmm', 'addmv', 'addr', 'affine_grid_generator', 'all', 'allclose',
#     'alpha_dropout', 'amax', 'amin', 'angle', 'any', 'argmax', 'argmin', 'argsort', 'asin', '_assert_async', 'arcsin',
#     'asinh', 'arcsinh', 'atan', 'arctan', 'atan2', 'atanh', 'arctanh', 'atleast_1d', 'atleast_2d', 'atleast_3d',
#     'avg_pool1d', 'baddbmm', 'batch_norm', 'batch_norm_backward_elemt', 'batch_norm_backward_reduce',
#     'batch_norm_elemt', 'batch_norm_gather_stats', 'batch_norm_gather_stats_with_counts', 'batch_norm_stats',
#     'batch_norm_update_stats', 'bernoulli', 'bilinear', 'binary_cross_entropy_with_logits', 'bincount', 'binomial',
#     'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'block_diag', 'bmm', 'broadcast_tensors',
#     'broadcast_to', 'bucketize', 'cartesian_prod', 'cat', 'cdist', 'ceil', 'celu', 'chain_matmul', 'channel_shuffle',
#     'cholesky', 'linalg_cholesky', 'linalg_cholesky_ex', 'cholesky_inverse', 'cholesky_solve',
#     'choose_qparams_optimized', 'chunk', 'clamp', 'clip', 'clamp_min', 'clamp_max', 'column_stack', 'clone',
#     'combinations', 'complex', 'copysign', 'polar', 'linalg_cond', 'conj', 'constant_pad_nd', 'conv1d', 'conv2d',
#     'conv3d', 'convolution', 'conv_tbc', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'cos',
#     'cosine_embedding_loss', 'cosh', 'cosine_similarity', 'count_nonzero', 'cross', 'ctc_loss', 'cummax', 'cummin',
#     'cumprod', 'cumsum', 'logcumsumexp', 'deg2rad', 'dequantize', 'det', 'linalg_det', 'detach', 'diag', 'diag_embed',
#     'diagflat', 'diff', 'diagonal', 'digamma', 'dist', 'div', 'divide', 'dot', 'dropout', 'dsmm', 'hsmm', 'dsplit',
#     'dstack', 'eig', 'linalg_eig', 'linalg_eigvals', 'linalg_eigh', 'linalg_eigvalsh', 'einsum', 'embedding',
#     'embedding_bag', 'empty_like', 'eq', 'equal', 'erf', 'erfc', 'erfinv', 'exp', 'exp2', 'expm1',
#     'fake_quantize_per_channel_affine', 'fake_quantize_per_tensor_affine', 'fbgemm_linear_fp16_weight',
#     'fbgemm_linear_fp16_weight_fp32_activation', 'fbgemm_linear_int8_weight',
#     'fbgemm_linear_int8_weight_fp32_activation', 'fbgemm_linear_quantize_weight', 'fbgemm_pack_gemm_matrix_fp16',
#     'fbgemm_pack_quantized_matrix', 'feature_alpha_dropout', 'feature_dropout', 'fft_fft', 'fft_ifft', 'fft_rfft',
#     'fft_irfft', 'fft_hfft', 'fft_ihfft', 'fft_fftn', 'fft_ifftn', 'fft_rfftn', 'fft_irfftn', 'fft_fft2', 'fft_ifft2',
#     'fft_rfft2', 'fft_irfft2', 'fft_fftshift', 'fft_ifftshift', 'fix', 'flatten', 'flip', 'fliplr', 'flipud',
#     'frobenius_norm', 'floor', 'floor_divide', 'float_power', 'fmod', 'frac', 'frexp', 'full_like', 'lu_unpack',
#     'gather', 'gcd', 'ge', 'greater_equal', 'geqrf', 'i0', 'inner', 'outer', 'ger', 'gradient', 'grid_sampler',
#     'grid_sampler_2d', 'grid_sampler_3d', 'group_norm', 'gru', 'gru_cell', 'gt', 'greater', 'hardshrink', 'heaviside',
#     'hinge_embedding_loss', 'histc', 'linalg_householder_product', 'hsplit', 'hstack', 'hypot', 'igamma', 'igammac',
#     'imag', 'index_add', 'index_copy', 'index_put', 'index_select', 'index_fill', 'isfinite', 'isinf', 'isreal',
#     'isposinf', 'isneginf', 'instance_norm', 'int_repr', 'inverse', 'linalg_inv', 'linalg_inv_ex', 'is_complex',
#     'is_distributed', 'is_floating_point', 'is_nonzero', 'is_same_size', 'is_signed', 'isclose', 'isnan', 'istft',
#     'kl_div', 'kron', 'kthvalue', 'layer_norm', 'lcm', 'ldexp', 'le', 'less_equal', 'lerp', 'lgamma', 'lobpcg', 'log',
#     'log_softmax', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logdet', 'xlogy', 'logical_and',
#     'logical_not', 'logical_or', 'logical_xor', 'logsumexp', 'logit', 'lstm', 'lstm_cell', 'lstsq', 'lt', 'less',
#     'lu', 'lu_solve', 'margin_ranking_loss', 'masked_fill', 'masked_scatter', 'masked_select', 'matmul',
#     'matrix_power', 'linalg_matrix_power', 'matrix_rank', 'linalg_matrix_rank', 'linalg_multi_dot', 'matrix_exp',
#     'max', 'maximum', 'fmax', 'max_pool1d', 'max_pool2d', 'max_pool3d', 'max_pool1d_with_indices', 'mean', 'median',
#     'nanmedian', 'meshgrid', 'min', 'minimum', 'fmin', 'miopen_batch_norm', 'miopen_convolution',
#     'miopen_convolution_transpose', 'miopen_depthwise_convolution', 'miopen_rnn', 'mode', 'movedim', 'moveaxis',
#     'msort', 'mul', 'multiply', 'multinomial', 'mv', 'mvlgamma', 'narrow', 'narrow_copy', 'nan_to_num',
#     'native_batch_norm', 'native_layer_norm', 'native_group_norm', 'native_norm', 'ne', 'not_equal', 'neg',
#     'negative', 'nextafter', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d', 'adaptive_max_pool1d',
#     'adaptive_max_pool1d_with_indices', 'adaptive_max_pool2d', 'adaptive_max_pool2d_with_indices',
#     'adaptive_max_pool3d', 'adaptive_max_pool3d_with_indices', 'affine_grid', 'alpha_dropout', 'avg_pool2d',
#     'avg_pool3d', 'batch_norm', 'bilinear', 'binary_cross_entropy', 'binary_cross_entropy_with_logits', 'celu',
#     'cosine_embedding_loss', 'cross_entropy', 'ctc_loss', 'dropout', 'dropout2d', 'dropout3d', 'elu', 'embedding',
#     'embedding_bag', 'feature_alpha_dropout', 'fold', 'fractional_max_pool2d', 'fractional_max_pool2d_with_indices',
#     'fractional_max_pool3d', 'fractional_max_pool3d_with_indices', 'gaussian_nll_loss', 'gelu', 'glu', 'grid_sample',
#     'group_norm', 'gumbel_softmax', 'hardshrink', 'hardtanh', 'hinge_embedding_loss', 'instance_norm', 'interpolate',
#     'kl_div', 'l1_loss', 'layer_norm', 'leaky_relu', 'linear', 'local_response_norm', 'log_softmax', 'log_sigmoid',
#     'lp_pool1d', 'lp_pool2d', 'margin_ranking_loss', 'max_pool1d', 'max_pool1d_with_indices', 'max_pool2d',
#     'max_pool2d_with_indices', 'max_pool3d', 'max_pool3d_with_indices', 'max_unpool1d', 'max_unpool2d', 'max_unpool3d',
#     'mse_loss', 'multi_head_attention_forward', 'multi_margin_loss', 'multilabel_margin_loss',
#     'multilabel_soft_margin_loss', 'nll_loss', 'normalize', 'one_hot', '_pad', 'pairwise_distance',
#     'poisson_nll_loss', 'prelu', 'relu', 'relu6', 'rrelu', 'selu', 'silu', 'mish', 'smooth_l1_loss', 'huber_loss',
#     'soft_margin_loss', 'softmax', 'softmin', 'softplus', 'softshrink', 'softsign', 'tanhshrink', '_threshold',
#     'triplet_margin_loss', 'triplet_margin_with_distance_loss', 'unfold', 'nonzero', 'norm', 'linalg_norm',
#     'linalg_vector_norm', 'linalg_matrix_norm', 'norm_except_dim', 'nuclear_norm', 'numel', 'orgqr', 'ormqr',
#     'pairwise_distance', 'permute', 'pca_lowrank', 'pdist', 'pinverse', 'linalg_pinv', 'pixel_shuffle',
#     'pixel_unshuffle', 'poisson', 'poisson_nll_loss', 'polygamma', 'positive', 'prelu', 'ones_like', 'pow', 'prod',
#     'put', 'q_per_channel_axis', 'q_per_channel_scales', 'q_per_channel_zero_points', 'q_scale', 'q_zero_point', 'qr',
#     'linalg_qr', 'quantile', 'nanquantile', 'quantize_per_channel', 'quantize_per_tensor', 'quantized_batch_norm',
#     'quantized_gru_cell', 'quantized_lstm_cell', 'quantized_max_pool1d', 'quantized_max_pool2d',
#     'quantized_rnn_relu_cell', 'quantized_rnn_tanh_cell', 'rad2deg', 'rand_like', 'randint_like', 'randn_like',
#     'ravel', 'real', 'vdot', 'view_as_real', 'view_as_complex', 'reciprocal', 'relu', 'remainder', 'renorm',
#     'repeat_interleave', 'reshape', 'rnn_relu', 'rnn_relu_cell', 'rnn_tanh', 'rnn_tanh_cell', 'roll', 'rot90',
#     'round', 'row_stack', '_rowwise_prune', 'rrelu', 'rsqrt', 'rsub', 'saddmm', 'scatter', 'scatter_add',
#     'searchsorted', 'segment_reduce', 'select', 'selu', 'sigmoid', 'sign', 'signbit', 'sgn', 'sin', 'sinc', 'sinh',
#     'slogdet', 'linalg_slogdet', 'smm', 'softmax', 'solve', 'linalg_solve', 'sort', 'split', 'split_with_sizes',
#     'sqrt', 'square', 'squeeze', 'stack', 'std', 'std_mean', 'stft', 'sub', 'subtract', 'sum', 'nansum', 'svd',
#     'svd_lowrank', 'linalg_svd', 'linalg_svdvals', 'symeig', 'swapaxes', 'swapdims', 'special_entr', 'special_erf',
#     'special_erfc', 'special_erfinv', 'special_exp2', 'special_expm1', 'special_expit', 'special_gammaln',
#     'special_i0e', 'special_logit', 'special_xlog1py', 't', 'take', 'take_along_dim', 'tan', 'tanh',
#     'linalg_tensorinv', 'linalg_tensorsolve', 'tensordot', 'tensor_split', 'threshold', 'tile', 'topk', 'trace',
#     'transpose', 'trapz', 'triangular_solve', 'tril', 'triplet_margin_loss', 'triu', 'true_divide', 'trunc', 'unbind',
#     'unique', 'unique_consecutive', 'unsafe_chunk', 'unsafe_split', 'unsafe_split_with_sizes', 'unsqueeze', 'var',
#     'var_mean', 'vsplit', 'vstack', 'where', 'zeros_like', '__floordiv__', '__rfloordiv__', '__ifloordiv__',
#     '__truediv__', '__rdiv__', '__idiv__', '__lshift__', '__ilshift__', '__rshift__', '__irshift__', '__float__',
#     '__complex__', '__array__', '__bool__', '__contains__', 'neg', '__invert__', '__mod__', '__imod__',
#     '__array_wrap__', '__getitem__', '__deepcopy__', '__int__', '__long__', '__hash__', '__index__', '__len__',
#     '__format__', '__reduce_ex__', '__reversed__', '__repr__', '__setitem__', '__setstate__', '__get__', 'type',
#     '_coalesced_', '_dimI', '_dimV', '_indices', '_is_view', '_nnz', 'crow_indices', 'col_indices',
#     '_update_names', '_values', 'align_as', 'align_to', 'apply_', 'as_strided', 'as_strided_', 'backward', 'bfloat16',
#     'bool', 'byte', 'char', 'cauchy_', 'coalesce', 'contiguous', 'copy_', 'cpu', 'cuda', 'xpu', 'data_ptr',
#     'dense_dim', 'dim', 'double', 'cdouble', 'element_size', 'expand', 'expand_as', 'exponential_', 'fill_',
#     'fill_diagonal_', 'float', 'cfloat', 'geometric_', 'get_device', 'half', 'has_names', 'indices', 'int',
#     'is_coalesced', 'is_contiguous', 'is_pinned', 'is_set_to', 'is_shared', 'item', 'log_normal_', 'log_softmax',
#     'long', 'map_', 'map2_', 'mm', 'narrow_copy', 'ndimension', 'nelement', 'normal_', 'numpy', 'permute',
#     'pin_memory', 'put_', 'qscheme', 'random_', 'record_stream', 'refine_names', 'register_hook', 'rename', 'repeat',
#     'requires_grad_', 'reshape_as', 'resize', 'resize_', 'resize_as', 'retain_grad', 'set_', 'share_memory_',
#     'short', 'size', 'sparse_dim', 'sparse_mask', 'sparse_resize_', 'sparse_resize_and_clear_', 'sspaddmm', 'storage',
#     'storage_offset', 'storage_type', 'sum_to_size', 'tile', 'to', 'to_dense', 'to_sparse', 'tolist', 'to_mkldnn',
#     'type_as', 'unfold', 'uniform_', 'values', 'view', 'view_as', 'zero_', 'linalg_lstsq', 'abs', 'abs_', 'absolute',
#     'absolute_', 'acos', 'acos_', 'arccos', 'arccos_', 'acosh', 'acosh_', 'arccosh', 'arccosh_', 'add', 'add_',
#     '__add__', '__iadd__', '__radd__', 'addbmm', 'addbmm_', 'addcdiv', 'addcdiv_', 'addcmul', 'addcmul_', 'addmm',
#     'addmm_', 'addmv', 'addmv_', 'addr', 'addr_', 'all', 'allclose', 'amax', 'amin', 'angle', 'any', 'argmax',
#     'argmin', 'argsort', 'asin', 'asin_', 'arcsin', 'arcsin_', 'asinh', 'asinh_', 'arcsinh', 'arcsinh_', 'atan',
#     'atan_', 'arctan', 'arctan_', 'atan2', 'atan2_', 'atanh', 'atanh_', 'arctanh', 'arctanh_', 'baddbmm', 'baddbmm_',
#     'bernoulli', 'bernoulli_', 'bincount', 'bitwise_and', 'bitwise_and_', '__and__', '__iand__', 'bitwise_not',
#     'bitwise_not_', 'bitwise_or', 'bitwise_or_', '__or__', '__ior__', 'bitwise_xor', 'bitwise_xor_', '__xor__',
#     '__ixor__', 'bmm', 'broadcast_to', 'ceil', 'ceil_', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 'chunk',
#     'clamp', 'clamp_', 'clip', 'clip_', 'clamp_min', 'clamp_min_', 'clamp_max', 'clamp_max_', 'clone', 'copysign',
#     'copysign_', 'conj', 'cos', 'cos_', 'cosh', 'cosh_', 'count_nonzero', 'cross', 'cummax', 'cummin', 'cumprod',
#     'cumprod_', 'cumsum', 'cumsum_', 'logcumsumexp', 'deg2rad', 'deg2rad_', 'dequantize', 'det', 'detach', 'detach_',
#     'diag', 'diag_embed', 'diagflat', 'diff', 'diagonal', 'digamma', 'digamma_', 'dist', 'div', 'div_', '__div__',
#     'divide', 'divide_', 'dot', 'dsplit', 'eig', 'eq', 'eq_', '__eq__', 'equal', 'erf', 'erf_', 'erfc', 'erfc_',
#     'erfinv', 'erfinv_', 'exp', 'exp_', 'exp2', 'exp2_', 'expm1', 'expm1_', 'fix', 'fix_', 'flatten', 'flip', 'fliplr',
#     'flipud', 'floor', 'floor_', 'floor_divide', 'floor_divide_', 'float_power', 'float_power_', 'fmod', 'fmod_',
#     'frac', 'frac_', 'frexp', 'gather', 'gcd', 'gcd_', 'ge', 'ge_', '__ge__', 'greater_equal', 'greater_equal_',
#     'geqrf', 'i0', 'i0_', 'inner', 'outer', 'ger', 'gt', 'gt_', '__gt__', 'greater', 'greater_', 'hardshrink',
#     'heaviside', 'heaviside_', 'histc', 'hsplit', 'hypot', 'hypot_', 'igamma', 'igamma_', 'igammac', 'igammac_',
#     'index_add', 'index_add_', 'index_copy', 'index_copy_', 'index_put', 'index_put_', 'index_select', 'index_fill',
#     'index_fill_', 'isfinite', 'isinf', 'isreal', 'isposinf', 'isneginf', 'int_repr', 'inverse', 'is_complex',
#     'is_distributed', 'is_floating_point', 'is_nonzero', 'is_same_size', 'is_signed', 'isclose', 'isnan', 'istft',
#     'kron', 'kthvalue', 'lcm', 'lcm_', 'ldexp', 'ldexp_', 'le', 'le_', '__le__', 'less_equal', 'less_equal_',
#     'lerp', 'lerp_', 'lgamma', 'lgamma_', 'log', 'log_', 'log10', 'log10_', 'log1p', 'log1p_', 'log2', 'log2_',
#     'logaddexp', 'logaddexp2', 'logdet', 'xlogy', 'xlogy_', 'logical_and', 'logical_and_', 'logical_not',
#     'logical_not_', 'logical_or', 'logical_or_', 'logical_xor', 'logical_xor_', 'logsumexp', 'logit', 'logit_',
#     'lstsq', 'lt', 'lt_', '__lt__', 'less', 'less_', 'lu', 'lu_solve', 'masked_fill', 'masked_fill_',
#     'masked_scatter', 'masked_scatter_', 'masked_select', 'matmul', '__matmul__', 'matrix_power', 'matrix_exp',
#     'max', 'maximum', 'fmax', 'mean', 'median', 'nanmedian', 'min', 'minimum', 'fmin', 'mode', 'movedim', 'moveaxis',
#     'msort', 'mul', 'mul_', '__mul__', '__imul__', '__rmul__', 'multiply', 'multiply_', 'multinomial', 'mv',
#     'mvlgamma', 'mvlgamma_', 'narrow', 'nan_to_num', 'nan_to_num_', 'ne', 'ne_', '__ne__', 'not_equal', 'not_equal_',
#     'neg_', 'negative', 'negative_', 'nextafter', 'nextafter_', 'prelu', 'relu', 'relu_', 'softmax', 'nonzero',
#     '__nonzero__', 'norm', 'numel', 'orgqr', 'ormqr', 'pinverse', 'polygamma', 'polygamma_', 'positive', 'pow',
#     'pow_', '__ipow__', '__rpow__', 'prod', 'put', 'q_per_channel_axis', 'q_per_channel_scales',
#     'q_per_channel_zero_points', 'q_scale', 'q_zero_point', 'qr', 'quantile', 'nanquantile', 'rad2deg', 'rad2deg_',
#     'ravel', 'vdot', 'reciprocal', 'reciprocal_', 'remainder', 'remainder_', 'renorm', 'renorm_',
#     'repeat_interleave', 'reshape', 'roll', 'rot90', 'round', 'round_', 'rsqrt', 'rsqrt_', '__rsub__', 'scatter',
#     'scatter_', 'scatter_add', 'scatter_add_', 'select', 'sigmoid', 'sigmoid_', 'sign', 'sign_', 'signbit', 'sgn',
#     'sgn_', 'sin', 'sin_', 'sinc', 'sinc_', 'sinh', 'sinh_', 'slogdet', 'smm', 'solve', 'sort', 'split',
#     'split_with_sizes', 'sqrt', 'sqrt_', 'square', 'square_', 'squeeze', 'squeeze_', 'std', 'stft', 'sub', 'sub_',
#     '__sub__', '__isub__', 'subtract', 'subtract_', 'sum', 'nansum', 'svd', 'symeig', 'swapaxes', 'swapaxes_',
#     'swapdims', 'swapdims_', 't', 't_', 'take', 'take_along_dim', 'tan', 'tan_', 'tanh', 'tanh_', 'tensor_split',
#     'topk', 'trace', 'transpose', 'transpose_', 'triangular_solve', 'tril', 'tril_', 'triu', 'triu_', 'true_divide',
#     'true_divide_', 'trunc', 'trunc_', 'unbind', 'unique', 'unique_consecutive', 'unsafe_chunk', 'unsafe_split',
#     'unsafe_split_with_sizes', 'unsqueeze', 'unsqueeze_', 'var', 'vsplit', 'where', 'rename_', 'resize_as_'
# ]

# Torch ops which can be directly converted to timetensors
TORCH_OPS_DIRECT = [
    # Indexing, etc
    'cat', 'chunk', 'dsplit', 'column_stack', 'gather', 'hsplit', 'hstack', 'index_select', 'narrow', 'scatter',
    'scatter_add', "split", 'tensor_split', 'tile', 'vsplit', 'where',
    # Pointwise operations,
    'abs', 'absolute', 'acos', 'arccos', 'acosh', 'arccosh', 'add', 'addcdiv', 'addcmul', 'angle', 'asin', 'arcsin',
    'asinh', 'arcsinh', 'atan', 'arctan', 'atanh', 'arctanh', 'atan2', 'bitwose_not', 'bitwise_and', 'bitwise_or',
    'bitwise_xor', 'ceil', 'clamp', 'clip', 'conj', 'copysign', 'cos', 'cosh', 'deg2rad', 'div', 'divide', 'digamma',
    'erf', 'erfc', 'erfinv', 'exp', 'exp2', 'expm1', # fake_quantize_per_channel_affine, fake_quantize_per_tensor_affine,
    'fix', 'float_power', 'floor', 'floor_divide', 'fmod', 'frac', 'frexp', 'gradient', 'imag', 'ldexp', 'lerp',
    'lgamma', 'log', 'log10', 'log1p', 'log2', 'logit', # hypot
    'i0', 'igamma', 'igammac', 'mul', 'multiply', 'mvlgamma', 'nan_to_num', 'neg', 'negative', # nextafter
    # logaddexp, logaddexp2
    'logical_and', 'logical_not', 'logical_or', 'logical_xor',
    # Other operation
    'atleast_1d', 'block_diag', 'broadcast_to', 'bucketize', 'clone',
    # Reduction ops
]

# Torch reduction ops which can be converted to timetensor depending on the value of the parameter 'dim'
TORCH_OPS_REDUCTION = [
    'argmax', 'argmin', 'amax', 'amin', 'max', 'min', 'logsumexp', 'mean', 'median', 'nanmedian', 'mode', 'nansum',
    'prod', 'quantile', 'nanquantile', 'std', 'std_mean', 'sum', 'unique', 'unique_consecutive', 'var', 'var_mean',
    'count_nonzero', 'argsort',
]

# List of torch ops implemented, if not in this list, we print a warning
TORCH_OPS_IMPLEMENTED = [
    # Indexing, etc
    'cat', 'chunk', 'dsplit', 'column_stack', 'dstack', 'gather', 'hsplit', 'hstack', 'index_select', 'masked_select',
    'movedim', 'moveaxis', 'narrow', 'nonzero', 'reshape', 'row_stack', 'scatter', 'scatter_add', "split", 'squeeze',
    'stack', 'swapaxes', 'swapdims', 't', 'atleast_3d', 'take', 'take_along_dim', 'tensor_split', 'tile', 'transpose',
    'unbind', 'unsqueeze', 'vsplit', 'vstack', 'where',
    # Pointwise operations,
    'abs', 'absolute', 'acos', 'arccos', 'acosh', 'arccosh', 'add', 'addcdiv', 'addcmul', 'angle', 'asin', 'arcsin',
    'asinh', 'arcsinh', 'atan', 'arctan', 'atanh', 'arctanh', 'atan2', 'bitwose_not', 'bitwise_and', 'bitwise_or',
    'bitwise_xor', 'ceil', 'clamp', 'clip', 'conj', 'copysign', 'cos', 'cosh', 'deg2rad', 'div', 'divide', 'digamma',
    'erf', 'erfc', 'erfinv', 'exp', 'exp2', 'expm1', # fake_quantize_per_channel_affine, fake_quantize_per_tensor_affine,
    'fix', 'float_power', 'floor', 'floor_divide', 'fmod', 'frac', 'frexp', 'gradient', 'imag', 'ldexp', 'lerp',
    'lgamma', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_or', 'logical_not',
    'logical_xor', 'logit', 'hypot', 'i0', 'igamma', 'igammac', 'mul', 'multiply', 'mvlgamma', 'nan_to_num', 'neg',
    'negative', 'nextafter', 'polygamma', 'positive', 'pow', 'rad2deg', 'real', 'reciprocal', 'remainder', 'round',
    'rsqrt', 'sigmoid', 'sign', 'sgn', 'signbit',
    # Other operations,
    'atleast_1d', 'atleast_2d', 'atleast_3d', 'bincount', 'block_diag', 'broadcast_tensors', 'broadcast_to',
    'bucketize', 'cartesian_prod', 'cdist', 'clone', 'combinations', 'cross', 'cummax', 'cummin',
    # Reduction ops
    'argmax', 'argmin', 'amax', 'amin', 'all', 'any', 'max', 'min', 'logsumexp', 'mean', 'median', 'nanmedian',
    'mode', 'nansum', 'prod', 'quantile', 'nanquantile', 'std', 'std_mean', 'sum', 'unique', 'unique_consecutive',
    'var', 'var_mean', 'count_nonzero',
    # Comparison
    'allclose', 'argsort', 'eq', 'equal', 'ge', 'greater_equal', 'gt', 'greated', 'isclose', 'isfinite', 'isinf',
    'kthvalue', 'le', 'less_equal', 'lt', 'less', 'maximum', 'minimum', 'fmax', 'fmin', 'ne', 'not_equal', 'sort',
    'topk', 'msort',
    # Spectral
    'stft', 'istft',
    # BLAS and LAPACK
    'mm',
    # Convolution
    'conv1d', 'conv2d', 'conv3d', 'conv_tranpose1d', 'conv_tranpose2d', 'conv_tranpose3d', 'unfold', 'fold',
    # Pooling
    'avg_pool1d', 'avg_pool2d', 'avg_pool3d', 'max_pool1d', 'max_pool1d_indices', 'max_pool2d', 'max_pool2d_indices',
    'max_pool3d', 'max_pool3d_indices', 'max_unpool1d', 'max_unpool2d', 'max_unpool3d', 'lp_pool1d', 'lp_pool2d',
    'lp_pool3d', 'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d', 'adaptive_avg_pool1d',
    'adaptive_avg_pool2d', 'adaptive_avg_pool3d', 'fractional_max_pool2d', 'fractional_max_pool3d',
    # Linear
    'linear', 'bilinear',
    # Dropout
    'dropout', 'alpha_dropout', 'feature_alpha_dropout', 'dropout2d', 'dropout3d',
    # Sparse
    'embedding', 'embedding_bag', 'one_hot',
    # Distance
    'pairwise_distance', 'cosine_similarity', 'pdist',
    # Vision
    'pixel_shuffle', 'pixel_unshuffle', 'pad', 'interpolate', 'grid_sample',
]

# Rejected Torch operations
TORCH_OPS_UNSUPPORTED = [
    'affine_grid'
]


# region TIMETENSOR

# TimeTensor
def check_time_lengths(
        time_len: int,
        time_lengths: Optional[torch.LongTensor],
        batch_sizes: torch.Size
):
    r"""Check time lengths

    :param time_lengths:
    :param batch_sizes:
    :return:
    """
    # Check that the given lengths tensor has the right
    # dimensions
    if time_lengths.size() != batch_sizes:
        raise ValueError(ERROR_WRONG_TIME_LENGTHS_SIZES.format(batch_sizes))
    # end if

    # Check that all lengths are not bigger
    # than the actual time-tensor
    if torch.any(time_lengths > time_len):
        raise ValueError(ERROR_TIME_LENGTHS_TOO_BIG)
    # end if

    return True
# end check_time_lengths


# TimeTensor
class TimeTensor(BaseTensor):
    r"""A  special tensor with a time dimension.
    """

    # region CONSTRUCTORS

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Copy if already a timetensor
        # transform otherwise
        if type(data) is TimeTensor:
            tensor_data = data.tensor
        else:
            tensor_data = data
        # end if

        # The tensor must have enough dimension
        # for the time dimension
        if tensor_data.ndim < time_dim + 1:
            # Error
            raise ValueError(
                ERROR_TENSOR_TO_SMALL.format(time_dim, tensor_data.ndim, time_dim + 1)
            )
        # end if

        # Set tensor and time index
        self._tensor = tensor_data
        self._time_dim = time_dim
    # end __init__

    # endregion CONSTRUCTORS

    # region PROPERTIES

    # Time dimension (getter)
    @property
    def time_dim(self) -> int:
        r"""Get the index of the time dimension.

        :return: The index of the time dimension.
        :rtype: ``ìnt``
        """
        return self._time_dim
    # end time_dim

    # Time dimension (setter)
    @time_dim.setter
    def time_dim(
            self,
            value: int
    ) -> None:
        r"""Set the index of the time dimension if valid.

        :param value: New index of the time dimension.
        :type value: ``ìnt``
        """
        # Check time dim is valid
        if value >= self.tensor.ndim:
            # Error
            raise ValueError(ERROR_TENSOR_TO_SMALL.format(value, self._tensor.ndim))
        elif value < 0:
            raise ValueError(ERROR_TIME_DIM_NEGATIVE)
        # end if

        # Set new time dim
        self._time_dim = value
    # end time_dim

    # Time length
    @property
    def tlen(self) -> int:
        r"""Returns the length of the time dimension.

        :return: the length of the time dimension.
        :rtype: ``int``
        """
        return self._tensor.size()[self._time_dim]
    # end tlen

    # Number of channel dimensions
    @property
    def cdim(self) -> int:
        r"""Number of channel dimensions.

        :return: the number of channel dimensions.
        :rtype: ``ìnt``
        """
        return self._tensor.ndim - self._time_dim - 1
    # end cdim

    # Number of batch dimensions
    @property
    def bdim(self) -> int:
        r"""Number of batch dimensions.

        :return: the number of batch dimensions.
        :rtype: ``ìnt``
        """
        return self._tensor.ndim - self.cdim - 1
    # end bdim

    # endregion PROPERTIES

    # region PUBLIC

    # Size of channel dimensions
    def csize(self) -> torch.Size:
        r"""Size of channel dimensions.
        """
        if self._time_dim != self._tensor.ndim - 1:
            tensor_size = self._tensor.size()
            return tensor_size[self.time_dim+1:]
        else:
            return torch.Size([])
        # end if
    # end csize

    # Size of batch dimensions
    def bsize(self) -> torch.Size:
        r"""Size of batch dimensions.
        """
        if self._time_dim == 0:
            return torch.Size([])
        else:
            tensor_size = self._tensor.size()
            return tensor_size[:self._time_dim]
        # end if
    # end bsize

    # Number of channel elements
    def numelc(self):
        r"""Returns the number of elements in the channel dimensions.
        """
        # Multiply sizes
        num_el = 1
        for c_size in list(self.csize()):
            num_el *= c_size
        # end for
        return num_el
    # end numelc

    # Number of batch elements
    def numelb(self):
        r"""Returns the number of elements in the batch dimensions.
        """
        # Multiply sizes
        num_el = 1
        for b_size in list(self.bsize()):
            num_el *= b_size
        # end for
        return num_el
    # end numelb

    # region CAST

    # To
    def to(self, *args, **kwargs) -> 'TimeTensor':
        r"""Performs TimeTensor dtype and/or device concersion. A ``torch.dtype`` and ``torch.device`` are inferred
        from the arguments of ``self.to(*args, **kwargs)

        .. note::
            From PyTorch documentation: if the ``self`` TimeTensor already has the correct ``torch.dtype`` and
            ``torch.device``, then ``self`` is returned. Otherwise, the returned timetensor is a copy of ``self``
            with the desired ``torch.dtype`` and ``torch.device``.

        Example::
            >>> ttensor = echotorch.randn(2, length=20)
            >>> ttensor.to(torch.float64)

        """
        # New tensor
        ntensor = self._tensor.to(*args, **kwargs)

        # Same tensor?
        if self._tensor == ntensor:
            return self
        else:
            return TimeTensor(
                ntensor,
                time_dim=self._time_dim
            )
        # end if
    # end to

    # endregion CAST

    # Indexing time tensor
    def indexing_timetensor(
            self,
            item
    ) -> 'TimeTensor':
        r"""Return a view of a :class:`TimeTensor` according to an indexing item.

        :param item: Data item to recover.
        :rtype: :class:`TimeTensor`
        """
        return TimeTensor(
            self._tensor[item],
            time_dim=self._time_dim
        )
    # end indexing_timetensor

    # endregion PUBLIC

    # region TORCH_FUNCTION

    # region TORCH_INDEXING

    # After dstack
    def after_dstack(
            self,
            func_output: Any,
            *ops_inputs,
            dim: int = 0
    ) -> 'TimeTensor':
        r"""After :func:`torch.dstack`.
        """
        # 0-D timeseries
        if self.ndim == 1:
            return TimeTensor(
                data=func_output,
                time_dim=1
            )
        else:
            return TimeTensor(
                data=func_output,
                time_dim=self._time_dim
            )
        # end if
    # end after_dstack

    # After movedim
    def after_movedim(
            self,
            func_output: Any,
            ops_input,
            source,
            destination
    ) -> 'TimeTensor':
        r"""After :func:`torch.movedim` we change index of time dimension of concerned.
        """
        # Keep time dim
        new_time_dim = self.time_dim

        # New time dim if in dest or source
        if source == self.time_dim:
            new_time_dim = destination
        elif destination == self.time_dim:
            new_time_dim = source
        # end if

        # New timetensor
        return TimeTensor(
            data=func_output,
            time_dim=new_time_dim
        )
    # end after_movedim

    # After squeeze
    def after_squeeze(
            self,
            func_output,
            input,
            dim=None
    ) -> Union['TimeTensor', torch.Tensor]:
        r"""
        """
        if dim is None:
            if self.tlen == 1:
                return func_output
            else:
                # How many dim at one before time dim?
                removed_dim = torch.sum(torch.tensor(self.size()[:self.time_dim]) == 1)

                # Return with modified time dim
                return TimeTensor(
                    data=func_output,
                    time_dim=self.time_dim - removed_dim
                )
            # end if
        else:
            # Time dim targeted
            if dim == self.time_dim and self.tlen == 1:
                return func_output
            # end if

            # If dim removed and before time dim
            if self.size()[dim] == 1 and dim < self.time_dim:
                return TimeTensor(
                    data=func_output,
                    time_dim=self.time_dim - 1
                )
            else:
                return TimeTensor(
                    data=func_output,
                    time_dim=self.time_dim
                )
            # end if
        # end if
    # end after_squeeze

    # After stack
    def after_stack(
            self,
            func_output,
            tensors,
            dim=0
    ) -> 'TimeTensor':
        r"""After :func:`torch.stack`, we increment time dim if needed.
        """
        if dim <= self.time_dim:
            return TimeTensor(
                data=func_output,
                time_dim=self.time_dim+1
            )
        else:
            return TimeTensor(
                data=func_output,
                time_dim=self.time_dim
            )
        # end if
    # end after_stack

    # after t
    def after_t(
            self,
            func_output,
            input
    ) -> 'TimeTensor':
        r"""After :func:`torch.t`, swap time dim.
        """
        if self.ndim == 1:
            return TimeTensor(
                data=func_output,
                time_dim=self.time_dim
            )
        else:
            return TimeTensor(
                data=func_output,
                time_dim=1 - self.time_dim
            )
        # end if
    # end after_t

    # After tranpose
    def after_transpose(
            self,
            func_output,
            input,
            dim0,
            dim1
    ) -> 'TimeTensor':
        r"""After :func:`torch.t`, swap time dim.
        """
        if self.time_dim in [dim0, dim1]:
            return TimeTensor(
                data=func_output,
                time_dim=dim0 if self.time_dim == dim1 else dim1
            )
        else:
            return TimeTensor(
                data=func_output,
                time_dim=self.time_dim
            )
        # end if
    # end after_transpose

    # After unbind
    def after_unbind(
            self,
            func_output,
            input,
            dim=0
    ) -> Tuple['TimeTensor', torch.Tensor]:
        r"""After :func:`torch.unbind`, remove time dim if needed.
        """
        if dim == self.time_dim:
            return func_output
        else:
            return tuple(self.transform_to_timetensors(func_output))
        # end if
    # end after_unbind

    # After unsqueeze
    def after_unsqueeze(
            self,
            func_output: Any,
            input: Any,
            dim: int
    ) -> 'TimeTensor':
        r"""After :func:`torch.unsqueeze`, remove time dim if needed.

        :param func_output: The output of the torch.unsqueeze function.
        :type func_output:
        :param dim: The request dimension from unsqueeze.
        :type dim:
        :return: The computed output.
        :rtype:
        """
        if dim <= self.time_dim:
            return TimeTensor(
                func_output,
                time_dim=self._time_dim + 1
            )
        else:
            return TimeTensor(
                func_output,
                time_dim=self._time_dim
            )
        # end if
    # end after_unsqueeze

    # After vstack
    def after_vstack(
            self,
            func_output,
            tensors
    ) -> 'TimeTensor':
        r"""After :func:`torch.vstack, we add 1 to the index of time
        dim if it is a 0-D timeseries, other keep the same
        """
        if self.ndim == 1:
            return TimeTensor(
                data=func_output,
                time_dim=1
            )
        else:
            return TimeTensor(
                data=func_output,
                time_dim=self.time_dim
            )
        # end if
    # end after_vstack

    # endregion TORCH_INDEXING

    # region TORCH_COMPARISON

    # After kthvalue
    def after_kthvalue(
            self,
            func_ret,
            input,
            k,
            dim=None,
            keepdim=False
    ):
        r"""After :func:`torch.kthvalue`.
        """
        return (
            self.convert_after_reduction(func_ret.values, input, dim, keepdim),
            self.convert_after_reduction(func_ret.indices, input, dim, keepdim)
        )
    # end after_kthvalue

    # After topk
    def after_topk(
            self,
            func_ret,
            input,
            k,
            dim=None,
            largest=True,
            sorted=True
    ):
        r"""After :func:`torch.kthvalue`.
        """
        return (
            self.convert_after_reduction(func_ret.values, input, dim, True),
            self.convert_after_reduction(func_ret.indices, input, dim, True)
        )
    # end after_kthvalue

    # endregion TORCH_COMPARISON

    # region TORCH_SPECTRAL

    # Short-time Fourier transform (STFT)
    def after_stft(
            self,
            func_ret,
            input,
            n_fft,
            hop_length=None,
            win_length=None,
            window=None,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=None,
            return_complex=None
    ) -> 'TimeTensor':
        r"""After :func:`torch.stft`.
        """
        if input.ndim == 1:
            return TimeTensor(
                data=func_ret,
                time_dim=1
            )
        else:
            return TimeTensor(
                data=func_ret,
                time_dim=2
            )
        # end if
    # end after_stft

    # Inverse Short-time Fourier transform (ISTFT)
    def after_istft(
            self,
            func_ret,
            input,
            n_fft,
            hop_length=None,
            win_length=None,
            window=None,
            center=True,
            normalized=False,
            onesided=None,
            length=None,
            return_complex=False
    ) -> 'TimeTensor':
        r"""After :func:`torch.istft`
        """
        if input.ndim == 3:
            return TimeTensor(
                data=func_ret,
                time_dim=0
            )
        else:
            return TimeTensor(
                data=func_ret,
                time_dim=1
            )
        # end if
    # end after_istft

    # endregion TORCH_SPECTRAL

    # region TORCH_OTHER

    # After atleast_2d
    def after_atleast_2d(
            self,
            func_output: Any,
            *ops_inputs,
            dim: int = 0
    ) -> 'TimeTensor':
        r"""After :func:`torch.atleast_2d`.
        """
        # 0-D timeseries
        if self.ndim == 1:
            return TimeTensor(
                data=func_output,
                time_dim=1
            )
        else:
            return TimeTensor(
                data=func_output,
                time_dim=self._time_dim
            )
        # end if
    # end after_atleast_2d

    # After atleast_3d
    def after_atleast_3d(
            self,
            func_output: Any,
            *ops_inputs,
            dim: int = 0
    ) -> 'TimeTensor':
        r"""After :func:`torch.atleast_3d`.
        """
        # 0-D timeseries
        if self.ndim == 1:
            return TimeTensor(
                data=func_output,
                time_dim=1
            )
        else:
            return TimeTensor(
                data=func_output,
                time_dim=self._time_dim
            )
        # end if
    # end after_atleast_3d

    # After broadcast_tensors
    def after_broadcast_tensors(
            self,
            func_output,
            *tensors
    ) -> Tuple['TimeTensor']:
        r"""After :func:`torch.broadcast_tensors`.
        """
        output_list = list()
        for t_i, tensor in enumerate(func_output):
            if isinstance(tensors[t_i], TimeTensor):
                output_list.append(
                    TimeTensor(
                        data=tensor,
                        time_dim=tensors[t_i].time_dim
                    )
                )
            else:
                output_list.append(tensor)
            # end if
        # end for
        return tuple(output_list)
    # end after_broadcast_tensors

    # After cartesian_prod
    def after_cartesian_prod(
            self,
            func_output,
            *tensors
    ) -> 'TimeTensor':
        r"""After :func:`torch.cartesian_prod`.
        """
        return TimeTensor(
            data=func_output,
            time_dim=0
        )
    # end after_cartesian_prod

    # After cdist
    def after_cdist(
            self,
            func_output,
            *tensors,
            p=2.0,
            compute_mode='use_mm_for_euclid_dist_if_necessary'
    ) -> Union[torch.Tensor, 'TimeTensor']:
        r"""After :func:`torch.cdist`.
        """
        if self is tensors[0]:
            if self.time_dim in [0, 1]:
                return TimeTensor(
                    data=func_output,
                    time_dim=self.time_dim
                )
            else:
                return func_output
            # end if
        else:
            if self.time_dim == 0:
                return TimeTensor(
                    data=func_output,
                    time_dim=0
                )
            elif self.time_dim == 1:
                return TimeTensor(
                    data=func_output,
                    time_dim=2
                )
            else:
                return func_output
            # end if
        # end if
    # end after_cdist

    # After combinations
    def after_combinations(
            self,
            func_ret,
            input,
            r=2,
            with_replacement=False
    ) -> 'TimeTensor':
        r"""After :func:`torch.combinations`.
        """
        return TimeTensor(
            data=func_ret,
            time_dim=0
        )
    # end after_combinations

    # After cummax
    def after_cummax(
            self,
            func_ret,
            input,
            dim=None
    ):
        r"""After :func:`torch.cummax`.
        """
        return (
            self.convert_after_reduction(func_ret.values, input, dim, True),
            self.convert_after_reduction(func_ret.indices, input, dim, True)
        )
    # end after_cummax

    # After cummin
    def after_cummin(
            self,
            func_ret,
            input,
            dim=None
    ):
        r"""After :func:`torch.cummax`.
        """
        return (
            self.convert_after_reduction(func_ret.values, input, dim, True),
            self.convert_after_reduction(func_ret.indices, input, dim, True)
        )
    # end after_cummin

    # endregion TORCH_OTHER

    # region TORCH_BLAS_LAPACK

    # After mm (matrix multiplication)
    def mm(
            self,
            func_output: Any,
            m1,
            m2
    ) -> Union['TimeTensor', torch.Tensor]:
        r"""After mm (matrix multiplication)

        :param m1: first tensor.
        :type m1: :class:`TimeTensor` or ``torch.Tensor``
        :param m2: second tensor.
        :type m2: :class:`TimeTensor` or ``torch.Tensor``
        """
        return func_output
    # end mm

    # endregion TORCH_BLAS_LAPACK

    # region TORCH_CONVOLUTION

    # After fold
    def after_fold(
            self,
            func_output: Any,
            output_size,
            kernel_size,
            dilation=1,
            padding=0,
            stride=1
    ) -> 'TimeTensor':
        r"""After :func:`torch.fold`.
        """
        return echotorch.as_timetensor(
            data=func_output,
            time_dim=2
        )
    # end after_fold

    # endregion TORCH_CONVOLUTION

    # region TORCH_SPARSE

    # After embedding_bag
    def after_embedding_bag(
            self,
            func_output,
            input,
            weight,
            offsets=None,
            max_norm=None,
            norm_type=2,
            scale_grad_by_freq=False,
            mode='mean',
            sparse=False,
            per_sample_weights=None,
            include_last_offset=False,
            padding_idx=None
    ) -> Union['TimeTensor', torch.Tensor]:
        r"""After :func:`torch.embedding_bag`.
        """
        if input.ndim == 1:
            return echotorch.as_timetensor(
                data=func_output,
                time_dim=0
            )
        else:
            return func_output
        # end if
    # end after_embedding_bag

    # endregion TORCH_SPARSE

    # region TORCH_DISTANCE

    # After pairwise_distance
    def pairwise_distance(
            self,
            func_output,
            x1,
            x2,
            p=2.0,
            eps=1e-06,
            keepdim=False
    ) -> Union['TimeTensor', torch.Tensor]:
        r"""After :func:`torch.pairwise_distance`.
        """
        if self.time_dim == 0:
            return echotorch.as_timetensor(
                data=func_output,
                time_dim=0
            )
        else:
            return func_output
        # end if
    # end pairwise_distance

    # endregion TORCH_DISTANCE

    # Transform to timetensor
    def transform_to_timetensors(
            self,
            tensors: Any
    ):
        r"""Transform :class:`torch.Tensor` to :class:`TimeTensor`.
        """
        return [echotorch.as_timetensor(o, time_dim=self.time_dim) for o in tensors]
    # end transform_to_timetensors

    # Transform object to timetensor if possible
    def transform_object_to_timetensor(
            self,
            obj
    ):
        r"""Transform object to timetensor if possible
        """
        if isinstance(obj, torch.Tensor):
            return echotorch.as_timetensor(
                data=obj,
                time_dim=self.time_dim
            )
        else:
            return obj
        # end if
    # end transform_object_to_timetensor

    # Convert to timetensor
    def convert_to_timetensor(self, func_ret):
        r"""Convert to timetensor.
        """
        if isinstance(func_ret, torch.Tensor):
            return TimeTensor(
                data=func_ret,
                time_dim=self.time_dim
            )
        elif isinstance(func_ret, list):
            return [self.transform_object_to_timetensor(x) for x in func_ret]
        elif isinstance(func_ret, tuple):
            return tuple([self.transform_object_to_timetensor(x) for x in func_ret])
        else:
            return func_ret
        # end if
    # end convert_to_timetensor

    # Transform to timetensor if coherent
    def transform_similar_tensors(self, inpt):
        r"""Convert input to timetensors if elements are tensor with the same number of dimension.
        """
        if isinstance(inpt, torch.Tensor) and inpt.ndim == self.ndim:
            return echotorch.as_timetensor(
                data=inpt,
                time_dim=self.time_dim
            )
        else:
            return inpt
        # end if
    # end transform_similar_tensors

    # Convert to timetensor if coherent
    def convert_similar_tensors(self, inpt):
        r"""Convert input to timetensors if elements are tensor with the same number of dimension.
        """
        if isinstance(inpt, torch.Tensor) and inpt.ndim == self.ndim:
            return echotorch.as_timetensor(
                data=inpt,
                time_dim=self.time_dim
            )
        elif isinstance(inpt, list):
            return [self.transform_similar_tensors(x) for x in inpt]
        elif isinstance(inpt, tuple):
            return tuple([self.transform_similar_tensors(x) for x in inpt])
        else:
            return inpt
        # end if
    # end convert_similar_tensors

    # Check that all timetensors have the right time dimension index
    def check_time_dim(
            self,
            tensors: Any
    ) -> None:
        r"""Check that all timetensors have the right time dimension index.
        """
        for tensor in tensors:
            if isinstance(tensor, TimeTensor) and self.time_dim != tensor.time_dim:
                raise RuntimeError(
                    "Expected timetensors with the same time dimension index, got {} and {}".format(
                        self.time_dim,
                        tensor.time_dim
                    )
                )
            # end if
        # end for
    # end check_time_dim

    # Convert the output of a reduction operation
    def convert_after_reduction(
            self,
            func_ret,
            input,
            dim: int = None,
            keepdim: bool = False,
            **kwargs
    ) -> Union['TimeTensor', torch.Tensor]:
        r"""Convert the output of a reduction operation.
        """
        if (dim is None or dim == self.time_dim) and not keepdim:
            return func_ret
        else:
            return self.convert_to_timetensor(func_ret)
        # end if
    # end convert_after_reduction

    # Transpose
    # def t(self) -> 'TimeTensor':
    #     r"""Expects the timetensor to be <= 1-D timetensor and transposes dimensions 0 and 1.
    #
    #     If time dimension is in position 0, then it is switched to 1, and vice versa.
    #
    #     TODO: complet doc
    #     """
    #     if self.ndim == 2:
    #         return TimeTensor(
    #             data=self._tensor.t(),
    #             time_dim=1-self._time_dim
    #         )
    #     elif self.ndim < 2:
    #         return self
    #     else:
    #         # Inverse time dim if targeted
    #         if self._time_dim in [0, 1]:
    #             return TimeTensor(
    #                 data=self._tensor.t(),
    #                 time_dim=1 - self._time_dim
    #             )
    #         else:
    #             return TimeTensor(
    #                 data=self._tensor.t(),
    #                 time_dim=self._time_dim
    #             )
    #         # end if
    #     # end if
    # # end t

    # As strided
    def as_strided(
            self,
            size,
            stride,
            storage_offset=0,
            time_dim=None
    ) -> 'TimeTensor':
        r"""TODO: document

        :param size:
        :param stride:
        :param storage_offset:
        :return:
        :rtype:
        """
        # Strided tensor
        data_tensor = self._tensor.as_strided(size, stride, storage_offset)

        # Time dim still present
        if len(size) >= self._time_dim + 1:
            # Return timetensor
            return TimeTensor.new_timetensor(
                data=data_tensor,
                time_dim=self._time_dim if time_dim is None else time_dim
            )
        elif time_dim is not None:
            pass
        else:
            return data_tensor
        # end if
    # end as_strided

    # Torch functions
    def __torch_function__(
            self,
            func,
            types,
            args=(),
            kwargs=None
    ):
        """
        Torch functions
        """
        # Dict if None
        if kwargs is None:
            kwargs = {}

        # end if

        # Convert timetensor to tensors
        def convert(args):
            if type(args) is TimeTensor:
                return args.tensor
            elif type(args) is tuple:
                return tuple([convert(a) for a in args])
            elif type(args) is list:
                return [convert(a) for a in args]
            else:
                return args
            # end if

        # end convert

        # Raise error if unsupported operation
        if func.__name__ in TORCH_OPS_UNSUPPORTED:
            raise RuntimeError(
                "Operation {} is not supported for timetensors".format(func.__name__)
            )
        # end if

        # Print warning if not implemented
        if func.__name__ not in TORCH_OPS_IMPLEMENTED:
            warnings.warn(
                "Operation {} not implemented for timetensors, unpredictable behaviors here!".format(func.__name__)
            )
        # end if

        # Validate ops inputs
        if hasattr(self, 'validate_' + func.__name__): getattr(self, 'validate_' + func.__name__)(*args, **kwargs)

        # Before callback
        if hasattr(self, 'before_' + func.__name__): args = getattr(self, 'before_' + func.__name__)(*args, **kwargs)

        # Get the tensor in the arguments
        conv_args = [convert(a) for a in args]

        # Middle callback
        if hasattr(self, 'middle_' + func.__name__): args = getattr(self, 'middle_' + func.__name__)(*args, **kwargs)

        # Execute function
        ret = func(*conv_args, **kwargs)

        # If output can be directly converted to timetensor
        if func.__name__ in TORCH_OPS_DIRECT:
            ret = self.convert_to_timetensor(ret)
        elif func.__name__ in TORCH_OPS_REDUCTION:
            ret = self.convert_after_reduction(ret, *args, **kwargs)
        # end if

        # Create TimeTensor and returns or returns directly
        if hasattr(self, 'after_' + func.__name__):
            return getattr(self, 'after_' + func.__name__)(ret, *args, **kwargs)
        else:
            return self.convert_similar_tensors(ret)
        # end if
    # end __torch_function__

    # endregion TORCH_FUNCTION

    # region OVERRIDE

    # Get item
    def __getitem__(self, item) -> Union['TimeTensor', torch.Tensor]:
        """
        Get data in the tensor
        """
        # Multiple indices
        if type(item) is tuple:
            # If time dim is in
            if len(item) > self._time_dim:
                # Selection or slice?
                if type(item[self._time_dim]) in [slice, list]:
                    return self.indexing_timetensor(item)
                else:
                    return self._tensor[item]
                # end if
            else:
                return self.indexing_timetensor(item)
            # end if
        elif type(item) in [slice, list]:
            return self.indexing_timetensor(item)
        else:
            # Time selection?
            if self._time_dim == 0:
                return self._tensor[item]
            else:
                return self.indexing_timetensor(item)
        # end if
    # end __getitem__

    # Set item
    def __setitem__(self, key, value) -> None:
        """
        Set data in the tensor
        """
        self._tensor[key] = value
    # end __setitem__

    # Length
    def __len__(self) -> int:
        """
        Time length of the time series
        """
        return self.tlen
    # end __len__

    # Get representation
    def __repr__(self) -> str:
        r"""Get a string representation

        :return: ``TimeTensor`` representation.
        :rtype: ``str``
        """
        tensor_desc = self._tensor.__repr__()
        tensor_desc = tensor_desc[7:-1]
        return "timetensor({}, time_dim: {})".format(tensor_desc, self._time_dim)
    # end __repr__

    # Less than operation with time tensors.
    def __lt__(self, other) -> 'TimeTensor':
        r"""Less than operation with time tensors.
        """
        return TimeTensor(
            data=self._tensor < other,
            time_dim=self.time_dim
        )
    # end __lt__

    # Less or equal than operation with time tensors.
    def __le__(self, other) -> 'TimeTensor':
        r"""Less than operation with time tensors.
        """
        return TimeTensor(
            data=self._tensor <= other,
            time_dim=self.time_dim
        )
    # end __le__

    # Greater than operation with time tensors.
    def __gt__(self, other) -> 'TimeTensor':
        r"""Greater than operation with time tensors.
        """
        return TimeTensor(
            data=self._tensor > other,
            time_dim=self.time_dim
        )
    # end __gt__

    # Greater or equal than operation with time tensors.
    def __ge__(self, other) -> 'TimeTensor':
        r"""Greater or equal than operation with time tensors.
        """
        return TimeTensor(
            data=self._tensor >= other,
            time_dim=self.time_dim
        )
    # end __ge__

    # Are two time-tensors equivalent
    def __eq__(
            self,
            other: 'TimeTensor'
    ) -> bool:
        r"""Are two time-tensors equivalent?

        :param other: The other time-tensor
        :type other: ``TimeTensor``

        """
        return super(TimeTensor, self).__eq__(other) and self.time_dim == other.time_dim
    # end __eq__

    # Are two time-tensors not equal
    def __ne__(
            self,
            other: 'TimeTensor'
    ) -> bool:
        r"""Are two time-tensors not equal
        """
        return not(self.__eq__(self, other))
    # end __ne__

    # endregion OVERRIDE

    # region STATIC

    # Returns a new TimeTensor with data as the tensor data.
    @classmethod
    def new_timetensor(
            cls,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> 'TimeTensor':
        """
        Returns a new TimeTensor with data as the tensor data.
        @param data:
        @param time_lengths:
        @param time_dim:
        @param copy_data:
        @return:
        """
        return TimeTensor(
            data,
            time_dim=time_dim
        )
    # end new_timetensor

    # Returns new time tensor with a specific function
    @classmethod
    def new_timetensor_with_func(
            cls,
            *size: Tuple[int],
            func: Callable,
            length: int,
            batch_size: Optional[Tuple[int]] = None,
            out: Optional['TimeTensor'] = None,
            **kwargs
    ) -> 'TimeTensor':
        r"""Returns a new :class:`TimeTensor` with a specific function to generate the data.

        :param func:
        :type func:
        :param size:
        :type size:
        :param length:
        :type length:
        :param batch_size:
        :type batch_size:
        :param out:
        :type out:
        """
        # Batch size
        batch_size = tuple() if batch_size is None else batch_size

        # Time dim
        time_dim = len(batch_size)

        # Total size
        tt_size = list(batch_size) + [length] + list(size)

        # Output object
        out = out.tensor if isinstance(out, TimeTensor) else out

        # Out mode
        if out is None:
            # Create TimeTensor
            return TimeTensor(
                data=func(tuple(tt_size), **kwargs),
                time_dim=time_dim
            )
        else:
            # Call function
            func(tuple(tt_size), out=out.tensor, **kwargs)
            out.time_dim = time_dim
            return out
        # end if
    # end new_timetensor_with_func

    # endregion STATIC

# end TimeTensor

# endregion TIMETENSOR


# region VARIANTS

# Float time tensor
class FloatTimeTensor(TimeTensor):
    r"""Float time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""Float TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(FloatTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Transform type
        self.float()
    # end __init__

# end FloatTimeTensor


# Double time tensor
class DoubleTimeTensor(TimeTensor):
    r"""Double time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""Double TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(DoubleTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast data
        self.double()
    # end __init__

# end DoubleTimeTensor


# Half time tensor
class HalfTimeTensor(TimeTensor):
    r"""Half time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""Half TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(HalfTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast data
        self.halt()
    # end __init__

# end HalfTimeTensor


# 16-bit floating point 2 time tensor
class BFloat16Tensor(TimeTensor):
    r"""16-bit floating point 2 time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""16-bit TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(BFloat16Tensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast
        self.bfloat16()
    # end __init__

# end BFloat16Tensor


# 8-bit integer (unsigned) time tensor
class ByteTimeTensor(TimeTensor):
    r"""8-bit integer (unsigned) time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""8-bit integer (unsigned) TimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(ByteTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast
        self.byte()
    # end __init__

# end ByteTimeTensor


# 8-bit integer (signed) time tensor
class CharTimeTensor(TimeTensor):
    r"""8-bit integer (unsigned) time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""CharTimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(CharTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast
        self.char()
    # end __init__

# end CharTimeTensor


# Boolean time tensor
class BooleanTimeTensor(TimeTensor):
    r"""Boolean time tensor.
    """

    # Constructor
    def __init__(
            self,
            data: Union[torch.Tensor, 'TimeTensor'],
            time_dim: Optional[int] = 0
    ) -> None:
        r"""BooleanTimeTensor constructor

        :param data: The data in a torch tensor to transform to timetensor.
        :param time_dim: The position of the time dimension.
        """
        # Super call
        super(BooleanTimeTensor, self).__init__(
            self,
            data,
            time_dim=time_dim
        )

        # Cast
        self.bool()
    # end __init__

# end CharTimeTensor


# endregion VARIANTS
