# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
import numpy as np
from collections import namedtuple

from nets.network import Network
from model.config import cfg

# https://www.cnblogs.com/hellcat/p/9726528.html
# 分离卷积模块
def separable_conv2d_same(inputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D separable convolution with 'SAME' padding.
  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.
  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
    inputs: 是size为[batch_size, height, width, channels]的tensor
    kernel_size: 是filter的size [kernel_height, kernel_width]，如果filter的长宽一样可以只填入一个int值。
    depth_multiplier：深度因子
  """

  # By passing filters=None
  # separable_conv2d produces only a depth-wise convolution layer
  if stride == 1:
    return slim.separable_conv2d(inputs, None, kernel_size, 
                                  depth_multiplier=1, stride=1, rate=rate,
                                  padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.separable_conv2d(inputs, None, kernel_size, 
                                  depth_multiplier=1, stride=stride, rate=rate, 
                                  padding='VALID', scope=scope)

# The following is adapted from:
# https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.py

# Conv and DepthSepConv named tuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
# 网络结构
_CONV_DEFS = [
    Conv(kernel=3, stride=2, depth=32),
    DepthSepConv(kernel=3, stride=1, depth=64),
    DepthSepConv(kernel=3, stride=2, depth=128),
    DepthSepConv(kernel=3, stride=1, depth=128),
    DepthSepConv(kernel=3, stride=2, depth=256),
    DepthSepConv(kernel=3, stride=1, depth=256),
    DepthSepConv(kernel=3, stride=2, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    # use stride 1 for the 13th layer
    DepthSepConv(kernel=3, stride=1, depth=1024),
    DepthSepConv(kernel=3, stride=1, depth=1024)
]

# Modified mobilenet_v1
def mobilenet_v1_base(inputs,
                      conv_defs,
                      starting_layer=0,
                      min_depth=8,          # min_depth:所有卷积操作的最小深度值(通道数)。
                      depth_multiplier=1.0,
                      output_stride=None,
                      reuse=None,
                      scope=None):
  """Mobilenet v1.
  Constructs a Mobilenet v1 network from inputs to the given final endpoint.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    starting_layer: specifies the current starting layer. For region proposal 
      network it is 0, for region classification it is 12 by default.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef named tuples specifying the net architecture.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. 
    scope: Optional variable_scope.
  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
  Raises:
    ValueError: if depth_multiplier <= 0, or convolution type is not defined.
  """
  # 通道数最小为8
  depth = lambda d: max(int(d * depth_multiplier), min_depth)
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse):
    # The current_stride variable keeps track of the output stride of the
    # activations, i.e., the running product of convolution strides up to the
    # current network layer. This allows us to invoke atrous convolution
    # whenever applying the next convolution would result in the activations
    # having output stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    net = inputs
    for i, conv_def in enumerate(conv_defs):
      end_point_base = 'Conv2d_%d' % (i + starting_layer)

      # 默认为空，未使用空洞卷积
      if output_stride is not None and current_stride == output_stride:
        # If we have reached the target output_stride, then we need to employ
        # atrous convolution with stride=1 and multiply the atrous rate by the
        # current unit's stride for use in subsequent layers.
        layer_stride = 1
        layer_rate = rate
        rate *= conv_def.stride
      else:
          # 未使用
        layer_stride = conv_def.stride
        layer_rate = 1
        current_stride *= conv_def.stride

          # 定义第一层conv层
      if isinstance(conv_def, Conv):
        end_point = end_point_base
        net = resnet_utils.conv2d_same(net, depth(conv_def.depth), conv_def.kernel,
                          stride=conv_def.stride,
                          scope=end_point)

        # 定义DW层
      elif isinstance(conv_def, DepthSepConv):
        end_point = end_point_base + '_depthwise'
        
        net = separable_conv2d_same(net, conv_def.kernel,           # depthwise
                                    stride=layer_stride,
                                    rate=layer_rate,
                                    scope=end_point)

        end_point = end_point_base + '_pointwise'

        net = slim.conv2d(net, depth(conv_def.depth), [1, 1],           # pointwise
                          stride=1,
                          scope=end_point)

      else:
        raise ValueError('Unknown convolution type %s for layer %d'
                         % (conv_def.ltype, i))

    return net

# Modified arg_scope to incorporate configs
def mobilenet_v1_arg_scope(is_training=True,
                           stddev=0.09):
  batch_norm_params = {
      'is_training': False,
      'center': True,
      'scale': True,
      'decay': 0.9997,
      'epsilon': 0.001,
      'trainable': False,
  }

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  # 指定权重初始化和正则化方式
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(cfg.MOBILENET.WEIGHT_DECAY)
  # 是否对DW进行正则化，默认为不进行，论文中也不推荐进行正则化
  if cfg.MOBILENET.REGU_DEPTH:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None

    # 设置默认参数
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      trainable=is_training,
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, 
                      normalizer_fn=slim.batch_norm,
                      padding='SAME'):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc

class mobilenetv1(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER
    self._scope = 'MobilenetV1'

  def _image_to_head(self, is_training, reuse=None):
      # Base bottleneck
      # FIXED_LAYERS默认为5 reuse默认为none
    assert (0 <= cfg.MOBILENET.FIXED_LAYERS <= 12)
    net_conv = self._image
    if cfg.MOBILENET.FIXED_LAYERS > 0:
      with slim.arg_scope(mobilenet_v1_arg_scope(is_training=False)):
        net_conv = mobilenet_v1_base(net_conv,
                                      _CONV_DEFS[:cfg.MOBILENET.FIXED_LAYERS],
                                      starting_layer=0,
                                      depth_multiplier=self._depth_multiplier,
                                      reuse=reuse,
                                      scope=self._scope)
    if cfg.MOBILENET.FIXED_LAYERS < 12:
      with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        net_conv = mobilenet_v1_base(net_conv,
                                      _CONV_DEFS[cfg.MOBILENET.FIXED_LAYERS:12],
                                      starting_layer=cfg.MOBILENET.FIXED_LAYERS,
                                      depth_multiplier=self._depth_multiplier,
                                      reuse=reuse,
                                      scope=self._scope)

    self._act_summaries.append(net_conv)
    self._layers['head'] = net_conv

    return net_conv

  def _head_to_tail(self, pool5, is_training, reuse=None):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
      fc7 = mobilenet_v1_base(pool5,
                              _CONV_DEFS[12:],
                              starting_layer=12,
                              depth_multiplier=self._depth_multiplier,
                              reuse=reuse,
                              scope=self._scope)
      # average pooling done by reduce_mean
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])
    return fc7

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      # 把第一层加入固定参数列表中，其余层加入重载参数列表中
      if v.name == (self._scope + '/Conv2d_0/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix MobileNet V1 layers..')
    with tf.variable_scope('Fix_MobileNet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR, and match the scale by (255.0 / 2.0)
        Conv2d_0_rgb = tf.get_variable("Conv2d_0_rgb", 
                                    [3, 3, 3, max(int(32 * self._depth_multiplier), 8)], 
                                    trainable=False)
        restorer_fc = tf.train.Saver({self._scope + "/Conv2d_0/weights": Conv2d_0_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._scope + "/Conv2d_0/weights:0"], 
                           tf.reverse(Conv2d_0_rgb / (255.0 / 2.0), [2])))
