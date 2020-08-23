# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from layer_utils.generate_anchors import generate_anchors

'''
实际上是对generate_anchors()的封装
generate_anchors就是生成anchor的
'''

# 功能是给定不同的scales和ratios生成各种尺度的anchors
def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]      # 9

  # width、height是feature map的大小
  shift_x = np.arange(0, width) * feat_stride
  shift_y = np.arange(0, height) * feat_stride

  # 通过meshgrid生成坐标矩阵
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)

  # 先拉直，然后叠在一起，最后xy互换
  # 得到的shifts就是所有anchor再原图上的坐标
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  K = shifts.shape[0]
  # width changes faster, so here it is H, W, C
  # 计算出anchors，操作没看懂
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length

def generate_anchors_pre_tf(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
  shift_x = tf.range(width) * feat_stride # width
  shift_y = tf.range(height) * feat_stride # height
  shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
  sx = tf.reshape(shift_x, shape=(-1,))
  sy = tf.reshape(shift_y, shape=(-1,))
  shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
  K = tf.multiply(width, height)
  shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))

  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

  length = K * A
  anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))
  
  return tf.cast(anchors_tf, dtype=tf.float32), length
