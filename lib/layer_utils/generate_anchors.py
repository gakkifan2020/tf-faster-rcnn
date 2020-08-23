# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])
# base_size=16,因为经过卷积之后，M*N的矩阵对应的特征图为M/16*N/16，（4个pooling）不知道是不是这样
# 所以feature map上一点对应到原图就是16*16的区域
# ratios表示宽高比为：1:2,1:1,2:1
# 2**x表示:scales:[2^3 2^4 2^5],即:[8 16 32]
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
  """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """

  base_anchor = np.array([1, 1, base_size, base_size]) - 1      # [0,0,15,15]
  ratio_anchors = _ratio_enum(base_anchor, ratios)              # 枚举各种宽高比
  anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) # 对已经进行宽高比改变的，在进行3种尺度改变
                       for i in range(ratio_anchors.shape[0])])
  return anchors

# _whctrs返回一个anchor的中心点(x,y)和w,h
def _whctrs(anchor):
  """
  Return width, height, x center, and y center for an anchor (window).
  """

  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr

# 输出各个anchors
def _mkanchors(ws, hs, x_ctr, y_ctr):
  """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """

  ws = ws[:, np.newaxis]      # newaxis:将数组转置
  hs = hs[:, np.newaxis]
  anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                       y_ctr - 0.5 * (hs - 1),
                       x_ctr + 0.5 * (ws - 1),
                       y_ctr + 0.5 * (hs - 1)))
  return anchors


def _ratio_enum(anchor, ratios):
  """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """
  # 列举关于一个anchor的三种宽高比 1:2,1:1,2:1
  w, h, x_ctr, y_ctr = _whctrs(anchor)      # _whctrs返回一个anchor的中心点(x,y)和w,h
  size = w * h      # 16*16
  size_ratios = size / ratios     # 16*16=256/[0.5,1,2]=[512,256,128]
  # round()方法返回x的四舍五入的数字，sqrt()方法返回数字x的平方根
  ws = np.round(np.sqrt(size_ratios))     # [23,16,11]
  hs = np.round(ws * ratios)              # [12,16,22]
  # 就是将中心(x,y),w,h转换为左上和右下坐标
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)      # 输出各个预测窗口
  return anchors


def _scale_enum(anchor, scales):
  """
  Enumerate a set of anchors for each scale wrt an anchor.
  """
  # 列举关于一个anchor的三种尺度 128*128,256*256,512*512
  # 3种尺度，3种宽高比，所有9种
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


if __name__ == '__main__':
  import time

  t = time.time()
  a = generate_anchors()
  print(time.time() - t)
  print(a)
  from IPython import embed;

  embed()
