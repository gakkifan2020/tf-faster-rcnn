# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

'''
一些关于box的一些操作
'''

# 获取不重复框的数组指标，过滤重复框
def unique_boxes(boxes, scale=1.0):
  """Return indices of unique boxes."""
  # 不是很懂它的操作
  # 我的理解是它构造了一个hash函数，用了round函数，对于一些近似的点，乘上它们对应的权值，得到一个hash值，最后保留唯一的hash值
  # 例如例如[2.6,3.2]和[2.9,3.4]经过round之后乘上对应的权值[1,10],结果均为33,说明两个box比较接近，然后过滤掉其中一个
  v = np.array([1, 1e3, 1e6, 1e9])
  hashes = np.round(boxes * scale).dot(v)
  # 保留唯一框的索引
  _, index = np.unique(hashes, return_index=True)
  return np.sort(index)

# 由xywh的格式转为xmin,ymin,xmax,ymax的格式
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

# 由xmin,ymin,xmax,ymax的格式转为xywh的格式
def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

# validata_boxes函数用于去除无效框，即超出边界或者上下左右的坐标不满足几何关系的点，
def validate_boxes(boxes, width=0, height=0):
  """Check that a set of boxes are valid."""
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]
  assert (x1 >= 0).all()
  assert (y1 >= 0).all()
  assert (x2 >= x1).all()
  assert (y2 >= y1).all()
  assert (x2 < width).all()
  assert (y2 < height).all()

# 过滤尺寸较小的box
def filter_small_boxes(boxes, min_size):
  # 获得box的w和h
  w = boxes[:, 2] - boxes[:, 0]
  h = boxes[:, 3] - boxes[:, 1]
  # 获取的w和h均要大于min_size
  keep = np.where((w >= min_size) & (h > min_size))[0]
  return keep
