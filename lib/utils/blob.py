# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

# ims是缩放后的多张图片  list
# 作用：将缩放后的图片信息存到blob中
def im_list_to_blob(ims):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  # ims里不同图片的shape可能不一样，取出其最大值
  max_shape = np.array([im.shape for im in ims]).max(axis=0)
  num_images = len(ims)
  # 构建一个全0的array，3代表BGR通道
  blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
  # 然后把图片信息赋值到blob中
  for i in range(num_images):
    im = ims[i]
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  return blob

# 对图片进行缩放，返回缩放后的img和比例
def prep_im_for_blob(im, pixel_means, target_size, max_size):
  """Mean subtract and scale an image for use in a blob."""
  im = im.astype(np.float32, copy=False)
  # Pixel mean values (BGR order) as a (1, 1, 3) array
  # 减去3通道的平均值
  im -= pixel_means     # __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
  im_shape = im.shape    # (w,h,3)
  # 比较长宽获得最大最小值
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  # 缩放比例，距离目标尺寸的比例
  im_scale = float(target_size) / float(im_size_min)
  # Prevent the biggest axis from being more than MAX_SIZE
  # 防止最长的边超过max_size
  if np.round(im_scale * im_size_max) > max_size:     # __C.TRAIN.MAX_SIZE = 1200
    im_scale = float(max_size) / float(im_size_max)
  # 对im进行缩放,缩放比例为im_scale
  im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                  interpolation=cv2.INTER_LINEAR)

  return im, im_scale
