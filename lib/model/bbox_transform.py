# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# 计算与anchor有最大IOU的GT的偏移量
# ex_rois：表示anchor；gt_rois：表示GT
def bbox_transform(ex_rois, gt_rois):     # 这里的gt是：与anchor最匹配的gt，不是所有的gt .anchor_target_layer有调用
  # anchor的坐标
  # 计算每一个anchor的w和h
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
  # 计算每一个anchor的中心点x和y
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  # gt的坐标
  # 计算每一个gt的w和h
  gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
  gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0

  # 计算每一个gt的中心点x和y
  gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
  gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

  # 偏移量
  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dw = np.log(gt_widths / ex_widths)
  targets_dh = np.log(gt_heights / ex_heights)

  targets = np.vstack(
    (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
  return targets

# 根据anchor和偏移量得到一个改善后的anchor
def bbox_transform_inv(boxes, deltas):
  # 假设图像尺寸=[h,w,3]
  # boxes=[h*w*9,4] 预设anchors的坐标 [xmin,ymin,xmax,ymax]
  # deltas=[h*w*9,4] 预测的坐标偏移量
  if boxes.shape[0] == 0:
    return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

  # 计算宽高和中心点坐标
  boxes = boxes.astype(deltas.dtype, copy=False)
  # 改为(x,y,w,h)的格式
  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  # ::单独提取为一列
  # 0::4表示先取第一个元素，以后每4个取一个
  dx = deltas[:, 0::4]
  dy = deltas[:, 1::4]
  dw = deltas[:, 2::4]
  dh = deltas[:, 3::4]

  # 计算预测后的中心点和w,h
  pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]      # np.newaxis,表示将widths增加一维，使得其能够相加
  pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
  pred_w = np.exp(dw) * widths[:, np.newaxis]
  pred_h = np.exp(dh) * heights[:, np.newaxis]

  pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
  # x1
  pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
  # y1
  pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
  # x2
  pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
  # y2
  pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

  return pred_boxes


def clip_boxes(boxes, im_shape):
  """
  Clip boxes to image boundaries.
      调整boxes的坐标，使其全部在图像的范围内,全部大于0，小于图像宽高
  """

  # x1 >= 0
  boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
  return boxes



def bbox_transform_inv_tf(boxes, deltas):
  boxes = tf.cast(boxes, deltas.dtype)
  widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
  heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
  ctr_x = tf.add(boxes[:, 0], widths * 0.5)
  ctr_y = tf.add(boxes[:, 1], heights * 0.5)

  dx = deltas[:, 0]
  dy = deltas[:, 1]
  dw = deltas[:, 2]
  dh = deltas[:, 3]

  pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
  pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
  pred_w = tf.multiply(tf.exp(dw), widths)
  pred_h = tf.multiply(tf.exp(dh), heights)

  pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
  pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
  pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
  pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

  return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def clip_boxes_tf(boxes, im_info):
  b0 = tf.maximum(tf.minimum(boxes[:, 0], im_info[1] - 1), 0)
  b1 = tf.maximum(tf.minimum(boxes[:, 1], im_info[0] - 1), 0)
  b2 = tf.maximum(tf.minimum(boxes[:, 2], im_info[1] - 1), 0)
  b3 = tf.maximum(tf.minimum(boxes[:, 3], im_info[0] - 1), 0)
  return tf.stack([b0, b1, b2, b3], axis=1)


