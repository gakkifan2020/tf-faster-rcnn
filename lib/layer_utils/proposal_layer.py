# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from model.nms_wrapper import nms

# 将anchor变成proposals，根据pre_nms_topN和post_nms_topN进行选择
def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  # cfg_key代表TRAIN还是TEST
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  '''
  pre_nms_topN: 在NMS处理之前，分数在前面的rois
  post_nms_topN: 在NMS处理之后，分数在前面的rois
  nms_thresh: NMS的阈值
  '''
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH      # __C.TRAIN.RPN_NMS_THRESH = 0.7

  # Get the scores and bounding boxes
  # 其中第四维度前9位是背景的分数，后9位是前景的分数
  # 假设rpn_cls_prob = (1,38,50,18)
  scores = rpn_cls_prob[:, :, :, num_anchors:]      # scores = (1,38,50,9)
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))    # rpn_bbox_pred = （1,38,50,36）->(17100,4)
  scores = scores.reshape((-1, 1))                  # scores = (17100,1)
  # bbox_transform_inv 根据anchor和偏移量计算proposals
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  # clip_boxes作用：调整boxes的坐标，使其全部在图像的范围内
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  # 首先变成一维，然后argsort返回数组值从小到大的索引值,然后加上[::-1]，翻转序列
  # order保存数组值从大到小的索引值
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:      # 只取前pre_nms_topN
    order = order[:pre_nms_topN]
  # order对应的是下标，然后把得分最高的前pre_nms_topN的区域保存
  proposals = proposals[order, :]
  # 只保存前pre_nms_topN个得分
  scores = scores[order]

  # Non-maximal suppression
  # 非极大值抑制 np.hstack把他们拼接成(区域 分数)的形式
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  # 在nms之后，选择前post_nms_topN个
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  # 这点多出来一个batch_inds，拼接之后blob的第一列全是0，不知道后面是不是有什么操作。。。
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores


def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  scores = tf.reshape(scores, shape=(-1,))
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

  proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
  proposals = clip_boxes_tf(proposals, im_info[:2])

  # Non-maximal suppression
  indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)

  boxes = tf.gather(proposals, indices)
  boxes = tf.to_float(boxes)
  scores = tf.gather(scores, indices)
  scores = tf.reshape(scores, shape=(-1, 1))

  # Only support single image as input
  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)

  return blob, scores


