# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf

import tensorflow as tf
import numpy as np
import numpy.random as npr

# 对rpn计算结果roi proposals的优选
# 这个和proposal_layer是对应的
# 当TEST.MODE = 'top'使用proposal_top_layer，
# 当TEST.MODE = 'nms'使用proposal_layer，
# 上面是自认为的.....

def proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
  """A layer that just selects the top region proposals
     without using non-maximal suppression,
     For details please see the technical report
  """
  # __C.TEST.RPN_TOP_N = 5000 仅TEST.MODE = 'top' 的时候使用
  # __C.TEST.MODE = 'nms'
  rpn_top_n = cfg.TEST.RPN_TOP_N

  # 提取概率分数
  scores = rpn_cls_prob[:, :, :, num_anchors:]

  # 对提取的预测狂reshape # rpn_bbox_pred：RPN层输出的box的取值，即：tx,ty,tw,th
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))

  # 统计有多少个框
  length = scores.shape[0]
  if length < rpn_top_n:      # 如果框小于5000个，需要随即重复采样，让他变成5000个
    # Random selection, maybe unnecessary and loses good proposals
    # But such case rarely happens
    top_inds = npr.choice(length, size=rpn_top_n, replace=True)
  else:
    # 从大到小排序，取列索引
    top_inds = scores.argsort(0)[::-1]
    # 取前大的5000个
    top_inds = top_inds[:rpn_top_n]
    top_inds = top_inds.reshape(rpn_top_n, )

  # Do the selection here
  # 选择/重排
  # 按照索引提取anchor数据
  anchors = anchors[top_inds, :]
  rpn_bbox_pred = rpn_bbox_pred[top_inds, :]
  scores = scores[top_inds]

  # Convert anchors into proposals via bbox transformations
  # bbox_transform_inv : 根据anchor和偏移量计算proposals
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)

  # Clip predicted boxes to image
  # clip_boxes : proposals的边界限制在图片内
  proposals = clip_boxes(proposals, im_info[:2])

  # Output rois blob
  # Our RPN implementation only supports a single input image, so all
  # batch inds are 0
  # 和 proposal_layer 一样，多出来一列0，然后拼接
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
  return blob, scores


def proposal_top_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
  """A layer that just selects the top region proposals
     without using non-maximal suppression,
     For details please see the technical report
  """
  rpn_top_n = cfg.TEST.RPN_TOP_N

  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))
  scores = tf.reshape(scores, shape=(-1,))

  # Do the selection here
  top_scores, top_inds = tf.nn.top_k(scores, k=rpn_top_n)
  top_scores = tf.reshape(top_scores, shape=(-1, 1))
  top_anchors = tf.gather(anchors, top_inds)
  top_rpn_bbox = tf.gather(rpn_bbox_pred, top_inds)
  proposals = bbox_transform_inv_tf(top_anchors, top_rpn_bbox)

  # Clip predicted boxes to image
  proposals = clip_boxes_tf(proposals, im_info[:2])

  # Output rois blob
  # Our RPN implementation only supports a single input image, so all
  # batch inds are 0
  proposals = tf.to_float(proposals)
  batch_inds = tf.zeros((rpn_top_n, 1))
  blob = tf.concat([batch_inds, proposals], 1)
  return blob, top_scores
