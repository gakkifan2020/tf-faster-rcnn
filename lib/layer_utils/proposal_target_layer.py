# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps


#  根据gt，对rpn产生的proposal打上分类标签以及计算回归的偏差
def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # 这点rois的格式，可能和proposal_layer返回的blob对应，可能
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois
  # 去除gt_boxes的最后一列（label）,拼接第一列全是0的数组，然后并到all_rois中
  if cfg.TRAIN.USE_GT:
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
      (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))      # gt_boxes格式（x1, y1, x2, y2, cls_label）
    )
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = np.vstack((all_scores, zeros))

  num_images = 1
  # 每一张图片允许的roi区域batch
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  # 计算每一张图片的batch个roi中前景的数量
  fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)      # __C.TRAIN.FG_FRACTION = 0.25

  # Sample rois with classification labels and bounding box regression
  # targets
  '''
  bbox_target_data[label,tx,ty,tw,th]
  labels[0.5,0.6,0.5,0.7,0.9,....,0,0,0,0,0,0,0,....]
  rois是对all_rois重新排列了一下，前景在前，背景在后
  roi_scores是对all_scores重新排列了一下，前景在前，背景在后
  '''
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
    all_rois, all_scores, gt_boxes, fg_rois_per_image,
    rois_per_image, _num_classes)

  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(-1, 1)
  bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
  bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """
  # 获取bbox_target_data所有的类别
  clss = bbox_target_data[:, 0]
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  # 找出clss>0的，也就是找出前景编号(背景在_sample_rois函数已经全部标记为0)
  # 这个inds是下标，是clss中所有值大于0的下标
  inds = np.where(clss > 0)[0]
  # 遍历的是前景
  for ind in inds:
    # 有疑问？？？既然inds是下标，那ind就是唯一的
    # 那为啥还要根据对应下标分start和end,直接[0:4]就可以存储了
    # 因为bbox_targets每行只有一个数据(数据的意思是一个完整的坐标)
    cls = clss[ind]
    start = int(4 * cls)
    end = start + 4
    # 对ind行(clss中值>0的行)赋值
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS     # __C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""
  # 计算偏移量，用于box回归
  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  '''
  对ROI进行采样
  '''
  # overlaps: (rois x gt_boxes)
  # 计算rois和gt_boxes的overlaps
  # roi格式(0, x1, y1, x2, y2)，gt_box格式(x,y,x,y,label)
  # 只取对应的xyxy
  overlaps = bbox_overlaps(
    np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
    np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
  # 返回每一行最大那一列的下标，也就是rois对应overlap最大的gt_box的索引
  gt_assignment = overlaps.argmax(axis=1)
  # 一样，只不过返回的是值
  max_overlaps = overlaps.max(axis=1)
  # 对应最大gt_box的label
  labels = gt_boxes[gt_assignment, 4]

  # Select foreground RoIs as those with >= FG_THRESH overlap
  # max_overlaps>=0.5的记录为前景,返回的也是下标
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]      # __C.TRAIN.FG_THRESH = 0.5
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  # max_overlaps在[0.1,0.5]记录为背景
  bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &      # __C.TRAIN.BG_THRESH_HI = 0.5
                     (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]   # __C.TRAIN.BG_THRESH_LO = 0.1

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.size > 0 and bg_inds.size > 0:     # 如果前景、背景都存在
    # 下面的意思就是如果样本很多，则随机采样去除一些
    # 在anchor_target_layer里面有相同的操作
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
    fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.size < bg_rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
  elif fg_inds.size > 0:
    to_replace = fg_inds.size < rois_per_image
    fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = rois_per_image
  elif bg_inds.size > 0:
    to_replace = bg_inds.size < rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  # 将前景背景的下标拼起来
  keep_inds = np.append(fg_inds, bg_inds)
  # Select sampled values from various arrays:
  # 提取对应的labels，相当于重新排了一下，前景在前，背景在后
  labels = labels[keep_inds]
  # Clamp labels for the background RoIs to 0
  # 然后把背景的全部赋值为0
  labels[int(fg_rois_per_image):] = 0
  # 下两个提取对应的roi和得分
  rois = all_rois[keep_inds]
  roi_scores = all_scores[keep_inds]
  # 用_compute_targets函数把xyxy坐标转换成delta坐标 ，也就是计算偏移量

  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)     # gt_assignment = overlaps.argmax(axis=1)
  # 最后bbox_target_data格式[[label,tx,ty,tw,th],[label,tx,ty,tw,th]]

  # 根据bbox_target_data偏移量，计算出回归的label
  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
