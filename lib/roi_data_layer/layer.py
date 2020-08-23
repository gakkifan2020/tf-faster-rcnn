# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import time

class RoIDataLayer(object):
  """Fast R-CNN data layer used for training."""

  def __init__(self, roidb, num_classes, random=False):
    """Set the roidb to be used by this layer during training."""
    self._roidb = roidb
    self._num_classes = num_classes
    # Also set a random flag
    self._random = random
    self._shuffle_roidb_inds()      # 随机排列roidb

  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    # If the random flag is set, 
    # then the database is shuffled according to system time
    # Useful for the validation set
    if self._random:      # 生成随机数种子
      st0 = np.random.get_state()
      millis = int(round(time.time() * 1000)) % 4294967295
      np.random.seed(millis)
    
    if cfg.TRAIN.ASPECT_GROUPING:     # 使用纵横比对图像重排
      # __C.TRAIN.ASPECT_GROUPING = False
      widths = np.array([r['width'] for r in self._roidb])
      heights = np.array([r['height'] for r in self._roidb])
      horz = (widths >= heights)
      vert = np.logical_not(horz)
      horz_inds = np.where(horz)[0]
      vert_inds = np.where(vert)[0]
      inds = np.hstack((
          np.random.permutation(horz_inds),
          np.random.permutation(vert_inds)))
      inds = np.reshape(inds, (-1, 2))
      row_perm = np.random.permutation(np.arange(inds.shape[0]))
      inds = np.reshape(inds[row_perm, :], (-1,))
      self._perm = inds     # 把roidb的索引打乱，造成的shuffle，打乱的索引存储的地方

    else:     # 将np.arange(len(self._roidb))打乱
      self._perm = np.random.permutation(np.arange(len(self._roidb)))
    # Restore the random state
    if self._random:
      np.random.set_state(st0)
      
    self._cur = 0     # 相当于一个指向_perm的指针，每次取走图片后，他会跟着变化

  # 依次取IMS_PER_BATCH大小个roi的index
  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    # IMS_PER_BATCH应该在train_rpn里面改成了1
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):     # 越界
      self._shuffle_roidb_inds()

    db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]     # 下一次roidb的下标
    # 相当于一个指向_perm的指针，随着取走图片的个数而变化
    self._cur += cfg.TRAIN.IMS_PER_BATCH

    return db_inds

  def _get_next_minibatch(self):
    """Return the blobs to be used for the next minibatch.
        并没有搜到 cfg.TRAIN.USE_PREFETCH
    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
    # 获取下一批图片的索引
    db_inds = self._get_next_minibatch_inds()
    # 把对应引索的图像信息（dict）取出来，放入一个列表中
    minibatch_db = [self._roidb[i] for i in db_inds]      # 我感觉里面只有一个图像的信息'.'！
    return get_minibatch(minibatch_db, self._num_classes)
      
  def forward(self):
    """Get blobs and copy them into this layer's top blob vector."""
    blobs = self._get_next_minibatch()
    return blobs
