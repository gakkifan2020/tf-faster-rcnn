# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import PIL
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from model.config import cfg

'''
imdb class为所有数据集的父类，包含了所有数据集共有的特性。
例如：数据集名称（name）、数据集类名列表（classes）、数据集的文件名列表（_image_index）、roi集合、config
'''

'''
roidb是由字典组成的list，roidb[img_index]包含了该图片索引所包含到roi信息，下面以roidb[img_index]为例说明：
boxes：box位置信息，box_num*4的np array
gt_overlaps：所有box在不同类别的得分，box_num*class_num矩阵
gt_classes：所有box的真实类别，box_num长度的list
filpped：是否翻转
max_overlaps：每个box的在所有类别的得分最大值，box_num长度
max_classes：每个box的得分最高所对应的类，box_num长度
'''

class imdb(object):
  """Image database."""

  def __init__(self, name, classes=None):
    self._name = name   # 数据集名称
    self._num_classes = 0     # 数据集类别个数
    if not classes:
      self._classes = []
    else:
      self._classes = classes     # 数据集类名列表
    self._image_index = []        # 数据集图片文件名列表 例如 data/VOCdevkit2007/VOC2007/ImageSets/Main/{image_set}.txt
    self._obj_proposer = 'gt'     # 这是一个字典，里面包含了gt_box、真实标签、gt_overlaps和翻转标签 flipped: true,代表图片被水平反转
    self._roidb = None
    self._roidb_handler = self.default_roidb      # roi数据列表
    # Use this dict for storing dataset specific config options
    self.config = {}

  @property
  def name(self):
    return self._name

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def image_index(self):
    return self._image_index

  # 返回ground-truth每个ROI构成的数据集
  @property
  def roidb_handler(self):
    return self._roidb_handler

  @roidb_handler.setter
  def roidb_handler(self, val):
    self._roidb_handler = val

  def set_proposal_method(self, method):
    method = eval('self.' + method + '_roidb')
    self.roidb_handler = method

  @property
  def roidb(self):
    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes
    #   gt_overlaps
    #   gt_classes
    #   flipped
    # 如果已经有了，那么直接返回，没有就通过指针指向的函数生成
    if self._roidb is not None:
      return self._roidb
    self._roidb = self.roidb_handler()
    return self._roidb

  # cache_path用来生成roidb缓存文件的文件夹，用来存储数据集的roi
  @property
  def cache_path(self):
    cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
    if not os.path.exists(cache_path):
      os.makedirs(cache_path)
    return cache_path

  @property
  def num_images(self):
    return len(self.image_index)

  def image_path_at(self, i):
    raise NotImplementedError

  def default_roidb(self):
    raise NotImplementedError

  def evaluate_detections(self, all_boxes, output_dir=None):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    raise NotImplementedError

  # 返回图像的size[0]，即宽度值
  def _get_widths(self):
    return [PIL.Image.open(self.image_path_at(i)).size[0]
            for i in range(self.num_images)]

  # 对图像数据进行水平翻转，进行数据增强
  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    '''
    扩展下copy()
    例dic = {'name': 'liubo', 'num': [1, 2, 3]}
    dic1 = dic
    dic2 = dic.copy()
    dic['name'] = '123123'  # 修改父对象dic
    dic['num'].remove(1)  # 修改父对象dic中的[1, 2, 3]列表子对象
    # 输出结果
    print(dic)  # {'name': '123123', 'age': [2, 3]}
    print(dic1)  # {'name': '123123', 'age': [2, 3]}
    print(dic2)  # {'name': 'liubo', 'age': [2, 3]}
    也就是说用copy，父对象不会因为dic的改变而改变，而子对象会
    '''
    for i in range(num_images):
      # roidb['boxes']有四个元素，分别代表roi的四个点xmin,ymin,xmax,ymax
      boxes = self.roidb[i]['boxes'].copy()
      # 假设boxes=([1,2,4,2]),oldx1=[1],oldx2=[4]
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      # widths[i]代表宽
      # 变换坐标，将xmax变成xmin,xmin变成xmax关于x=xmin对称的点,翻转后boxes变成[-2,2,1,4]
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()     # 翻转后的xmax肯定大于xmin
      entry = {'boxes': boxes,
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'gt_classes': self.roidb[i]['gt_classes'],
               'flipped': True}     # flipped变为True代表水平翻转
      # 将翻转后的图像数据也加入
      self.roidb.append(entry)
    # 因为是按顺序翻转，所有只需要将原来的扩大一倍，roidb里面的图片信息索引与image_index索引对应
    self._image_index = self._image_index * 2

  # 根据RP来确定候选框的recall值
  def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                      area='all', limit=None):
    """Evaluate detection proposal recall metrics. 评估recall值
    Returns:
        results: dictionary of results with keys  返回结果是下面4个指标的字典
            'ar': average recall
            'recalls': vector recalls at each IoU overlap threshold
            'thresholds': vector of IoU overlap thresholds
            'gt_overlaps': vector of all ground-truth overlaps
    """
    # Record max overlap value for each gt box 记录每个gt_box的最大重叠值
    # Return vector of overlap values

    # 制定了一些area范围，先根据area找到index，再通过area_ranges[index]找到范围
    areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
             '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
    area_ranges = [[0 ** 2, 1e5 ** 2],  # all
                   [0 ** 2, 32 ** 2],  # small
                   [32 ** 2, 96 ** 2],  # medium
                   [96 ** 2, 1e5 ** 2],  # large
                   [96 ** 2, 128 ** 2],  # 96-128
                   [128 ** 2, 256 ** 2],  # 128-256
                   [256 ** 2, 512 ** 2],  # 256-512
                   [512 ** 2, 1e5 ** 2],  # 512-inf
                   ]
    assert area in areas, 'unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = np.zeros(0)
    num_pos = 0
    '''
    roidb是由字典组成的list，roidb[img_index]包含了该图片索引所包含到roi信息，下面以roidb[img_index]为例说明：
    boxes：box位置信息，box_num*4的np array
    gt_overlaps：所有box在不同类别的得分，box_num*class_num矩阵
    gt_classes：所有box的真实类别，box_num长度的list
    filpped：是否翻转
    max_overlaps：每个box的在所有类别的得分最大值，box_num长度
    max_classes：每个box的得分最高所对应的类，box_num长度
    '''
    for i in range(self.num_images):
      # Checking for max_overlaps == 1 avoids including crowd annotations
      # (...pretty hacking :/)
      # 首先获取roidb中的值
      max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)      # 取出每一行最大的得分
      # 得到满足gt_classes>0&&max_gt_overlaps=1对应的第0列
      gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                         (max_gt_overlaps == 1))[0]     # 如果max_gt_overlaps最大的得分==1，这个下表对应的就是gt
      gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
      gt_areas = self.roidb[i]['seg_areas'][gt_inds]      # seg_areas：box的面积
      valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                               (gt_areas <= area_range[1]))[0]      # 满足范围之内的面积
      gt_boxes = gt_boxes[valid_gt_inds, :]     # 只取在范围之内的
      num_pos += len(valid_gt_inds)

      if candidate_boxes is None:
        # If candidate_boxes is not supplied, the default is to use the
        # non-ground-truth boxes from this roidb
        # 所有满足roidb[i]['gt_classes']==0的横坐标
        non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
        boxes = self.roidb[i]['boxes'][non_gt_inds, :]
      else:
        boxes = candidate_boxes[i]
      if boxes.shape[0] == 0:
        continue
      if limit is not None and boxes.shape[0] > limit:
        boxes = boxes[:limit, :]

      # 计算当前图像的boxes与gtboxes的IOU overlap
      overlaps = bbox_overlaps(boxes.astype(np.float),
                               gt_boxes.astype(np.float))

      _gt_overlaps = np.zeros((gt_boxes.shape[0]))
      # 对于每一张图片内的每一个gt_boxes，都要找到最大的IoU
      for j in range(gt_boxes.shape[0]):
        # find which proposal box maximally covers each gt box
        # 找到某一个框能最大限度的覆盖gt
        argmax_overlaps = overlaps.argmax(axis=0)     # 每一列最大值的下标
        # and get the iou amount of coverage for each gt box
        # 获得某个框对于gt的覆盖最大值
        max_overlaps = overlaps.max(axis=0)     # 每一列的最大值
        # find which gt box is 'best' covered (i.e. 'best' = most iou)
        gt_ind = max_overlaps.argmax()      # 得到最大值所对应的下标
        gt_ovr = max_overlaps.max()
        assert (gt_ovr >= 0)
        # find the proposal box that covers the best covered gt box
        # 找到这个最大覆盖的gt_index所对应的box_index
        box_ind = argmax_overlaps[gt_ind]
        # record the iou coverage of this gt box
        # 记录gt_box的IOU值
        _gt_overlaps[j] = overlaps[box_ind, gt_ind]
        '''
        上面一系列操作可以理解为找到数组中最大值的横坐标box_ind和纵坐标gt_ind
        所以_gt_overlaps[j]必定是等于我最初求得的gt_ovr,方法复杂但降低了复杂度
        举例：
        overlaps=[[1, 4, 2, 3], #为了方便都用整数
                  [5, 2, 4, 1], 
                  [3, 1, 8, 4]]
         目标找到gt_ovr=8，box_ind=2,gt_ind=2
         argmax_overlaps = overlaps.argmax(axis=0)  # 每一列最大值的下标 [1,0,2,2]
         max_overlaps = overlaps.max(axis=0)  # 每一列的最大值 [5,4,8,4]
         gt_ind = max_overlaps.argmax()  # 得到最大值所对应的下标 2
         gt_ovr = max_overlaps.max()  # 最大值 8
         box_ind = argmax_overlaps[gt_ind] # 2
        '''
        assert (_gt_overlaps[j] == gt_ovr)
        # mark the proposal box and the gt box as used
        # 标记该点使用过，并且把对应的行和列都改为-1
        overlaps[box_ind, :] = -1
        overlaps[:, gt_ind] = -1
      # append recorded iou coverage level
      # 类似于拼接，把所有获得的IOU值都保存在1行n列的数组中
      gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
    gt_overlaps = np.sort(gt_overlaps)
    # 生成threshold来进行不同间隔内的recall计算
    if thresholds is None:
      step = 0.05
      thresholds = np.arange(0.5, 0.95 + 1e-5, step)
    recalls = np.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
      recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
            'gt_overlaps': gt_overlaps}

  '''
  roidb是由字典组成的list，roidb[img_index]包含了该图片索引所包含到roi信息，下面以roidb[img_index]为例说明：
  boxes：box位置信息，box_num*4的np array
  gt_overlaps：所有box在不同类别的得分，box_num*class_num矩阵
  gt_classes：所有box的真实类别，box_num长度的list
  filpped：是否翻转
  max_overlaps：每个box的在所有类别的得分最大值，box_num长度
  max_classes：每个box的得分最高所对应的类，box_num长度
  '''

  def create_roidb_from_box_list(self, box_list, gt_roidb):
    # box_list的长度必须跟图片的数量相同，相当于为每个图片创造roi，各图像要一一对应
    assert len(box_list) == self.num_images, \
      'Number of boxes must match number of ground-truth images'
    roidb = []
    for i in range(self.num_images):
      # 遍历每张图片，boxes代表当前图像中的box
      boxes = box_list[i]
      # 代表当前boxes中box的个数
      num_boxes = boxes.shape[0]
      # overlaps的shape始终为：num_boxes × num_classes 。
      overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

      if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
        # 获取所有的box和class
        gt_boxes = gt_roidb[i]['boxes']
        gt_classes = gt_roidb[i]['gt_classes']
        # 计算当前图像的boxes与gtboxes的IOU overlap
        # shape为num_boxes × num_gtboxes
        gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                    gt_boxes.astype(np.float))
        argmaxes = gt_overlaps.argmax(axis=1)
        maxes = gt_overlaps.max(axis=1)
        I = np.where(maxes > 0)[0]      # 所有满足值的横坐标
        '''
        上述操作获取每一行的最大值、最大值下标、最大值的横坐标（满足最大值>0）
        然后将满足最大值的横坐标对应的max值赋值给overlaps对应的坐标
        举例
        gt_overlaps = [[1, 4, 2, 5], 
                      [-1, -1, -1, -1], 
                      [7, 2, -1, 3]]
        argmaxes = gt_overlaps.argmax(axis=1) # [3, 0, 0]
        maxes = gt_overlaps.max(axis=1) # [5, -1, 7]
        I=np.where(maxes > 0)[0] # [0, 2]
        对应赋值到下面
        overlaps[0, gt_classes[3]] = 5
        overlaps[2, gt_classes[0]] = 7
        '''
        overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
      '''
      a = np.array([[3, 1], [5, 6]])
      print(scipy.sparse.csr_matrix(a))
      (0, 0)	3
      (0, 1)	1
      (1, 0)	5
      (1, 1)	6
      '''
      # gt_overlaps：所有box在不同类别的得分，box_num*class_num矩阵

      overlaps = scipy.sparse.csr_matrix(overlaps)
      roidb.append({
        'boxes': boxes,
        'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
        'gt_overlaps': overlaps,
        'flipped': False,
        'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
      })
    return roidb

  # 将a b两个roidb归并为一个roidb
  @staticmethod
  def merge_roidbs(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
      a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
      a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                      b[i]['gt_classes']))
      a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                 b[i]['gt_overlaps']])
      a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                     b[i]['seg_areas']))
    return a

  def competition_mode(self, on):
    """Turn competition mode on or off."""
    pass
