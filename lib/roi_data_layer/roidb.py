# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import PIL

'''
roidb是由字典组成的list，roidb[img_index]包含了该图片索引所包含到roi信息，下面以roidb[img_index]为例说明：
boxes：box位置信息，box_num*4的np array
gt_overlaps：所有box在不同类别的得分，box_num*class_num矩阵
gt_classes：所有box的真实类别，box_num长度的list
filpped：是否翻转
max_overlaps：每个box的在所有类别的得分最大值，box_num长度
max_classes：每个box的得分最高所对应的类，box_num长度
image：图片的路径
width：图片的宽
height：图片的高
'''
# 为roidata添加一些说明性的附加属性
def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """
  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):
    # 获取每张图片的尺寸
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]
  '''
  image_path_at调用了image_path_from_index得到图片路径# /home/liubo/tf-faster-rcnn1/data/VOCdevkit/VOC2007/JPEGImages/000001.JPG
  image_index调用_load_image_set_index获取图片索引也就是得到/VOCdevkit2007/VOC2007/ImageSets/Main/{image_set}.txt里面的所有内容
  详细代码在pascal_voc.py中(imdb中只是实现了一些接口)
  pascal_voc.py代码解读：https://blog.csdn.net/qq_33193309/article/details/98615621
  '''
  for i in range(len(imdb.image_index)):
    # roidb中的'image'保存的是图片地址
    roidb[i]['image'] = imdb.image_path_at(i)
    if not (imdb.name.startswith('coco')):
      # 由获得的sizes对宽高进行赋值
      roidb[i]['width'] = sizes[i][0]
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    # gt_overlaps：所有box在不同类别的得分，box_num*class_num矩阵
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    # 每个box在所有类别的最大得分
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    # 每个box在所有类别的最大得分的下标，也就是最高分对应的类
    max_classes = gt_overlaps.argmax(axis=1)
    # max_classes：每个box的得分最高所对应的类，box_num长度
    roidb[i]['max_classes'] = max_classes
    # max_overlaps：每个box的在所有类别的得分最大值，box_num长度
    roidb[i]['max_overlaps'] = max_overlaps
    '''
    也就是这几个字典值，都可以通过'gt_overlaps'字典得到
    '''
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    # 做一个检查，0为背景，1为前景
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)
