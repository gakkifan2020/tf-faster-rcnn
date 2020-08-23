# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

'''
roidb是由字典组成的list，roidb[img_index]包含了该图片索引所包含到roi信息，下面以roidb[img_index]为例说明：
boxes：box位置信息，box_num*4的np array
gt_overlaps：所有box在不同类别的得分，box_num*class_num矩阵
gt_classes：所有box的真实类别，box_num长度的list
filpped：是否翻转
max_overlaps：每个box的在所有类别的得分最大值，box_num长度
max_classes：每个box的得分最高所对应的类，box_num长度
image：该图片的路径，字符串
bbox_targets:每个box的类别，以及与最接近的gt-box的4个方位偏移
'''

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  # 获取roidb中保存的图片数量
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  '''
  # Scale to use during training (can list multiple scales)
  # The scale is the pixel size of an image's shortest side
  __C.TRAIN.SCALES = (800,)
  '''
  # 随即生成num_images个下标，对应的是__C.TRAIN.SCALES的值
  # __C.TRAIN.SCALES : 图片被缩放的target_size列表
  # 因为config里面设置__C.TRAIN.SCALES = (800,)，所以得到的random_scale_inds是全0
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  # 要求batchsize必须是图片数量的整数倍
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  # 得到blob和缩放比例。roidb中图片经过了减去均值、缩放操作
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
  # 存入blob字典
  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:      # __C.TRAIN.USE_ALL_GT = True
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]      # 获得所有box真实类别的下标
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  # 整合gt_box信息
  '''
  这点全部用的是roidb[0]，看了其他的函数
  传进来的roidb是通过_get_next_minibatch获取pre-minbatch里面的图片
  然后通过_get_next_minibatch_inds获取图片的下标
  虽然可以改变__C.TRAIN.IMS_PER_BATCH = 2，使每个pre_minbatch为2
  但在train_rpn里面将其修改为1，所以__C.TRAIN.IMS_PER_BATCH = 1
  也就是说其实传进来的roidb只有一个图片的信息，所以下面都用的roidb[0]
  '''
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)

  '''
  blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
  im_blob.shape[1]代表w, im_blob.shape[2]代表h, im_scales[0]代表缩放比例
  '''
  return blobs

# 对roidb的图像进行缩放，并返回blob、和缩放比例
def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    # roidb[i]['image']这是一个路径,获取路径图片(3维的)
    im = cv2.imread(roidb[i]['image'])
    # 如果之前roi有翻转，水平翻转该图片
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    # __C.TRAIN.SCALES = (800,)
    # 因为config就设置了800，没有多的，所有的target_size均为800
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    # 对图像进行缩放，返回缩放后的image以及缩放比例
    # prep_im_for_blob再blob.py中
    # __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    # 保存图片的缩放比例
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  # im_list_to_blob ：将缩放后的图片放入blob中.（返回blob）
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
