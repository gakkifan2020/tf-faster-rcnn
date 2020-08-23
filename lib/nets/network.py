# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from model.config import cfg

class Network(object):
  def __init__(self):
    self._predictions = {}      # 保存预测结果
    self._losses = {}           # 保存损失值
    self._anchor_targets = {}   # 保存预设anchor的坐标
    self._proposal_targets = {}
    self._layers = {}           # 保存网络
    self._gt_image = None
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = []
    self._event_summaries = {}
    self._variables_to_fix = {}   # 保存fine-tune时需要固定值的变量

  # 还原图像，加上均值并进行通道改变
  def _add_gt_image(self):
    # add back mean
    # 预处理时去均值，此处重新加回均值
    image = self._image + cfg.PIXEL_MEANS
    # BGR to RGB (opencv uses BGR)
    # 预处理时缩放图片，此处放大回去 im_info=[缩放后h，缩放后w，缩放比例]
    resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
    # 预处理时RGB转成BGR，此处转回RGB
    self._gt_image = tf.reverse(resized, axis=[-1])

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    if self._gt_image is None:
      self._add_gt_image()
    # image是绘制了box和类别文字后的图片
    image = tf.py_func(draw_bounding_boxes, 
                      [self._gt_image, self._gt_boxes, self._im_info],
                      tf.float32, name="gt_boxes")
    
    return tf.summary.image('GROUND_TRUTH', image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    # 作用是将输入的Tensor中0元素在所有元素中所占的比例计算并返回
    # 因为relu激活函数有时会大面积的将输入参数设为0，所以此函数可以有效衡量relu激活函数的有效性。
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      # 输入bottom为[b,h,w,2A]
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # to_caffe为[b,2A,h,w]

      # then force it to have channel 2
      # reshaped为[1,2,b*A*h,w]，且[0,0,b*A*h,w]对应bg，[0,1,b*A*h,w]对应fg
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back

      # to_tf为[1,b*A*h,w,2]
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      # 以上面to_tf为例，to_tf为[b,A*h,w,2]
      input_shape = tf.shape(bottom)
      # bottom_reshaped为[b*A*h*w,2] 所有像素点的9个anchor全部平铺成列，平铺时遍历的顺序依次是w,h,A,b
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

    # 对rpn计算结果roiproposals的优选
    # 当TEST.MODE = 'top'使用proposal_top_layer，
  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    # '''
    # 对rpn计算结果roi proposals的优选
    # 当TEST.MODE = 'top'使用proposal_top_layer
    # 当TEST.MODE = 'nms'使用proposal_layer
    # 默认使用nms，作者说top模式效果更好，但速度慢
    # '''
    with tf.variable_scope(name) as scope:
      if cfg.USE_E2E_TF:      # 使用tensorflow端到端实现，不用numpy
        rois, rpn_scores = proposal_top_layer_tf(
          rpn_cls_prob,
          rpn_bbox_pred,
          self._im_info,
          self._feat_stride,
          self._anchors,
          self._num_anchors
        )
      else:
        rois, rpn_scores = tf.py_func(proposal_top_layer,
                              [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                               self._feat_stride, self._anchors, self._num_anchors],
                              [tf.float32, tf.float32], name="proposal_top")
        
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

    return rois, rpn_scores

  # 当TEST.MODE = 'nms'使用proposal_layer，
  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      if cfg.USE_E2E_TF:
        rois, rpn_scores = proposal_layer_tf(
          rpn_cls_prob,
          rpn_bbox_pred,
          self._im_info,
          self._mode,
          self._feat_stride,
          self._anchors,
          self._num_anchors
        )
      else:
        rois, rpn_scores = tf.py_func(proposal_layer,
                              [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                               self._feat_stride, self._anchors, self._num_anchors],
                              [tf.float32, tf.float32], name="proposal")

      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  # 默认用_crop_pool_layer实现ROI Pooling
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]

  # roi_pooling
  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      # 提取batch_id,并转为一行，为全0，目的是指定所有crop均来自同一张图片（输入本来也就只有一张特征图）
      # rois为[batch_ids,xmin,ymin,xmax,ymax]
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bounding boxes
      bottom_shape = tf.shape(bottom)
      # self._feat_stride[0]=16 经过4次pooling
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      # 为了进行max_pooling，将范围扩大到14*14，这样经过下面的max_pooling得出的结果就是7*7
      pre_pool_size = cfg.POOLING_SIZE * 2      # __C.POOLING_SIZE = 7
      # crops为[num_boxes, crop_height, crop_width, depth]，一个bottom会输出num_boxes个图像
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  # dropout，概率为ratio
  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  # 根据gt给所有预设anchor计算标签和偏移量
  def _anchor_target_layer(self, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32],
        name="anchor_target")

      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels     # rpn_labels=[1, 1, A * height, width] 整张图所有预设anchor的标签
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets     # rpn_bbox_targets=[1, height, width, A * 4] 整张图所有anchor的偏移量
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights     # rpn_bbox_inside_weights=[1, height, width, A * 4] 在图片范围内的边框权重
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

      self._score_summaries.update(self._anchor_targets)

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores, name):
    with tf.variable_scope(name) as scope:
      rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
        proposal_target_layer,
        [rois, roi_scores, self._gt_boxes, self._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        name="proposal_target")

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

      # '''
      # rpn_rois=[2000,5] [batch_inds,xmin,ymin,xmax,ymax] batch_inds全0 2000在cfg中调整RPN_POST_NMS_TOP_N
      # rpn_scores=[2000] clsc层前景的softmax值
      # rois是对rpn_rois区分前背景后筛选出batch_size个，并且前景在前，背景在后重新排列
      # roi_scores是对rpn_scores的相同处理
      # labels是batch_size个区域的标签，前景区域的标签与IOU最大的GT的标签相同，背景的标签为0
      # bbox_targets=[batch_size，num_class*4] 正确类别的坐标为回归值，其余类别的坐标为0
      # '''

      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

      self._score_summaries.update(self._proposal_targets)

      return rois, roi_scores

  # 生成anchors
  def _anchor_component(self):
    # 生成每张图的anchor
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      # 卷积特征图的尺寸 self._feat_stride[0]=16 经过4次pooling
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
      # 默认使用tensorflow端到端，不使用numpy
      if cfg.USE_E2E_TF:
        anchors, anchor_length = generate_anchors_pre_tf(
          height,
          width,
          self._feat_stride,
          self._anchor_scales,
          self._anchor_ratios
        )
      else:
        anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                            [height, width,
                                             self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                            [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  def _build_network(self, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:     # __C.TRAIN.TRUNCATED = False
      # 均值为0，标准差为0.01的截断正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:     # 随机初始化
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    # 经过特征提取网络，初步提取特征
    net_conv = self._image_to_head(is_training)     # _image_to_head在resnet_v1.py实现
    with tf.variable_scope(self._scope, self._scope):
      # build the anchors for the image
      # 生成每张图所有的预设anchor坐标
      self._anchor_component()
      # region proposal network
      # RPN层，输出的是proposal_layer筛选后得分比较高的区域坐标
      # rois为[batch_ids,xmin,ymin,xmax,ymax]
      rois = self._region_proposal(net_conv, is_training, initializer)
      # region of interest pooling
      # 进行ROI Pooling，输出与rois数量相同个数的7*7特征图（相当于拷贝了len（rois）个net_conv,每个都只保留判断为前景的某一部分）
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
      else:
        raise NotImplementedError

    fc7 = self._head_to_tail(pool5, is_training)
    with tf.variable_scope(self._scope, self._scope):
      # region classification
      cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
                                                        initializer, initializer_bbox)

    self._score_summaries.update(self._predictions)

    # rois为[batch_ids,xmin,ymin,xmax,ymax] batch_size个RPN候选区域坐标
    # cls_prob：全连接分类softmax
    # bbox_pred：全连接回归
    return rois, cls_prob, bbox_pred

  # 对于回归的loss计算
  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    # 以rpn层为例：
    # bbox_pred  [b,h,w,A*4] reg层特征图
    # bbox_targets [1, height, width, A * 4] 整张图所有anchor的偏移量
    # sigma=3.0
    # dim=[1, 2, 3]
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    # tf.less(a,b) a<b返回真，否则返回假
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    # smoothL1_sign用于实现分段函数
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    # 求和并降为1维
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

  def _add_losses(self, sigma_rpn=3.0):
    with tf.variable_scope('LOSS_' + self._tag) as scope:
      # RPN, class loss
      # [1,b*A*h,w,2] cls层特征图的reshape
      rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
      # rpn_labels=[1, 1, A * height, width] 整张图所有预设anchor的标签 0为负样本 1为正样本 -1为无效样本
      rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
      # 取出有效样本
      rpn_select = tf.where(tf.not_equal(rpn_label, -1))
      # 获得有效样本的预测得分
      rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
      # 获得有效样本的标签
      rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
      # 计算交叉熵损失 RPN_BATCH_SIZE个anchor的均值
      rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

      # RPN, bbox loss
      # rpn_bbox_pred=[b,h,w,A*4] reg层特征图
      rpn_bbox_pred = self._predictions['rpn_bbox_pred']
      # rpn_bbox_targets=[1, height, width, A * 4] 整张图所有anchor的偏移量
      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
      # 权重rpn_bbox_inside_weights=__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

      # inside_weights全是1，没起作用，outside_weights为有效样本数量的倒数，起到在一个batch内取平均的作用
      # rpn每次只处理一张图片，即一个batch，在一张图片上又取了RPN_BATCH_SIZE个anchor，作用就是对这些anchor取平均
      rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

      # RCNN, class loss
      # RCNN部分，也就是RPN层之后的部分的分类损失
      cls_score = self._predictions["cls_score"]
      label = tf.reshape(self._proposal_targets["labels"], [-1])
      cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

      # RCNN, bbox loss
      # 边框回归损失
      bbox_pred = self._predictions['bbox_pred']
      bbox_targets = self._proposal_targets['bbox_targets']

      # bbox_inside_weights = __C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
      # bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
      # inside_weights，outside_weights都是1，没起到作用
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
      loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box

      loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
      # 获取正则化损失
      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
      # 总损失
      self._losses['total_loss'] = loss + regularization_loss

      self._event_summaries.update(self._losses)

    return loss

  def _region_proposal(self, net_conv, is_training, initializer):
    # RPN层 紧接在特征提取层之后 256通道 权重初始化等设置和特征提取网络是分开处理的，不共享统一参数空间  没有其他默认参数
    # '''b=1,根据程序推测RPN层每次只能处理一张图片，yml里IMS_PER_BATCH也确实设置为1'''
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
    # 添加直方图
    self._act_summaries.append(rpn)
    # 默认self._num_anchors = 9
    # rpn_cls_score=[b,h,w,A*2]
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')
    # change it so that the score has 2 as its channel size
    # rpn_cls_score_reshape=[1,b*A*h,w,2]
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    # rpn_cls_prob_reshape=[1,b*A*h,w,2]
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    # rpn_cls_pred=[b*A*h*w,1] 最大得分的序号，即对应的类别
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
    # rpn_cls_prob=[1,b*h,w,2*A]
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    if is_training:
      # 将anchor变成proposals，然后进行NMS，并根据在NMS前后根据pre_nms_topN和post_nms_topN筛选出rpn_cls_prob较高的框
      # 推测：rpn_cls_prop=[batch,h,w,9*2] rpn_bbox_pred=[batch,h,w,9*4]
      # 推测：rois=[batch_inds,proposal_xmin,proposal_ymin,proposal_xmax,proposal_ymax]
      # 推测：roi_scores=[scores]
      # rois=[2000,5] [batch_inds,xmin,ymin,xmax,ymax] batch_inds全0 2000在cfg中调整RPN_POST_NMS_TOP_N
      # roi_scores=[2000] clsc层前景的softmax值
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")

      # rpn_labels=[1,1,A*h,w] anchor的标签 0,1,-1
      rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
      # Try to have a deterministic order for the computing graph, for reproducibility
      with tf.control_dependencies([rpn_labels]):
        rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      else:
        raise NotImplementedError

    # 计算得到的中间结果保存到self._predictions里面，在计算loss的时候或者测试的时候都会用到
    self._predictions["rpn_cls_score"] = rpn_cls_score      # [b,h,w,A*2] cls层特征图
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape      # [1,b*A*h,w,2] cls层特征图的reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob      # [1,b*h,w,2*A] cls层特征图的softmax
    self._predictions["rpn_cls_pred"] = rpn_cls_pred      # [b*A*h*w,1] 类别序号
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred      # [b,h,w,A*4] reg层特征图
    self._predictions["rois"] = rois                        # batch_size个RPN输出的候选区域坐标

    return rois

  # 最后的回归和分类层
  def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
    cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    cls_prob = self._softmax_layer(cls_score, "cls_prob")
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
    bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred')

    self._predictions["cls_score"] = cls_score      # 全连接分类输出
    self._predictions["cls_pred"] = cls_pred      # 全连接预测的类别序号
    self._predictions["cls_prob"] = cls_prob      # 全连接softmax
    self._predictions["bbox_pred"] = bbox_pred      # 全连接回归输出

    return cls_prob, bbox_pred

  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None):
    raise NotImplementedError

  def create_architecture(self, mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    self._tag = tag

    self._num_classes = num_classes     # 类别数
    self._mode = mode                   # 模式，nms或top
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag != None

    # handle most of the regularizers here
    # 正则化参数
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    # 获得模型的输出
    with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)): 
      rois, cls_prob, bbox_pred = self._build_network(training)

    layers_to_output = {'rois': rois}

    for var in tf.trainable_variables():
      self._train_summaries.append(var)

    if testing:
      # np.title 将矩阵横向复制self._num_classes次
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      # 对框进行修正
      self._predictions["bbox_pred"] *= stds
      self._predictions["bbox_pred"] += means
    else:
      self._add_losses()
      layers_to_output.update(self._losses)

      val_summaries = []
      with tf.device("/cpu:0"):
        # 'GROUND_TRUTH'目录下添加标注后的图片
        val_summaries.append(self._add_gt_image_summary())
        # _event_summaries 包含 self._losses
        for key, var in self._event_summaries.items():
          # 添加标量
          val_summaries.append(tf.summary.scalar(key, var))
        # self._score_summaries 包含self._anchor_targets、self._proposal_targets、self._predictions
        for key, var in self._score_summaries.items():
          # 'SCORE/'目录下添加直方图
          self._add_score_summary(key, var)
        # self._act_summaries rpn_conv/3x3的输出特征图
        for var in self._act_summaries:
          # 'ACT/'目录下添加直方图
          self._add_act_summary(var)
        # self._train_summaries 包含所有可训练变量
        for var in self._train_summaries:
          # 'TRAIN/'目录下添加直方图
          self._add_train_summary(var)

      self._summary_op = tf.summary.merge_all()
      self._summary_op_val = tf.summary.merge(val_summaries)

    layers_to_output.update(self._predictions)

    # 模型输出，包含'rois'，self._losses，self._predictions
    return layers_to_output

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["head"], feed_dict=feed_dict)
    return feat

  # only useful during testing mode
  # 输入是仅包含单张图片的blob，用于测试网络
  def test_image(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}

    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois

  # 计算summary 训练过程中进行验证时使用
  def get_summary(self, sess, blobs):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

    return summary

  # 不包含summary的训练op 正常训练时使用
  def train_step(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        self._losses['total_loss'],
                                                                        train_op],
                                                                       feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

  # 计算summary的训练op 满足保存summary的间隔，需要保存summary时使用
  def train_step_with_summary(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                 self._losses['rpn_loss_box'],
                                                                                 self._losses['cross_entropy'],
                                                                                 self._losses['loss_box'],
                                                                                 self._losses['total_loss'],
                                                                                 self._summary_op,
                                                                                 train_op],
                                                                                feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

  # 无返回值的训练op
  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    sess.run([train_op], feed_dict=feed_dict)

