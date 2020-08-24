# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
try:
  import cPickle as pickle
except ImportError:
  import pickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

# Slover的封装类，包含和训练有关的属性和方法
class SolverWrapper(object):
  """
    A wrapper class for the training process
  """

  def __init__(self, sess, network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
    self.net = network      # network类的实例
    self.imdb = imdb        # imdb类的实例
    self.roidb = roidb      # roidb字典
    self.valroidb = valroidb     # 验证roidb字典
    self.output_dir = output_dir  # 模型保存路径
    self.tbdir = tbdir            # tensorboard保存路径
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = tbdir + '_val'      # 验证过程的tensorboard保存路径
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model      # 预训练权重的路径

  # 保存快照，包括模型权重的ckpt文件和训练参数的pkl文件
  def snapshot(self, sess, iter):
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    # 保存模型权重 例：shufflenetv2_faster_rcnn_iter_10000.cpkt等三个文件
    # SNAPSHOT_PREFIX在yml文件中配置 例：'shufflenetv2_faster_rcnn'
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
    filename = os.path.join(self.output_dir, filename)
    self.saver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    # 保存随机数状态等训练过程参数
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    # 保存随机数种子
    st0 = np.random.get_state()
    # current position in the database
    # 保存当前图片序号
    cur = self.data_layer._cur
    # current shuffled indexes of the database
    # 保存打乱后的图片序号列表
    perm = self.data_layer._perm
    # current position in the validation database
    # 验证过程，同上
    cur_val = self.data_layer_val._cur
    # current shuffled indexes of the validation database
    perm_val = self.data_layer_val._perm

    # Dump the meta info
    # 写入 例：shufflenetv2_faster_rcnn_iter_10000.pkl
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename

  # 载入快照
  def from_snapshot(self, sess, sfile, nfile):
    print('Restoring model snapshots from {:s}'.format(sfile))
    self.saver.restore(sess, sfile)
    print('Restored.')
    # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
    # tried my best to find the random states so that it can be recovered exactly
    # However the Tensorflow state is currently not available
    with open(nfile, 'rb') as fid:
      st0 = pickle.load(fid)
      cur = pickle.load(fid)
      perm = pickle.load(fid)
      cur_val = pickle.load(fid)
      perm_val = pickle.load(fid)
      last_snapshot_iter = pickle.load(fid)

      np.random.set_state(st0)
      self.data_layer._cur = cur
      self.data_layer._perm = perm
      self.data_layer_val._cur = cur_val
      self.data_layer_val._perm = perm_val

    return last_snapshot_iter

  # 返回checkpoint文件中得到的参数词典
  def get_variables_in_checkpoint_file(self, file_name):
    try:
      reader = pywrap_tensorflow.NewCheckpointReader(file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      return var_to_shape_map 
    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")

  # 构建模型
  def construct_graph(self, sess):
    with sess.graph.as_default():
      # Set the random seed for tensorflow
      # 为了方便复现，随机数种子在cfg中设置，并在保存模型时保存在pkl文件中
      tf.set_random_seed(cfg.RNG_SEED)
      # Build the main computation graph
      # layers为模型输出，包含roi区域，loss值，预测结果，形式为字典
      layers = self.net.create_architecture('TRAIN', self.imdb.num_classes, tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
      # Define the loss
      loss = layers['total_loss']
      # Set learning rate and momentum
      # 设置优化器，设定学习速率和动量
      lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
      self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

      # 计算梯度，返回（gradient，variable）列表
      # Compute the gradients with regard to the loss
      gvs = self.optimizer.compute_gradients(loss)
      # Double the gradient of the bias if set
      # 在yml文件中重新设置为false，如果为true就将biases的梯度翻倍
      if cfg.TRAIN.DOUBLE_BIAS:
        final_gvs = []
        with tf.variable_scope('Gradient_Mult') as scope:
          for grad, var in gvs:
            scale = 1.
            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
              scale *= 2.
            if not np.allclose(scale, 1.0):
              grad = tf.multiply(grad, scale)
            final_gvs.append((grad, var))
            # 更新梯度
        train_op = self.optimizer.apply_gradients(final_gvs)
      else:
        # 更新梯度
        train_op = self.optimizer.apply_gradients(gvs)

      # 创建Saver类，默认保存所有检查点
      # We will handle the snapshots ourselves
      self.saver = tf.train.Saver(max_to_keep=100000)
      # Write the train and validation information to tensorboard
      # 创建训练和验证过程tensorboard的filewrite类
      self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
      self.valwriter = tf.summary.FileWriter(self.tbvaldir)

    return lr, train_op

  def find_previous(self):
    # 设置匹配字符串
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
    # 返回当前目录下符合格式的文件列表
    sfiles = glob.glob(sfiles)
    # 按照文件最后修改时间排序
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in TensorFlow
    # 不是很懂这部分的目的
    # '''lb:应该用于查看是否之前保存过模型，如果有的restore,没有就initialize'''
    redfiles = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      redfiles.append(os.path.join(self.output_dir, 
                      cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize+1)))
    sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
    nfiles = [nn for nn in nfiles if nn not in redfiles]

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    return lsf, nfiles, sfiles

  # 模型初始化
  def initialize(self, sess):
    # Initial file lists are empty
    # 作用是初始化变量
    np_paths = []
    ss_paths = []
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    # 获取全部变量
    variables = tf.global_variables()
    # Initialize all variables first
    # 初始化全部变量
    sess.run(tf.variables_initializer(variables, name='init'))
    # 从预训练的checkpoint中获得变量和对应值的字典
    var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
    # Get the variables to restore, ignoring the variables to fix
    # 在resnet等子类中实现，获得需要重载的参数字典
    variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

    # 从预训练权重中重载variables_to_restore中的参数
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, self.pretrained_model)
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    # 部分参数需要转化后再重载，在每个子网络中实现
    self.net.fix_variables(sess, self.pretrained_model)
    print('Fixed.')
    last_snapshot_iter = 0
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = list(cfg.TRAIN.STEPSIZE)

    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

  # 获取最新的快照(用于重载)
  def restore(self, sess, sfile, nfile):
    # Get the most recent snapshot and restore
    np_paths = [nfile]
    ss_paths = [sfile]
    # Restore model from snapshots
    # 从pkl中回复变量
    last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
    # Set the learning rate
    # 初始学习速率
    rate = cfg.TRAIN.LEARNING_RATE
    stepsizes = []

    # 目前cfg.TRAIN.STEPSIZE仅有一项，为[30000]，迭代超过30000次后，学习速率乘0.1
    # 在train_faster_rcnn.sh中根据数据集重新设置，voc_2007_trainval默认为[50000]
    for stepsize in cfg.TRAIN.STEPSIZE:
      if last_snapshot_iter > stepsize:
        rate *= cfg.TRAIN.GAMMA
      else:
        stepsizes.append(stepsize)

    return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

  # 删除多余的模型快照，默认保存3个
  def remove_snapshot(self, np_paths, ss_paths):
    to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      nfile = np_paths[0]
      os.remove(str(nfile))
      np_paths.remove(nfile)

    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      sfile = ss_paths[0]
      # To make the code compatible to earlier versions of Tensorflow,
      # where the naming tradition for checkpoints are different
      if os.path.exists(str(sfile)):
        os.remove(str(sfile))
      else:
        os.remove(str(sfile + '.data-00000-of-00001'))
        os.remove(str(sfile + '.index'))
      sfile_meta = sfile + '.meta'
      os.remove(str(sfile_meta))
      ss_paths.remove(sfile)

  # 训练模型
  def train_model(self, sess, max_iters):     # max_iters在train_faster_rcnn.sh指定
    # Build data layers for both training and validation set
    # 创建RoIDataLayer类
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

    # Construct the computation graph
    # 构建计算图
    lr, train_op = self.construct_graph(sess)

    # Find previous snapshots if there is any to restore from
    # 返回快照文件 lsf为快照文件个数
    lsf, nfiles, sfiles = self.find_previous()

    # Initialize the variables or restore them from the last snapshot
    # 如果没有快照文件就从头初始化，有则直接重载
    if lsf == 0:
      rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
    else:
      rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess, 
                                                                            str(sfiles[-1]), 
                                                                            str(nfiles[-1]))
    timer = Timer()
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()

    # Make sure the lists are not empty
    # max_iter被添加到列表末尾，又反向排列成了第一个，然后pop从末尾开始提取元素
    # 在initialize中：stepsizes = list(cfg.TRAIN.STEPSIZE)
    # 在restore中：如果cfg.TRAIN.STEPSIZE中的值大于last_snapshot_iter的保存下来
    stepsizes.append(max_iters)
    stepsizes.reverse()
    next_stepsize = stepsizes.pop()
    while iter < max_iters + 1:
      # Learning rate
      # 每满足下一个stepsize，就保存一次快照，并将学习速率降低
      if iter == next_stepsize + 1:
        # Add snapshot here before reducing the learning rate
        self.snapshot(sess, iter)
        rate *= cfg.TRAIN.GAMMA
        sess.run(tf.assign(lr, rate))
        next_stepsize = stepsizes.pop()

      # 开始计时
      timer.tic()
      # Get training data, one batch at a time
      # 提取一个batch的blobs数据
      # blobs包含三组键值对{data、gt_boxes、im_info},具体内容在roi_data_layer/minibatch.py中
      blobs = self.data_layer.forward()

      now = time.time()
      # cfg.TRAIN.SUMMARY_INTERVAL=180 每3分钟保存一次摘要
      if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
        # Compute the graph with summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
          self.net.train_step_with_summary(sess, blobs, train_op)
        # 保存摘要
        self.writer.add_summary(summary, float(iter))
        # Also check the summary on the validation set
        # 从验证集提取一个batch进行验证，并计算保存摘要
        blobs_val = self.data_layer_val.forward()
        summary_val = self.net.get_summary(sess, blobs_val)
        self.valwriter.add_summary(summary_val, float(iter))
        last_summary_time = now
      else:
        # Compute the graph without summary
        # 只训练，不计算保存摘要
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
          self.net.train_step(sess, blobs, train_op)
        # 结束计时
      timer.toc()

      # Display training information
      if iter % (cfg.TRAIN.DISPLAY) == 0:
        print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
              '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
              (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))
        print('speed: {:.3f}s / iter'.format(timer.average_time))

      # Snapshotting
      # 每间隔SNAPSHOT_ITERS次保存一次快照
      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        ss_path, np_path = self.snapshot(sess, iter)
        np_paths.append(np_path)
        ss_paths.append(ss_path)

        # Remove the old snapshots if there are too many
        # 删除多余快照
        if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          self.remove_snapshot(np_paths, ss_paths)

      iter += 1
    # 保存训练结束时的快照
    if last_snapshot_iter != iter - 1:
      self.snapshot(sess, iter - 1)

    self.writer.close()
    self.valwriter.close()


def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    # 数据增强，将图像水平翻转后添加到数据集末尾
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  # 为roidata添加一些说明性的附加属性
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  """
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
  """
  return imdb.roidb

# 删除无用的rois
def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  # 筛选有效图片
  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    # 获得前背景的下标
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    # 一张有效的图片至少有一个前景或背景
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid
  # 返回筛选后的roidb ，并打印筛掉的图片的数量
  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
  """Train a Faster R-CNN network."""
  # 删除无效图片，保证每张图都至少有一个前景和一个背景
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)

  # session运行参数，自动分配GPU,根据需要动态申请显存
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  with tf.Session(config=tfconfig) as sess:
    # 创建SolverWrapper类实例
    sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir, tb_dir,
                       pretrained_model=pretrained_model)
    print('Solving...')
    # 调用train_model方法进行训练
    sw.train_model(sess, max_iters)
    print('done solving')
