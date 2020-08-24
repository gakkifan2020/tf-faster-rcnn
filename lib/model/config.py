from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
# 创建一个easydict的字典，可以以属性的方式访问字典，方便使用
__C.TRAIN = edict()

# Initial learning rate
# 相当于添加字典键值对 {LEARNING_RATE：0.001} 下同
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0001

# Factor for reducing the learning rate
# 每次学习速率降低为原来的十分之一
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
# 每迭代stepsize次，学习速率降低一次
__C.TRAIN.STEPSIZE = [30000]

# Iteration intervals for showing the loss during training, on command line interface
# 每10个iter展示一次训练情况，包括loss，lr等
__C.TRAIN.DISPLAY = 10

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution 
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Whether to add ground truth boxes to the pool when sampling regions
# 是否把gt_boxes也加入到rpn产生的候选区域中，用于训练
__C.TRAIN.USE_GT = False

# Whether to use aspect-ratio grouping of training images, introduced merely for saving
# GPU memory
# 是否使用长宽比对训练图像进行分组
__C.TRAIN.ASPECT_GROUPING = False

# The number of snapshots kept, older ones are deleted to save space
# 模型快照pkl保存的最大数量
__C.TRAIN.SNAPSHOT_KEPT = 3

# The time interval for saving tensorflow summaries
# 保存摘要的时间间隔，单位秒
__C.TRAIN.SUMMARY_INTERVAL = 180

# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
# 将最短边缩放到600
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
# 如果缩放后最长边超过1000，则再次缩放，将最长边缩放到1000
__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
# 训练RPN网络时的batch_size
__C.TRAIN.IMS_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
# 每张图片经过RPN产生的候选区域中用于训练RCNN的数量（训练RCNN部分的batch_size）
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
# 一张图片经过RPN产生的候选区域中用于训练RCNN的前景的最大占比
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
# 大于阈值的rpn候选区域判断为前景
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
# 在[LO, HI)内的RPN候选区域判断为背景
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
# 训练bbox回归量
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
# 每5000个iter保存一次快照
__C.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True

# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
# 使用预计算数据标准化
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True

__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)

__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.

# Use RPN to detect objects
# 未使用
__C.TRAIN.HAS_RPN = True

# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# IOU < thresh: negative example
# 与任意的ground-truth的IOU<0.3为负样本
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# If an anchor satisfied by positive and negative conditions set to negative
# 如果同时满足正负条件设置为负样本
__C.TRAIN.RPN_CLOBBER_POSITIVES = False

# Max number of foreground examples
# RPN层对anchor打标签时保留有效标签中前景的最大比例  cfg.TRAIN.RPN_BATCHSIZE
__C.TRAIN.RPN_FG_FRACTION = 0.5

# Total number of examples
# RPN层对anchor打标签时保留有效标签的数量
__C.TRAIN.RPN_BATCHSIZE = 256

# NMS threshold used on RPN proposals
# RPN层NMS阈值
__C.TRAIN.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
# 在NMS处理之前的top proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
# 在NMS处理之后的top proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
# 设置正负样本的权重，如果为-1，则正负样本的权重值相同，均为1/样本数目
# 设置为0<p<1的值，则正样本权重为p/正样本数目，负样本权重为（1-p）/负样本数目
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# Whether to use all ground truth bounding boxes for training, 
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
__C.TRAIN.USE_ALL_GT = True

#
# Testing options
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
# 测试时图片的缩放尺寸
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
# 默认gt_roidb,只使用标注框，还可以选择用rpn_roidb,在标注框的基础上又加上了rpn得到的预测框
__C.TEST.PROPOSAL_METHOD = 'gt'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
# __C.TEST.RPN_MIN_SIZE = 16

# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
# 默认采用nms筛选rpn输出的proposal，但是top模式效果更好，虽然速度更慢
__C.TEST.MODE = 'nms'

# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
__C.TEST.RPN_TOP_N = 5000

#
# ResNet options
#

__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize. 
# if true, the region will be resized to a square of 2xPOOLING_SIZE, 
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1

#
# MobileNet options
#

__C.MOBILENET = edict()

# Whether to regularize the depth-wise filters during training
# 对DW层不进行正则化
__C.MOBILENET.REGU_DEPTH = False

# Number of fixed layers during training, by default the bottom 5 of 14 layers is fixed
# Range: 0 (none) to 12 (all)
# 固定权重的层数
__C.MOBILENET.FIXED_LAYERS = 5

# Weight decay for the mobilenet weights
# 论文中建议DW层的WEIGHT_DECAY尽量小或不加，因为轻量模型比起过拟合，更容易欠拟合
__C.MOBILENET.WEIGHT_DECAY = 0.00004

# Depth multiplier
# 深度因子，必须>0
__C.MOBILENET.DEPTH_MULTIPLIER = 1.

#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# 去均值
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
# 随机数种子
__C.RNG_SEED = 3

# Root directory of project
# 工程根目录的绝对路径 /home/xxx/tf-faster-rcnn
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
# /home/xxx/tf-faster-rcnn/data
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
# 用于指定模型存放路径，与tag参数对应
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
# 默认使用gpu进行NMS
__C.USE_GPU_NMS = True

# Use an end-to-end tensorflow model.
# Note: models in E2E tensorflow mode have only been tested in feed-forward mode,
#       but these models are exportable to other tensorflow instances as GraphDef files.
# 使用tensorflow端到端实现，不用numpy
__C.USE_E2E_TF = True

# Default pooling mode, only 'crop' is available
# 只能用'crop'
__C.POOLING_MODE = 'crop'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8,16,32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5,1,2]

# Number of filters for the RPN layer
__C.RPN_CHANNELS = 512


def get_output_dir(imdb, weights_filename):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  # 指定模型保存路径，如果没有则创建
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def get_output_tb_dir(imdb, weights_filename):
  """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  # 指定tensorboard保存路径，如果没有则创建
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  # 必须为edict对象
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    # a中的参数必须是b中有的
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    # 对应参数的类型必须相同
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        # 如果参数同样是edict字典，则递归调用
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      # 覆盖b中的值
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  # 从yaml文件中读取参数，并覆盖默认参数
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  # k是键，v是值
  # 用list中的参数覆盖默认参数
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
