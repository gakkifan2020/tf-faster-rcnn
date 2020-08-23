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
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg

# pascal_voc继承imdb
class pascal_voc(imdb):
  # image_set代表数据集名称（train，val，test...）,year代表版本如2007
  def __init__(self, image_set, year, use_diff=False):
    name = 'voc_' + year + '_' + image_set      # 例如voc_2007_train
    if use_diff:
      name += '_diff'
    # 调用imdb的构造函数
    imdb.__init__(self, name)
    self._year = year
    self._image_set = image_set
    # /home/liubo/tf-faster-rcnn1/data/VOCdevkit2007/
    self._devkit_path = self._get_default_path()
    # /home/liubo/tf-faster-rcnn1/data/VOCdevkit/VOC2007/
    self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
    self._classes = ('__background__',  # always index 0
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')
    # _class_to_ind保存的是{'__background__' ： 0 , 'aeroplane' ： 1 , 'bicycle' : 2 , 等}
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    # 图片的格式
    self._image_ext = '.jpg'
    # 里面包含所有数据集图像的名称，如[000001,000002,000003...]
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    # 得到roi图片信息,重载imdb中
    self._roidb_handler = self.gt_roidb
    # 定义一个随机的标识符
    self._salt = str(uuid.uuid4())      # 4a5ae4d0-03ff-490c-871b-33b615a19bb4
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': use_diff,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  # 返回图片所在的路径
  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    # /home/liubo/tf-faster-rcnn1/data/VOCdevkit/VOC2007/JPEGImages/000001.JPG
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # _data_path = /home/liubo/tf-faster-rcnn1/data/VOCdevkit/VOC2007/
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)      # _image_ext = .JPG 后缀名
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  # 获取图片索引
  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]      # 删除每一行最后的'\n' 或 '\t'
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    '''
    __C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
    __C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
    return Fsater-RCNN_TF/data/VOCdevkit+self._year
    '''
    return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

  # 返回的是gt_roidb，即groundtrue region of interest
  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    # name为数据集的名称
    # 例如 cache_file = Fsater-RCNN_TF/data/cache/voc_2007_train__gt_roidb.pkl
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        # pickle.load(file)是将.pkl文件解析
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      '''
      roidb的部分内容
      'seg_areas': array([437869., 402758.], dtype=float32), 'flipped': False}, {'boxes': array([[3141, 2479, 3603, 2834],
       [4034, 2506, 4493, 2912]], dtype=uint16), 'gt_classes': array([3, 3], dtype=int32), 'gt_overlaps': <2x4 sparse matrix of type '<type 'numpy.float32'>'
      '''
      return roidb

    gt_roidb = [self._load_pascal_annotation(index)
                for index in self.image_index]
    # 将gt_roidb存入临时文件cache_file并返回gt_roidb
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  # 将rpn方法计算得到的roidb与gt riodb整合为新的roidb
  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  # 读取rpn计算得到的roidb
  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
      # create_roidb_from_box_list为imdb.py的函数
      # 从box_list中创建roi的方法是为了创建一系列新的roi
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  # 处理xml文件
  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    # 下面是其中一个xml文件
    # '''
    # <annotation>
    #     <folder>所有均压环图片</folder>
    #     <filename>300002.JPG</filename>
    #     <path>C:\Users\Ilearn\Desktop\所有均压环图片\300002.JPG</path>
    #     <source>
    #         <database>Unknown</database>
    #     </source>
    #     <size>
    #         <width>6000</width>
    #         <height>3376</height>
    #         <depth>3</depth>
    #     </size>
    #     <segmented>0</segmented>
    #     <object>
    #         <name>tilt</name>
    #         <pose>Unspecified</pose>
    #         <truncated>0</truncated>
    #         <difficult>0</difficult>
    #         <bndbox>
    #             <xmin>2554</xmin>
    #             <ymin>552</ymin>
    #             <xmax>3914</xmax>
    #             <ymax>964</ymax>
    #         </bndbox>
    #     </object>
    # </annotation>
    # '''
    # / data / VOCdevkit +'year'/'VOC' + self._year/Annotations/000001.xml
    filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    # 从中去掉标注为difficult的难训练对象
    if not self.config['use_diff']:
      # Exclude the samples labeled as difficult
      non_diff_objs = [
        obj for obj in objs if int(obj.find('difficult').text) == 0]
      # if len(non_diff_objs) != len(objs):
      #     print 'Removed {} difficult objects'.format(
      #         len(objs) - len(non_diff_objs))
      # 只保留difficult = 0 的object
      objs = non_diff_objs

    num_objs = len(objs)
    # 初始化boxes，建立一个（num_objs，4）的全0矩阵，num_objs对应一张图片中有几个object，4对应着xmin,xmax,ymin,ymax
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    # 有几个object对应着几个gt_classes
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    # self.num_classes对应着之前定义所有分类的个数,第i个数据在自己的类别上得分为1
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    # 对应每个box的面积
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    # ix对应的是索引下标，obj是每个框，里面包含了name,xmin,ymin,xmax,ymax
    '''
    <object>
        <name>tilt</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>2554</xmin>
            <ymin>552</ymin>
            <xmax>3914</xmax>
            <ymax>964</ymax>
        </bndbox>
    </object>
    '''
    for ix, obj in enumerate(objs):
      bbox = obj.find('bndbox')
      # Make pixel indexes 0-based
      x1 = float(bbox.find('xmin').text) - 1
      y1 = float(bbox.find('ymin').text) - 1
      x2 = float(bbox.find('xmax').text) - 1
      y2 = float(bbox.find('ymax').text) - 1
      # self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
      # _class_to_ind保存的是{'__background__' ： 0 , 'tilt' ： 1 , 'miss' : 2 , 'normal' : 3}
      # 将得到的name转换为下标索引
      cls = self._class_to_ind[obj.find('name').text.lower().strip()]
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      # 类似于one-hot编码[[0,1，0,0,][0,0,0,1]]
      overlaps[ix, cls] = 1.0
      # box的面积
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    '''
    a = np.array([[0, 1], [1, 0]])
    print(scipy.sparse.csr_matrix(a))
    (0, 0)	0
    (0, 1)	1
    (1, 0)	1
    (1, 1)	0
    '''
    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,                # array
            'gt_classes': gt_classes,     # array
            'gt_overlaps': overlaps,      # scipy.sparse.csr.csr_matrix
            'flipped': False,             # bool
            'seg_areas': seg_areas}       # array

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      'VOC' + self._year,
      'Main',
      filename)
    return path

  # 参数是所有检测结果的Bbox
  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          # dets是所有label的集合
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            #  按照这个顺序写入到路径中：图片索引 检测结果 Xmin Ymin Xmax Ymax
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  # 用python计算AP和mAP
  def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(
      self._devkit_path,
      'VOC' + self._year,
      'Annotations',
      '{:s}.xml')
    imagesetfile = os.path.join(
      self._devkit_path,
      'VOC' + self._year,
      'ImageSets',
      'Main',
      self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      rec, prec, ap = voc_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
        use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
      aps += [ap]
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  # 用matlab计算AP和mAP
  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  # 设置参数
  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.pascal_voc import pascal_voc

  d = pascal_voc('trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()
