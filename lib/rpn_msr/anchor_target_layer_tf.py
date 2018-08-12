# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from rpn_msr.generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
import pdb

DEBUG = False

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, data, _feat_stride = [16,], anchor_scales = [4 ,8, 16, 32]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    # 生成基本anchor：（相当于特征图最左下角的滑动窗口生成的九个anchor在输入图片上的对应坐标位置）
    # array([[ -83.,  -39.,  100.,   56.],
    #       [-175.,  -87.,  192.,  104.],
    #       [-359., -183.,  376.,  200.],
    #       [ -55.,  -55.,   72.,   72.],
    #       [-119., -119.,  136.,  136.],
    #       [-247., -247.,  264.,  264.],
    #       [ -35.,  -79.,   52.,   96.],
    #       [ -79., -167.,   96.,  184.],
    #       [-167., -343.,  184.,  360.]])
    _anchors = generate_anchors(scales=np.array(anchor_scales))

    #9
    _num_anchors = _anchors.shape[0]

    if DEBUG:
        print 'anchors:'
        print _anchors
        print 'anchor shapes:'
        # [[183.  95.]
        #  [367. 191.]
        #  [735. 383.]
        #  [127. 127.]
        #  [255. 255.]
        #  [511. 511.]
        #  [87.  175.]
        #  [175. 351.]
        #  [351. 703.]]
        print np.hstack((
            # array([[183.],
            #        [367.],
            #        [735.],
            #        [127.],
            #        [255.],
            #        [511.],
            #        [87.],
            #        [175.],
            #        [351.]])
            _anchors[:, 2::4] - _anchors[:, 0::4],
            # array([[95.],
            #        [191.],
            #        [383.],
            #        [127.],
            #        [255.],
            #        [511.],
            #        [175.],
            #        [351.],
            #        [703.]])
            _anchors[:, 3::4] - _anchors[:, 1::4],
        ))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # 允许boxes超出图片边界的范围，0表示不能超出图片边界
    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]

    im_info = im_info[0]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    # rpn_cls_score.shape=[1,height,width,depth]
    # TODO: 1代表一张图片？
    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # rpn_cls_score.shape的第二位第三位分别存储高与宽
    # rpn_cls_score.shape=[1,height,width,depth],按前提来看，depth应为18

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    if DEBUG:
        print 'AnchorTargetLayer: height', height, 'width', width
        print ''
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])
        print 'height, width: ({}, {})'.format(height, width)
        print 'rpn: gt_boxes.shape', gt_boxes.shape
        print 'rpn: gt_boxes', gt_boxes

    # 1. Generate proposals from bbox deltas and shifted anchors

    # 该层对输入层的total stride为16，相当于在该层滑动1，在输入层滑动16个像素。
    # shift包含着x或y方向上不同位置的每一个窗口所对应的anchors的偏移量
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    # 将坐标向量转换为坐标矩阵，新的shift_x行向量为旧shift_x，有dim（shift_y）行，每一行是相同的，新的shift_y列向量为旧shift_y，有dim（shift_x）列，每一列是相同的
    # 最后生成的shift_x, shift_y的形状都是width*height，即可以包含rpn_cls_score所有点的x和y方向上的偏移量
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # numpy.ravel()多维数组降为一维，组合得到一个（width*height，4）的数组。ravel列举数据内的元素。
    # 因为box坐标的表示为（Xmin，Ymin，Xmax，Ymax），当需要加上坐标偏移时，加上的偏移量的形式就应该是（shift_x, shift_y，shift_x, shift_y）
    # vstack垂直堆叠后的形式是列向量的（shift_x, shift_y，shift_x, shift_y），因此需要转置transpose
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors

    # 9
    A = _num_anchors
    # weight*height
    K = shifts.shape[0]

    # (1, A, 4)与(K, 1, 4)的数组进行相加，得到(K, A, 4)数组，实验得证，每个(K, 1, 4)的4元素都依次与(1, A, 4)中的每一个4元素相加（numpy里array相加会自动拓展，按需通过重复某子元素拓展成（K，A，4）），
    # 最后得到(K, A, 4)数组，这样是合理的，因为_anchors中记录的是对用于左上角可视野的9个anchor的左上角坐标与右下角坐标的4个值，
    # 而shifts中记录width*height个可视野相对于左上角可视野的偏移量，两者相加可得到width*height*9个预测anchor的左上角与右下角坐标信息
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    # 获取不超出边界的anchors的index
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]

    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)

    # keep only inside anchors
    # 保存不超出边界的anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors.shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    # 返回[N,K]矩阵，记录每一个anchors和gt框的IoU。N为anchors数量，K为gt框数量。
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    # 记录每个anchor对应的最大IoU的index（0到K-1），为每一个anchor找到与其重叠最好的GT
    argmax_overlaps = overlaps.argmax(axis=1)
    # 记录每个anchor对应的最大IoU的值
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    # 记录和gt框IoU最高的anchors的index，为每一个GT找到与其重叠最好的一个anchor。当有多个anchor与GT框的IoU同时取到最大值时，只返回第一个的anchor的index
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    # 记录和每一个gt框有最大IoU值的anchors的IoU
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    # np.where返回的是一个tuple，tuple里元素为array。在这个情况中，tuple里存着两个array，第一个array指明axis=0时的index，第二个array指明axis=1时的index。故用[0]取array，表明anchor的index
    # 上面所求的gt_argmax_overlaps只能指定第一个有最大IoU的anchor位置，当有多个anchor的IoU同时取到最大值时，是不能同时取到这几个anchors的。
    # 通过以下这个方法，获取到全部这些anchors。
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]


    # PRN网络微调训练
    # 正样本：与Ground Truth相交IoU最大的anchors【以防后一种方式下没有正样本】+与Ground Truth相交IoU>0.7的anchors
    # 负样本：与Ground Truth相交IoU<0.3的anchors
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        # labels与max_overlaps都是[(len(inds_inside), 1]形状的，下一行代码表示将最大IoU小于TRAIN.RPN_NEGATIVE_OVERLAP的对应的anchors的labels设置为0
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    # 为了防止出现空的正样本，将与Ground Truth相交IoU最大的anchors设置为正样本
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    # 与Ground Truth相交IoU>TRAIN.RPN_POSITIVE_OVERLAP的anchors
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    # 这个参数就是看positive与negative谁比较强，先设置0说明positive强，因为0可能转1,而后设置0说明negative强，设置完1还可以设置成0
    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        #print "was %s inds, disabling %s, now %s inds" % (
            #len(bg_inds), len(disable_inds), np.sum(labels == 0))

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means:'
        print means
        print 'stdevs:'
        print stds

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # labels
    #pdb.set_trace()
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #assert bbox_inside_weights.shape[2] == height
    #assert bbox_inside_weights.shape[3] == width

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #assert bbox_outside_weights.shape[2] == height
    #assert bbox_outside_weights.shape[3] == width

    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights



def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
