import tensorflow as tf
from networks.network import Network


#define

n_classes = 21
# total stride 各层stride相乘
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        # gt_boxes
        # return {'boxes': boxes,
        #         'gt_classes': gt_classes,
        #         'gt_overlaps': overlaps,
        #         'flipped': False,
        #         'seg_areas': seg_areas}
        # 所以shape=[None, 5]
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        # 在论文中，为了节省内存，对VGG网络从conv3_1开始训练
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))   # 感受野为196*196
        #========= RPN ============

        # rpn_conv/3x3为3*3大小的滑动窗口层
        # rpn_cls_score的shape为[1,height,width,depth(18)]
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='rpn_conv/3x3') # len(anchor_scales)*3*2   anchor尺度数量*anchor的长宽比例*预测的分类数量
             .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))

        # 获取anchors的cls和reg信息用于训练，即生成样本
        # 用于RPN网络微调训练
        (self.feed('rpn_cls_score','gt_boxes','im_info','data')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3') # len(anchor_scales)*3*2   anchor尺度数量*anchor的长宽比例*anchor的四个顶点
             .conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============

        # 以下两个feed过程是为了实现同一个anchor的两个分类的softmax过程

        # shape为[1,height*9,width,2]
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        # shape为[1,height,width,18]
        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))

        # [post_nms_topN, 5],建议框在输入图片上的[0,x1，y1，x2，y2],0是batch index，这里都是为0
        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))

        # 生成RoI样本信息，用于Fast R-CNN网络的微调训练
        # 不同的cls有不同的回归信息
        (self.feed('rpn_rois','gt_boxes')
             .proposal_target_layer(n_classes,name = 'roi-data'))


        #========= RCNN ============
        (self.feed('conv5_3', 'roi-data')
             .roi_pool(7, 7, 1.0/16, name='pool_5')
             .fc(4096, name='fc6')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc7')
             .dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('drop7')
             .fc(n_classes*4, relu=False, name='bbox_pred'))

