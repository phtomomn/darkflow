# coding=utf-8

import math
import sys

import numpy as np
import tensorflow as tf

from graph_methods import (caculate_dis_matrix, caculate_subclass_distance,
                           class_use_graph, pca_via_svd, caculate_smoothness, caculate_final_result, class_feature_use_graph, class_target_result_with_source_centroid)


class AlexNetModel(object):

    def __init__(self, num_classes=1000, dropout_keep_prob=0.5):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.featurelen = 256

        self.dis_positive = tf.get_variable(name='source_dis', shape=[], initializer=tf.zeros_initializer(), trainable=False)
        self.dis_negative = tf.get_variable(name='target_dis', shape=[], initializer=tf.zeros_initializer(), trainable=False)

        self.smoothness = tf.get_variable(name='smooth', shape=[], initializer=tf.zeros_initializer(), trainable=False)

        self.source_moving_centroid = tf.get_variable(name='source_moving_centroid', shape=[
                                                      num_classes, self.featurelen], initializer=tf.zeros_initializer(), trainable=False)
        self.target_moving_centroid = tf.get_variable(name='target_moving_centroid', shape=[
                                                      num_classes, self.featurelen], initializer=tf.zeros_initializer(), trainable=False)

    def inference(self, x, training=False):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 1, 1e-5, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 1, 1e-5, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        conv4_flattened = tf.contrib.layers.flatten(conv4)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        self.flattened = flattened
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        if training:
            fc6 = dropout(fc6, self.dropout_keep_prob)
            self.fc6 = fc6

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        if training:
            fc7 = dropout(fc7, self.dropout_keep_prob)
            self.fc7 = fc7

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        fc8 = fc(fc7, 4096, 256, relu=False, name='fc8')
        self.fc8 = fc8
        self.score = fc(fc8, 256, self.num_classes,
                        relu=False, stddev=0.005, name='fc9')

        self.feature = self.fc8
        #self.result_graph = class_feature_use_graph(feature=self.feature, pca_dimension=64, k_neighbor=20)
        self.result = tf.argmax(self.score, 1)
        self.output = tf.nn.softmax(self.score)
    
        return self.score

    def adoptimize(self, learning_rate, train_layers=[]):
        var_list = [v for v in tf.trainable_variables() if 'D' in v.name]
        D_weights = [v for v in var_list if 'weights' in v.name]
        D_biases = [v for v in var_list if 'biases' in v.name]

        self.Dregloss = 0.0005 * \
            tf.reduce_mean([tf.nn.l2_loss(v)
                            for v in var_list if 'weights' in v.name])

        D_op1 = tf.train.MomentumOptimizer(
            learning_rate, 0.9).minimize(self.D_loss+self.Dregloss, var_list=D_weights)
        D_op2 = tf.train.MomentumOptimizer(
            learning_rate*2.0, 0.9).minimize(self.D_loss+self.Dregloss, var_list=D_biases)
        D_op = tf.group(D_op1, D_op2)
        return D_op

    def adloss(self, x, xt, y, global_step, xtl, ytl):
        with tf.variable_scope('reuse_inference') as scope:
            scope.reuse_variables()
            self.inference(x, training=True)
            feature_source = self.feature
            feature_number = tf.shape(feature_source)[0]

            scope.reuse_variables()
            self.inference(xt, training=True)
            feature_target = self.feature
            #predict_result_target_graph = self.result_graph
            output_target = self.output

            #=====Add at 2019.03.20=====
            scope.reuse_variables()
            self.inference(xtl, training=True)
            feature_target_with_label = self.feature
            #=====end=====
        
        self.feature_total = tf.concat([feature_source, feature_target, feature_target_with_label], 0)
        self.feature_total_low = pca_via_svd(self.feature_total, 16)
        self.feature_source_low = self.feature_total_low[0:feature_number]
        self.feature_target_low = self.feature_total_low[feature_number:2*feature_number]
        self.feature_target_with_label_low = self.feature_total_low[2*feature_number: 3*feature_number]
        predict_result_target_graph = class_feature_use_graph(self.feature_target_low, k_neighbor=5, pca=False)
        self.result_graph = predict_result_target_graph

        predict_result_source = tf.argmax(y, 1)                 #分类结果，范围[0, num_class]
        predict_result_target_label = tf.argmax(ytl, 1)
        predict_result_target = tf.argmax(output_target, 1)

        self.feature_target = feature_target
        self.feature_source = feature_source

        self.result_source = predict_result_source
        self.result_target_label = predict_result_target_label

        predict_result = tf.concat([predict_result_source, predict_result_target], axis=0)

        feature_source_positive1 = tf.gather_nd(feature_source, tf.where(tf.equal(predict_result_source, 1)))
        feature_target_positive1 = tf.gather_nd(feature_target, tf.where(tf.equal(predict_result_target, 1)))
 
        feature_source_negative = tf.gather_nd(feature_source, tf.where(tf.equal(predict_result_source, 0)))
        feature_target_negative = tf.gather_nd(feature_target, tf.where(tf.equal(predict_result_target, 0)))

        feature_positive1 = tf.concat([feature_source_positive1, feature_target_positive1], axis=0)


        # caculate centroid
        ones = tf.ones_like(feature_source)
        current_source_count = tf.unsorted_segment_sum(ones, predict_result_source, self.num_classes)

        ones = tf.ones_like(feature_target)
        current_target_count = tf.unsorted_segment_sum(ones, predict_result_target, self.num_classes)

        ones = tf.ones_like(feature_target_with_label)
        current_target_label_count = tf.unsorted_segment_sum(ones, predict_result_target_label, self.num_classes)

        current_positive_source_count = tf.maximum(current_source_count, tf.ones_like(current_source_count))    # 防止分母为零
        current_positive_target_count = tf.maximum(current_target_count, tf.ones_like(current_target_count))
        current_positive_target_label_count = tf.maximum(current_target_label_count, tf.ones_like(current_target_label_count))

        current_source_centroid = tf.unsorted_segment_sum(feature_source, predict_result_source, self.num_classes) / current_positive_source_count
        current_target_centroid = tf.unsorted_segment_sum(feature_target, predict_result_target, self.num_classes) / current_positive_target_count
        current_target_label_centroid = tf.unsorted_segment_sum(feature_target_with_label, predict_result_target_label, self.num_classes) / current_positive_target_label_count

        #self.test = tf.reduce_sum((feature_target - tf.gather(params=current_source_centroid, indices=predict_result_target))**2, 1)
        self.SemanticScore_source = 0.5*tf.exp(-0.01* tf.reduce_sum((feature_target - tf.gather(params=current_source_centroid, indices=predict_result_target))**2, 1))
        self.SemanticScore_target = 0.5*tf.exp(-0.01* tf.reduce_sum((feature_target - tf.gather(params=current_target_label_centroid, indices=predict_result_target))**2, 1))

        
        """NEW"""
        predict_result_target_final_use_graph = class_target_result_with_source_centroid(
            target_feature=feature_target,
            target_result=predict_result_target_graph,
            source_centroid=current_source_centroid
        )

        self.target_result_new = predict_result_target_final_use_graph

        self.target_result_graph_sub = predict_result_target_graph
        self.target_result_graph_total = self.target_result_new

        self.loss_new = tf.reduce_mean(tf.square(tf.cast(predict_result_target, tf.float32) - tf.cast(self.target_result_graph_total, tf.float32)))
        #图分类结果与FC分类结果的一致性损失


        smoothness = 0.2 * caculate_smoothness(feature_target, self.target_result_graph_sub) + 0.8 * caculate_smoothness(feature_target, self.target_result_graph_total)

        tf.summary.scalar('smoothness', smoothness)

        decay = tf.constant(0.3)
        smoothness = smoothness * (1-decay) + self.smoothness
        smoothness = tf.minimum(smoothness, 500.0)
        

        current_positive_centroid1 = tf.reduce_mean(feature_positive1)
        distance_pos_and_pos_centroid1 = tf.square(feature_positive1 - current_positive_centroid1) 
        self.disp = tf.reduce_mean(distance_pos_and_pos_centroid1)


        subclass_num = tf.reduce_max(self.target_result_graph_sub) + 1
        ones = tf.ones_like(feature_target)
        current_neg_subclass_count = tf.unsorted_segment_sum(ones, self.target_result_graph_sub, subclass_num)
        current_neg_subclass_count = tf.maximum(current_neg_subclass_count, tf.ones_like(current_neg_subclass_count))    # 防止分母为零
        current_neg_subclass_centroid = tf.unsorted_segment_sum(feature_target, self.target_result_graph_sub, subclass_num) / current_neg_subclass_count

        neg_subclass_centroid_classed = tf.gather(params=current_neg_subclass_centroid, indices=self.target_result_graph_sub)
        distance_neg_subclass_centroid = tf.square(feature_target - neg_subclass_centroid_classed)
        self.disn = tf.unsorted_segment_sum(distance_neg_subclass_centroid, self.target_result_graph_sub, subclass_num) / current_neg_subclass_count




        decay = tf.constant(0.3)
        self.decay = decay

        target_centroid = (decay)*current_target_centroid + \
            (1.-decay)*self.target_moving_centroid
        source_centroid = (decay)*current_source_centroid + \
            (1.-decay)*self.source_moving_centroid

        self.Semanticloss = tf.reduce_mean(tf.square(source_centroid - target_centroid))
        tf.summary.scalar('semanticloss', self.Semanticloss)


        dis_positive_current = tf.reduce_mean(self.disp) 
        dis_negative_current = tf.reduce_mean(self.disn) 

        self.dis_positive0 = dis_positive_current
        self.dis_negative0 = dis_negative_current

        decay = tf.constant(0.3)
        dis_positive = decay * dis_positive_current + (1.0 - decay) * self.dis_positive
        dis_negative = decay * dis_negative_current + (1.0 - decay) * self.dis_negative

        xx = dis_positive * 0.5 + dis_negative * 0.5
        self.total_distance = 1.0 - tf.exp(-1.0 * 0.1*xx)
        tf.summary.scalar('total_distance', self.total_distance)
        



        # caculate JSD
        with tf.variable_scope('reuse') as scope:
            source_logits, _ = D(feature_source)

            scope.reuse_variables()
            target_logits, _ = D(feature_target)

        D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=target_logits, labels=tf.ones_like(target_logits)))
        D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=source_logits, labels=tf.zeros_like(source_logits)))

        self.D_loss = D_real_loss + D_fake_loss
        self.G_loss = -self.D_loss
        tf.summary.scalar('G_loss', self.G_loss)
        tf.summary.scalar('JSD', self.G_loss/2+math.log(2))


        self.G_loss = 0.1*self.G_loss
        self.D_loss = 0.1*self.D_loss
        return dis_positive, dis_negative, source_centroid, target_centroid, smoothness


    # update 2019.03.20
    def closs(self, batch_x_source, batch_x_target_with_label, batch_y_source=None, batch_y_target_with_label=None):
        with tf.variable_scope('reuse_inference') as scope:
            y_predict = self.inference(batch_x_source, training=True)
        
            scope.reuse_variables()   
            y_predict_target_with_label = self.inference(batch_x_target_with_label, training=True)

        self.loss_source_supervised = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_predict, labels=batch_y_source))
        self.loss_target_supervised = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_predict_target_with_label, labels=batch_y_target_with_label))

        self.loss = 0.5 * self.loss_source_supervised + 0.5 * self.loss_target_supervised
        tf.summary.scalar('Closs', self.loss)

        return self.loss

    def optimize(self, learning_rate, train_layers, global_step, dis_positive, dis_negative, sc, tc, smoothness):
        var_list = [v for v in tf.trainable_variables(
        ) if v.name.split('/')[1] in train_layers+['fc9']]
        finetune_list = [v for v in var_list if v.name.split(
            '/')[1] in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']]
        new_list = [v for v in var_list if v.name.split(
            '/')[1] in ['fc8', 'fc9']]
        self.Gregloss = 0.0005 * \
            tf.reduce_mean([tf.nn.l2_loss(x)
                            for x in var_list if 'weights' in x.name])

        finetune_weights = [v for v in finetune_list if 'weights' in v.name]
        finetune_biases = [v for v in finetune_list if 'biases' in v.name]
        new_weights = [v for v in new_list if 'weights' in v.name]
        new_biases = [v for v in new_list if 'biases' in v.name]

        tf.summary.scalar('global_step', global_step)
        self.F_loss =  self.loss + \
            self.Gregloss + global_step * self.G_loss + \
            global_step * self.Semanticloss + \
            global_step **2 * self.total_distance + \
            global_step **8 * self.smoothness + \
            self.loss_new

        train_op1 = tf.train.MomentumOptimizer(
            learning_rate*0.1, 0.9).minimize(self.F_loss, var_list=finetune_weights)
        train_op2 = tf.train.MomentumOptimizer(
            learning_rate*0.2, 0.9).minimize(self.F_loss, var_list=finetune_biases)
        train_op3 = tf.train.MomentumOptimizer(
            learning_rate*1.0, 0.9).minimize(self.F_loss, var_list=new_weights)
        train_op4 = tf.train.MomentumOptimizer(
            learning_rate*2.0, 0.9).minimize(self.F_loss, var_list=new_biases)
        train_op = tf.group(train_op1, train_op2, train_op3, train_op4)

        with tf.control_dependencies([train_op1, train_op2, train_op3, train_op4]):
            update_pdis = self.dis_positive.assign(dis_positive)
            update_ndis = self.dis_negative.assign(dis_negative)
            update_sc = self.source_moving_centroid.assign(sc)
            update_tc = self.target_moving_centroid.assign(tc)
            update_smoothness = self.smoothness.assign(smoothness)


        return tf.group(update_pdis, update_ndis, update_sc, update_tc, update_smoothness)

    def load_original_weights(self, session, skip_layers=[]):
        weights_dict = np.load(
            './MSTN_MODEL/MSTN_models/bvlc_alexnet.npy', encoding='bytes').item()

        for op_name in weights_dict:
            # if op_name in skip_layers:
            #     continue

            if op_name == 'fc8' and self.num_classes != 1000:
                continue

            with tf.variable_scope('reuse_inference/'+op_name, reuse=True):
                #print('=============================OP_NAME  ========================================')
                for data in weights_dict[op_name]:
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases')
                        #print(op_name, var)
                        session.run(var.assign(data))
                    else:
                        var = tf.get_variable('weights')
                        #print(op_name, var)
                        session.run(var.assign(data))


"""
Helper methods
"""


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    input_channels = int(x.get_shape()[-1])

    def convolve(i, k): return tf.nn.conv2d(i, k, strides=[
        1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[
                                  filter_height, filter_width, input_channels/groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(
                axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k)
                             for i, k in zip(input_groups, weight_groups)]
            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases),
                          [-1]+conv.get_shape().as_list()[1:])
        relu = tf.nn.relu(bias, name=scope.name)
        return relu


def D(x):
    with tf.variable_scope('D'):
        num_units_in = int(x.get_shape()[-1])
        num_units_out = 1
        weights = tf.get_variable('weights', initializer=tf.truncated_normal(
            [num_units_in, 1024], stddev=0.01))
        biases = tf.get_variable(
            'biases', shape=[1024], initializer=tf.zeros_initializer())
        hx = (tf.matmul(x, weights)+biases)
        ax = tf.nn.dropout(tf.nn.relu(hx), 0.5)

        weights2 = tf.get_variable(
            'weights2', initializer=tf.truncated_normal([1024, 1024], stddev=0.01))
        biases2 = tf.get_variable(
            'biases2', shape=[1024], initializer=tf.zeros_initializer())
        hx2 = (tf.matmul(ax, weights2)+biases2)
        ax2 = tf.nn.dropout(tf.nn.relu(hx2), 0.5)

        weights3 = tf.get_variable('weights3', initializer=tf.truncated_normal([
                                   1024, num_units_out], stddev=0.3))
        biases3 = tf.get_variable(
            'biases3', shape=[num_units_out], initializer=tf.zeros_initializer())
        hx3 = tf.matmul(ax2, weights3)+biases3
        return hx3, tf.nn.sigmoid(hx3)


def fc(x, num_in, num_out, name, relu=True, stddev=0.01):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', initializer=tf.truncated_normal(
            [num_in, num_out], stddev=stddev))
        biases = tf.get_variable(
            'biases', initializer=tf.constant(0.1, shape=[num_out]))
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def outer(a, b):
    a = tf.reshape(a, [-1, a.get_shape()[-1], 1])
    b = tf.reshape(b, [-1, 1, b.get_shape()[-1]])
    c = a*b
    return tf.contrib.layers.flatten(c)


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

