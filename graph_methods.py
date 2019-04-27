# coding=utf-8
import tensorflow as tf
import numpy as np


"""
Helper methods: 分布
"""


def caculate_dis_matrix(feature0, feature1, similarity=False, exp_pos=False, gamma=0.01):
    """
    Return: Square E distance
        shape = [picnumber0, picnumber1]
    """
    def caculate_dis_matrix_cond(_square, feature0, feature1, i):
        picture_number0 = tf.shape(feature0)[0]
        return i < tf.cast(picture_number0, dtype=tf.int32)

    def caculate_dis_matrix_body(_square, feature0, feature1, i):
        feature0 = tf.cast(feature0, dtype=tf.float32)
        feature1 = tf.cast(feature1, dtype=tf.float32)

        square = tf.square(feature0[i] - feature1)
        square_dis = tf.reduce_sum(square, axis=1)  # shape=[picture_number1]
        _square = tf.concat([_square, square_dis], axis=0)

        i += 1

        return _square, feature0, feature1, i

    i = tf.constant(0)
    _square = tf.Variable([], trainable=False)

    _square, _, _, i = tf.while_loop(
        caculate_dis_matrix_cond,
        caculate_dis_matrix_body,
        [_square, feature0, feature1, i],
        shape_invariants=[tf.TensorShape([None]), feature0.get_shape(
        ), feature1.get_shape(), i.get_shape()]
    )

    picture_number0 = tf.cast(tf.shape(feature0)[0], dtype=tf.int32)
    picture_number1 = tf.cast(tf.shape(feature1)[0], dtype=tf.int32)

    _square_dis = tf.reshape(tf.cast(tf.stack(_square, axis=0), tf.float32), [
                             picture_number0, picture_number1])

    if similarity:
        if exp_pos:
            _square_dis = tf.exp(1.0*gamma*_square_dis) - 1.0
        else:
            _square_dis = tf.exp(-1.0*gamma*_square_dis)

    return _square_dis


def find_n_max_row(matrix, number=0):
    n = tf.shape(matrix)[0]
    if number == 0:
        number = tf.cast(n/4, dtype=tf.int32) + 1
    else:
        number = tf.constant(number)
        number = tf.minimum(n, number)

    i = tf.constant(0)
    data = tf.Variable([], dtype=tf.int32, trainable=False)

    def condition(matrix, data, i, n, number):
        return i < n

    def body(matrix, data, i, n, number):
        values, _ = tf.nn.top_k(matrix[i], k=number)
        data_current = tf.cast(tf.greater_equal(
            matrix[i], values[number-1]), tf.int32)
        data = tf.concat([data, data_current], axis=0)
        i += 1
        return matrix, data, i, n, number

    matrix, data, i, n, number = tf.while_loop(
        condition, body,
        [matrix, data, i, n, number],
        shape_invariants=[
            matrix.get_shape(),
            tf.TensorShape([None]),
            i.get_shape(),
            n.get_shape(),
            number.get_shape()
        ]
    )

    data = tf.reshape(data, shape=[n, n])
    return data


def label_create(data):
    n = tf.shape(data)[0]
    data -= tf.eye(n, dtype=tf.int32)
    visit = tf.Variable([], dtype=tf.int32, trainable=False)

    i = tf.constant(0)
    count = tf.constant(0)

    def condition_1(visit, i, n):
        return i < n

    def body_1(visit, i, n):
        visit = tf.concat([visit, tf.reshape(i, [1])], axis=0)
        i += 1
        return visit, i, n

    visit, i, n = tf.while_loop(
        condition_1,
        body_1,
        [visit, i, n],
        shape_invariants=[
            tf.TensorShape([None]),
            i.get_shape(),
            n.get_shape()
        ]
    )

    #visit = tf.reshape(visit, shape=[n, 1])

    def condition_2(data, visit, i, n):
        return i < n

    def body_2(data, visit, i, n):
        visit = visit * (1 - data[i]) + data[i] * visit[i]
        i += 1
        return data, visit, i, n

    i = tf.constant(0)
    data, visit, i, n = tf.while_loop(
        condition_2,
        body_2,
        [data, visit, i, n]
    )

    def condition_3(visit, result, count):
        return tf.reduce_max(visit) >= 0

    def body_3(visit, result, count):
        temp = tf.cast(tf.equal(visit, tf.reduce_max(visit)), dtype=tf.int32)
        result += temp * count
        visit -= temp * (visit + 1)
        count += 1
        return visit, result, count

    count = 0
    result = tf.zeros_like(visit, dtype=tf.int32)

    visit, result, count = tf.while_loop(
        condition_3,
        body_3,
        [visit, result, count]
    )

    return result


def class_use_graph(distance_matrix, k=10):
    """
    使用knn图分类
        输入：距离矩阵(n,n)
        输出：行向量，分类结果(n)

    """
    k += 1
    matrix = -1.0 * distance_matrix
    index_row = find_n_max_row(matrix, number=k)
    index = index_row * tf.transpose(index_row)
    result = label_create(index)
    return result


def caculate_subclass_distance(feature, result):
    picture_number = tf.size(result)
    class_number = tf.reduce_max(result) + 1
    distance = tf.Variable([], dtype=tf.float32, trainable=False)
    i = tf.constant(0)

    def condition(feature, result, i, class_number, distance):
        return i < class_number

    def body(feature, result, i, class_number, distance):
        feature_current_class = tf.gather_nd(
            params=feature, indices=tf.cast(tf.where(tf.equal(result, i)), tf.int32))
        distance_current = tf.reduce_mean(caculate_dis_matrix(
            feature_current_class, feature_current_class, similarity=True, exp_pos=True))
        distance = tf.concat(
            [distance, tf.reshape(distance_current, [1])], axis=0)
        i += 1
        return feature, result, i, class_number, distance

    feature, result, i, class_number, distance = tf.while_loop(
        cond=condition,
        body=body,
        loop_vars=[feature, result, i, class_number, distance],
        shape_invariants=[
            feature.get_shape(),
            result.get_shape(),
            i.get_shape(),
            class_number.get_shape(),
            tf.TensorShape([None])
        ]
    )

    return distance


def pca_via_svd(M, r):
    '''
    M: 原始矩阵
    r: 新的矩阵长度
    作者：巾梵
    链接：https://www.jianshu.com/p/107196a8f7f0
    來源：简书
    简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。
    '''
    # 将输入矩阵做奇异值分解
    s, u, v = tf.svd(M)
    # 使用奇异值将矩阵的第二维压缩到指定长度
    return tf.matmul(u[:, :r], tf.diag(s[:r]))


def caculate_smoothness(feature_all, label_all, gamma=1.0):
    label_all = tf.cast(label_all, tf.float32)

    n = tf.shape(label_all)[0]
    y = tf.reshape(label_all, shape=[1, n])
    W = tf.exp(-1.0 * gamma * caculate_dis_matrix(feature_all,
                                                  feature_all)) - tf.eye(n)
    D = tf.diag(tf.reduce_sum(W, axis=1))
    L = D - W
    smoothness = tf.matmul(tf.matmul(y, L), tf.transpose(y))
    smoothness_exp = tf.reduce_sum(1.0 - tf.exp(-1.0 * 0.001 * smoothness))

    return smoothness_exp


#feature_neg_low = pca_via_svd(feature_negative, 128)
#        negative_class_result = class_use_graph(caculate_dis_matrix(feature_neg_low, feature_neg_low), k=10)


def class_feature_use_graph(feature, pca_dimension=10, k_neighbor=10):
    """
    使用图分类特征
        若feature为[n*256], result为[n]
    """
    feature_low = pca_via_svd(feature, pca_dimension)
    dis_matrix = caculate_dis_matrix(feature_low, feature_low)
    result = class_use_graph(dis_matrix, k_neighbor)
    return result


def class_target_result_with_source_centroid(target_feature, target_result, source_centroid):
    i = tf.constant(0)
    result_num = tf.shape(target_feature)
    final_result = 0 * target_result
    target_class_num = tf.reduce_max(target_result)

    def loop_cond(target_feature, target_result, source_centroid, final_result, i, target_class_num):
        return i <= target_class_num

    def loop_body(target_feature, target_result, source_centroid, final_result, i, target_class_num):
        feature_current = tf.gather_nd(params=target_feature, indices=tf.where(tf.equal(target_result, i)))
        current_target_class_centroid = tf.reduce_mean(feature_current, axis=0)

        current_class_result = tf.argmin(tf.reduce_sum(
            tf.square(current_target_class_centroid - source_centroid), axis=1), axis=0)
        final_result += tf.cast(current_class_result, tf.int32) * \
            tf.cast(tf.reshape(tf.equal(target_result, i), [-1]), dtype=tf.int32)
        i += 1
        return target_feature, target_result, source_centroid, final_result, i, target_class_num

    [target_feature, target_result, source_centroid, final_result, i, target_class_num] = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars=[target_feature, target_result,
                   source_centroid, final_result, i, target_class_num],
        shape_invariants=[
            target_feature.get_shape(),
            target_result.get_shape(),
            source_centroid.get_shape(),
            final_result.get_shape(),
            i.get_shape(),
            target_class_num.get_shape()
        ]
    )

    return final_result


def caculate_final_result(result, neg_result):
    result = tf.cast(result, tf.int32)
    neg_result = tf.cast(neg_result, tf.int32)
    neg_index = tf.cast(tf.reshape(
        tf.where(tf.equal(result, 0)), [-1]), tf.int32)
    neg_result_same_shape = tf.cast(tf.zeros_like(result), tf.int32)

    i = tf.constant(0, dtype=tf.int32)
    n_total = tf.shape(result)[0]
    n = tf.shape(neg_result)[0]

    def condition_c(neg_result_same_shape, neg_result, neg_index, i, n, n_total):
        return i < n

    def body_c(neg_result_same_shape, neg_result, neg_index, i, n, n_total):
        neg_result_same_shape += neg_result[i] * tf.one_hot(
            indices=neg_index[i], depth=n_total, dtype=tf.int32)
        i += 1
        return neg_result_same_shape, neg_result, neg_index, i, n, n_total

    neg_result_same_shape, neg_result, neg_index, i, n, n_total = tf.while_loop(
        condition_c, body_c,
        loop_vars=[neg_result_same_shape,
                   neg_result, neg_index, i, n, n_total],
        shape_invariants=[
            neg_result_same_shape.get_shape(),
            neg_result.get_shape(),
            neg_index.get_shape(),
            i.get_shape(),
            n.get_shape(),
            n_total.get_shape()
        ]
    )

    return neg_result_same_shape + (-1) * result


def main():
    """
    from test_tf import feature_read, feature_reduce_dimension, feature_caculate_distance
    import shutil
    import os

    feature = feature_read("./feature100.txt")
    feature_low = feature_reduce_dimension(
        feature,
        final_dimension=2,
        use=False
    )

    k = 11
    feature_low = feature_low[50:100]
    feature_distance_matrix = feature_caculate_distance(feature_low)

    matrix = tf.constant(-1.0 * feature_distance_matrix)
    data_0 = find_n_max_row(matrix, number=k)
    data = data_0 * tf.transpose(data_0)
    result = label_create(data)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result, d0, _data = sess.run([result, data_0, data])

    print(result)
    import datetime
    current_name = datetime.datetime.now()
    save_pic('./target/', './class' + str(current_name) + '/', result)
    n_clusters_ = np.max(result) + 1
    X = feature_low

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.close('all')
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = result == k
        print('Class {}: {}'.format(k, np.where(class_members)))
        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')


    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    for k in range(n_clusters_):
        class_members = result == k
        print('Class {}: {}'.format(k, np.where(class_members)))

    """


if __name__ == "__main__":
    sys.exit(main())
