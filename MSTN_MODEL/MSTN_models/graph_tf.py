from test_tf import feature_read, feature_reduce_dimension, feature_caculate_distance
import tensorflow as tf
import numpy as np
import shutil
import sys
import os


def find_n_max_row(matrix, number=1):
    n = tf.shape(matrix)[0]
    number = tf.constant(number)
    i = tf.constant(0)
    data = tf.constant([], dtype=tf.int32)

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
    visit = tf.constant([], dtype=tf.int32)

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



def save_pic(path_original, path_target, class_result):
    class_number = np.max(class_result) + 1
    if not os.path.exists(path_target):
        os.makedirs(path_target)

    for k in range(class_number):
        os.mkdir(path_target + str(k))
        picture_list = np.where(class_result == k)[0]
        for i in range(len(picture_list)):
            shutil.copyfile(
                src=path_original + 'n' + str(picture_list[i] + 1) + '.jpg',
                dst=path_target + str(k) + '/n' +
                str(picture_list[i] + 1) + '.jpg'
            )


def class_use_graph(distance_matrix, k=10):
    """
    使用knn图分类
        输入：距离矩阵(n,n)
        输出：行向量，分类结果(n)

    """
    k += 1
    matrix = -1.0 * distance_matrix
    index_row = find_n_max_row(matrix, number=k)
    index = index * tf.transpose(index)
    result = label_create(index)
    return result


def get_subclass_distance(feature, result):
    picture_number = tf.size(result)
    class_number = tf.reduce_max(result) + 1
    distance = tf.constant([])
    i = tf.constant(0)

    def condition(feature, result, i, class_number):
        return i < class_number

    def body(feature, result, i, class_number):
        feature_current_class = tf.gather_nd(params=feature, indices=tf.cast(tf.where(tf.equal(result, i)), tf.int32))
        distance_current = 



def main():
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
    """
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
    """
    for k in range(n_clusters_):
        class_members = result == k
        print('Class {}: {}'.format(k, np.where(class_members)))


if __name__ == "__main__":
    sys.exit(main())
