# coding=utf-8
import datetime
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # pca needs
import sys
from itertools import cycle
from math import isnan

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import decomposition

from MSTN_MODEL.MSTN_models.mstnmodel_new import AlexNetModel
from MSTN_MODEL.MSTN_models.mstnmodel_noTL import AlexNetModel as AlexNetModel_noTL
from MSTN_MODEL.MSTN_models.preprocessor import BatchPreprocessor

tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          'Learning rate for adam optimizer')  # 0.001
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 1000,
                            'Number of epochs for training')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size')
tf.app.flags.DEFINE_string(
    'train_layers', 'fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '256,257',
                           'As preprocessing; scale the image randomly between 2 numbers and crop randomly at networs input size')
tf.app.flags.DEFINE_string(
    'train_root_dir', './MSTN_MODEL/MSTN_models/training', 'Root directory to put the training data')
tf.app.flags.DEFINE_integer(
    'log_step', 100, 'Logging period in terms of iteration')

NUM_CLASSES = 2
#TRAINING_FILE = './MSTN_models/source.txt'
#VAL_FILE = './test_img_predict_classes/picture_labels.txt'
#MODEL_PATH = "./trained_mstn_model/mstnmodel_test_source_to_target2500.ckpt"
FLAGS = tf.app.flags.FLAGS
MAX_STEP = 10000
MODEL_NAME = 'mstn'
BATCH_MAX = 200


def decay(start_rate, epoch, num_epochs):
    return start_rate/pow(1+0.001*epoch, 0.75)


def adaptation_factor(x):
    if x >= 1.0:
        return 1.0
    den = 1.0+math.exp(-10*x)
    lamb = 2.0/den-1.0
    return lamb


def mstn_trainmodel(TARGET_LABEL_FILE, TRAINING_FILE, VAL_FILE, val_file_num=100, epochs_limit=2500, step_log=True, model_name='mstn'):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir):
        os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir):
        os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir):
        os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    x = tf.placeholder(tf.float32, [None, 227, 227, 3], 'x')
    xt = tf.placeholder(tf.float32, [None, 227, 227, 3], 'xt')
    # add at 2019.03.20
    xtl = tf.placeholder(tf.float32, [None, 227, 227, 3], 'xtl')
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'y')
    yt = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'yt')
    # add at 2019.03.20
    ytl = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'ytl')
    adlamb = tf.placeholder(tf.float32)
    decay_learning_rate = tf.placeholder(tf.float32)
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = AlexNetModel(num_classes=NUM_CLASSES,
                         dropout_keep_prob=dropout_keep_prob)
    closs = model.closs(
        batch_x_source=x,
        batch_x_target_with_label=xtl,
        batch_y_source=y,
        batch_y_target_with_label=ytl
    )

    feature = model.feature
    result = model.result

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    correct_test = 100 * tf.argmax(model.score, 1) + tf.argmax(y, 1)
    correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    dis_positive, dis_negative, sc, tc, smoothness = model.adloss(
        x, xt, y, adlamb, xtl, ytl)

    target_correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(yt, 1))
    target_correct = tf.reduce_sum(tf.cast(target_correct_pred, tf.float32))
    target_accuracy = tf.reduce_mean(tf.cast(target_correct_pred, tf.float32))

    # optimizer
    train_op = model.optimize(
        decay_learning_rate,
        train_layers,
        adlamb,
        dis_positive,
        dis_negative,
        sc,
        tc,
        smoothness
    )
    D_op = model.adoptimize(decay_learning_rate, train_layers)

    optimizer = tf.group(train_op, D_op)

    train_writer = tf.summary.FileWriter(
        './MSTN_MODEL/MSTN_train_log/tensorboard'+MODEL_NAME)
    train_writer.add_graph(tf.get_default_graph())
    tf.summary.scalar('Testing Accuracy', target_accuracy)
    merged = tf.summary.merge_all()

    print('============================GLOBAL TRAINABLE VARIABLES ============================')
    print(tf.trainable_variables())

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None
    print('==================== MULTI SCALE===================================================')
    print(multi_scale)
    train_preprocessor_source = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                                  output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    train_preprocessor_target = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES,
                                                  output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor_target = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES,
                                                output_size=[227, 227], multi_scale=multi_scale, istraining=False)
    val_preprocessor_source = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                                output_size=[227, 227], multi_scale=multi_scale, istraining=False)
    print('Initialize done.')

    train_preprocessor_target_with_label = BatchPreprocessor(dataset_file_path=TARGET_LABEL_FILE, num_classes=NUM_CLASSES,
                                                             output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor_target_with_label = BatchPreprocessor(dataset_file_path=TARGET_LABEL_FILE, num_classes=NUM_CLASSES,
                                                           output_size=[227, 227], multi_scale=multi_scale, istraining=False)
    print('Backgroung initialize done.')

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch_source = np.floor(
        len(train_preprocessor_source.labels) / FLAGS.batch_size).astype(np.int16)
    train_batches_per_epoch_target = np.floor(
        len(train_preprocessor_target.labels) / FLAGS.batch_size).astype(np.int16)
    train_batches_per_epoch_target_with_label = np.floor(
        len(train_preprocessor_target_with_label.labels) / FLAGS.batch_size).astype(np.int16)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_writer.add_graph(sess.graph)
        # Load the pretrained weights
        model.load_original_weights(sess, skip_layers=train_layers)

        print(("{} Start training...".format(datetime.datetime.now())))
        print(("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir)))
        gs = 0
        gd = 0
        exit_flag = False
        for epoch in range(FLAGS.num_epochs):
            if exit_flag:
                break

            print("Epoch number: {}".format(epoch+1))
            step = 1
            # Start training
            while step < train_batches_per_epoch_source:
                gd += 1
                print(gd)
                lamb = adaptation_factor(gd*1.0/MAX_STEP)
                rate = decay(FLAGS.learning_rate, gd, MAX_STEP)

                for it in range(1):
                    gs += 1
                    if gs % train_batches_per_epoch_target == 0:
                        train_preprocessor_target.reset_pointer()
                    if gs % train_batches_per_epoch_source == 0:
                        train_preprocessor_source.reset_pointer()
                    if gs % train_batches_per_epoch_target_with_label == 0:
                        train_preprocessor_target_with_label.reset_pointer()

                    batch_x_source, batch_y_source = train_preprocessor_source.next_batch(
                        FLAGS.batch_size)
                    batch_x_target, batch_y_target = train_preprocessor_target.next_batch(
                        FLAGS.batch_size)
                    batch_x_target_with_label, batch_y_target_with_label = train_preprocessor_target_with_label.next_batch(
                        FLAGS.batch_size)

                    summary, _ = sess.run(
                        [merged, optimizer],
                        feed_dict={
                            x: batch_x_source,
                            xt: batch_x_target,
                            xtl: batch_x_target_with_label,
                            yt: batch_y_target,
                            ytl: batch_y_target_with_label,
                            adlamb: lamb,
                            decay_learning_rate: rate,
                            y: batch_y_source,
                            dropout_keep_prob: 0.5
                        }
                    )
                    train_writer.add_summary(summary, gd)

                closs, gloss, dloss, gregloss, dregloss, floss, totaldis, semloss, smoothness = sess.run(
                    [
                        model.loss,
                        model.G_loss,
                        model.D_loss,
                        model.Gregloss,
                        model.Dregloss,
                        model.F_loss,
                        model.total_distance,
                        model.Semanticloss,
                        model.smoothness
                    ],
                    feed_dict={
                        x: batch_x_source,
                        xt: batch_x_target,
                        xtl: batch_x_target_with_label,
                        adlamb: lamb,
                        decay_learning_rate: rate,
                        y: batch_y_source,
                        ytl: batch_y_target_with_label,
                        dropout_keep_prob: 0.5
                    }
                )

                step += 1

                if gd % 20 == 0 and step_log == True:
                    print(
                        '=================== Step {} ================='.format(gs))
                    print('Epoch {} Step {} \nCloss {} \nGloss {} \nDloss {} \nTotal_Loss {} \nGregloss {} \nDregloss {} \ntotal distance {}\nsemloss {}'.format(
                        epoch, step, closs, gloss, dloss, floss, gregloss, dregloss, totaldis, semloss))
                    print('lambda: ', lamb)
                    print('rate: ', rate)

                    # Epoch completed, start validation
                    print(("{} Start validation".format(datetime.datetime.now())))

                    # Draw feature space and calculate score
                    _result_total = np.zeros([val_file_num], dtype=np.int32)
                    batch_x_target, batch_y_target = val_preprocessor_target.next_batch(
                        val_file_num)
                    batch_x_target_with_label, batch_y_target_with_label = val_preprocessor_target_with_label.next_batch(
                        val_file_num)
                    batch_x_source, batch_y_source = val_preprocessor_source.next_batch(
                        200)

                    _result_graph_sub, _result_graph_total, _feature_source, _feature_target, _result_source, _SS_s, _SS_t = sess.run(
                        [
                            model.result_graph,
                            model.target_result_new,
                            model.feature_source,
                            model.feature_target,
                            model.result_source,
                            model.SemanticScore_source,
                            model.SemanticScore_target
                        ],
                        feed_dict={
                            x: batch_x_source,
                            xt: batch_x_target,
                            xtl: batch_x_target_with_label,
                            y: batch_y_source,
                            ytl: batch_y_target_with_label,
                            dropout_keep_prob: 1.0
                        }
                    )

                    print(_result_graph_sub, _result_graph_total)
                    print(_SS_s, _SS_t)

                    draw_feature_graph(
                        _feature_source,
                        _feature_target,
                        _result_source,
                        _result_graph_sub,
                        _result_graph_total,
                        './MSTN_MODEL/MSTN_train_log/' + train_dir_name + model_name,
                        gd
                    )

                    # Reset the dataset pointers
                    val_preprocessor_target.reset_pointer()
                    val_preprocessor_target_with_label.reset_pointer()
                    val_preprocessor_source.reset_pointer()

                if gd % 500 == 0 and gd > 0:
                    saver.save(sess, './MSTN_MODEL/MSTN_train_log/' + train_dir_name + '/' +
                               model_name+str(gd)+'.ckpt')
                    print(("{} Saving checkpoint of model...".format(
                        datetime.datetime.now())))

                if gd >= epochs_limit:
                    exit_flag = True

                    # Draw feature space and calculate score
                    _result_total = np.zeros([val_file_num], dtype=np.int32)
                    batch_x_target, batch_y_target = val_preprocessor_target.next_batch(
                        val_file_num)
                    batch_x_target_with_label, batch_y_target_with_label = val_preprocessor_target_with_label.next_batch(
                        val_file_num)
                    batch_x_source, batch_y_source = val_preprocessor_source.next_batch(
                        200)

                    _result_graph_sub, _result_graph_total, _feature_source, _feature_target, _result_source, _SS_s, _SS_t = sess.run(
                        [
                            model.result_graph,
                            model.target_result_new,
                            model.feature_source,
                            model.feature_target,
                            model.result_source,
                            model.SemanticScore_source,
                            model.SemanticScore_target
                        ],
                        feed_dict={
                            x: batch_x_source,
                            xt: batch_x_target,
                            xtl: batch_x_target_with_label,
                            y: batch_y_source,
                            ytl: batch_y_target_with_label,
                            dropout_keep_prob: 1.0
                        }
                    )

                    print(_result_graph_sub, _result_graph_total)
                    print(_SS_s, _SS_t)

                    draw_feature_graph(_feature_source, _feature_target, _result_source, _result_graph_sub,
                                       _result_graph_total, './MSTN_MODEL/MSTN_train_log/' + train_dir_name, gd)

                    break

        last_model_path = './MSTN_MODEL/trained_models/' + \
            model_name + str(gd)+'.ckpt'
        saver.save(sess, last_model_path)
        print(("{} Saving checkpoint of model...".format(datetime.datetime.now())))

    return last_model_path, _result_graph_total, [_SS_s, _SS_t]


def mstn_label_with_model(TRAINING_FILE, TARGET_LABEL_FILE, VAL_FILE, val_file_num=100, model_name='mstn', train_epoch=500, MODEL_PATH='null', SS_limit=0.2):
    """
    使用训练好的mstn模型给hard样本打标签

    返回值：
    [labellist_negetive, labellist_positive]:
        labellist_negetive：标记为0的图片引索
        labellist_positive：标记为1的图片引索
    feature_total:
        所有hard样本特征向量组成的array

    """
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir):
        os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir):
        os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir):
        os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    x = tf.placeholder(tf.float32, [None, 227, 227, 3], 'x')
    xt = tf.placeholder(tf.float32, [None, 227, 227, 3], 'xt')
    # add at 2019.03.20
    xtl = tf.placeholder(tf.float32, [None, 227, 227, 3], 'xtl')
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'y')
    yt = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'yt')
    # add at 2019.03.20
    ytl = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'ytl')
    adlamb = tf.placeholder(tf.float32)
    decay_learning_rate = tf.placeholder(tf.float32)
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = AlexNetModel(num_classes=NUM_CLASSES,
                         dropout_keep_prob=dropout_keep_prob)
    closs = model.closs(
        batch_x_source=x,
        batch_x_target_with_label=xtl,
        batch_y_source=y,
        batch_y_target_with_label=ytl
    )

    feature = model.feature
    result = model.result

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    correct_test = 100 * tf.argmax(model.score, 1) + tf.argmax(y, 1)
    correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    dis_positive, dis_negative, sc, tc, smoothness = model.adloss(
        x, xt, y, adlamb, xtl, ytl)

    target_correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(yt, 1))
    target_correct = tf.reduce_sum(tf.cast(target_correct_pred, tf.float32))
    target_accuracy = tf.reduce_mean(tf.cast(target_correct_pred, tf.float32))

    # optimizer
    train_op = model.optimize(
        decay_learning_rate,
        train_layers,
        adlamb,
        dis_positive,
        dis_negative,
        sc,
        tc,
        smoothness
    )
    D_op = model.adoptimize(decay_learning_rate, train_layers)

    optimizer = tf.group(train_op, D_op)

    train_writer = tf.summary.FileWriter(
        './MSTN_MODEL/MSTN_train_log/tensorboard'+MODEL_NAME)
    train_writer.add_graph(tf.get_default_graph())
    tf.summary.scalar('Testing Accuracy', target_accuracy)
    merged = tf.summary.merge_all()

    #print('============================GLOBAL TRAINABLE VARIABLES ============================')
    # print(tf.trainable_variables())

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None
    #print('==================== MULTI SCALE===================================================')
    # print(multi_scale)
    train_preprocessor_source = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                                  output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    train_preprocessor_target = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES,
                                                  output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor_target = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES, output_size=[
        227, 227], multi_scale=multi_scale, istraining=False)
    val_preprocessor_source = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                                output_size=[227, 227], multi_scale=multi_scale, istraining=False)
    print('Initialize done.')

    train_preprocessor_target_with_label = BatchPreprocessor(dataset_file_path=TARGET_LABEL_FILE, num_classes=NUM_CLASSES,
                                                             output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor_target_with_label = BatchPreprocessor(dataset_file_path=TARGET_LABEL_FILE, num_classes=NUM_CLASSES,
                                                           output_size=[227, 227], multi_scale=multi_scale, istraining=False)
    print('Backgroung initialize done.')

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch_source = np.floor(
        len(train_preprocessor_source.labels) / FLAGS.batch_size).astype(np.int16)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_writer.add_graph(sess.graph)
        # Load the pretrained weights
        #model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        if MODEL_PATH == 'null':
            MODEL_PATH = "./MSTN_MODEL/trained_models/" + model_name + str(train_epoch) + ".ckpt"

        saver.restore(sess, MODEL_PATH)

        #print(("{} Start training...".format(datetime.datetime.now())))
        #print(("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir)))
        gs = 0
        gd = 0
        labellist_positive = []
        labellist_negetive = []
        for epoch in range(FLAGS.num_epochs):
            #print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1
            # Start training
            while step < train_batches_per_epoch_source:
                gd += 1
                lamb = adaptation_factor(gd*1.0/MAX_STEP)
                rate = decay(FLAGS.learning_rate, gd, MAX_STEP)
                if gd % 1 == 0:
                    print(("{} Start validation".format(datetime.datetime.now())))
                    batch_max = BATCH_MAX

                    result_graph_total = []
                    result_graph_sub = []
                    SS_s = []
                    SS_t = []

                    for batch_start_num in range(1, val_file_num, batch_max):
                        _result_total = np.zeros(
                            [val_file_num], dtype=np.int32)

                        if val_file_num - batch_start_num < batch_max - 1:
                            batch_num_current = val_file_num - batch_start_num + 1
                        else:
                            batch_num_current = batch_max

                        batch_x_target, batch_y_target = val_preprocessor_target.next_batch(
                            batch_num_current)
                        batch_x_target_with_label, batch_y_target_with_label = val_preprocessor_target_with_label.next_batch(
                            batch_num_current)
                        batch_x_source, batch_y_source = val_preprocessor_source.next_batch(
                            batch_num_current)

                        _result_graph_sub, _result_graph_total, _feature_source, _feature_target, _result_source, _SS_s, _SS_t = sess.run(
                            [
                                model.result_graph,
                                model.target_result_new,
                                model.feature_source,
                                model.feature_target,
                                model.result_source,
                                model.SemanticScore_source,
                                model.SemanticScore_target
                            ],
                            feed_dict={
                                x: batch_x_source,
                                xt: batch_x_target,
                                xtl: batch_x_target_with_label,
                                y: batch_y_source,
                                ytl: batch_y_target_with_label,
                                dropout_keep_prob: 1.0
                            }
                        )

                        result_graph_total.append(_result_graph_total)
                        result_graph_sub.append(_result_graph_sub)
                        SS_s.append(_SS_s)
                        SS_t.append(_SS_t)

                        log_feature_low(
                            _feature_source,
                            _feature_target,
                            _result_source,
                            _result_graph_total,
                            _result_graph_sub,
                            SS=_SS_s+_SS_t,
                            savedir='./MSTN_MODEL/MSTN_train_log/' + train_dir_name + model_name + '/',
                            logname=str(batch_start_num),
                            SS_limit=SS_limit
                        )

                        print("Validation %.2f %% ..." %
                              (100*val_preprocessor_target.pointer/val_file_num))

                        # Reset the dataset pointers
                        val_preprocessor_target_with_label.reset_pointer()
                        val_preprocessor_source.reset_pointer()

                    print("Validation done.")
                    result_graph_total = np.array(
                        [i for item in result_graph_total for i in item])
                    result_graph_sub = np.array(
                        [i for item in result_graph_sub for i in item]
                    )
                    SS_s = np.array([i for item in SS_s for i in item])
                    SS_t = np.array([i for item in SS_t for i in item])

                    return result_graph_total, [SS_s, SS_t], result_graph_sub


def feature_reduce_dimension(feature, final_dimension):
    """
    使用PCA方法降维特征
        feature_list:array类型保存的特征
        final_dimension:需要降低到的维度
    返回值
        array类型降维后数据，每一行表示一张图片的降维后的特征
    """

    feature_array = np.array(feature).T
    pca = decomposition.PCA(n_components=final_dimension)
    pca.fit(feature_array)

    return pca.components_.T


def draw_feature(feature, result, savedir, count, show=False):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    result = np.array(result)
    acc_p = 2 * np.sum((result == -1)[0:50])
    acc_n = 2 * np.sum((result >= 0)[50:100])

    feature_low = feature_reduce_dimension(feature, 2)
    colors = cycle('brcmykrcmykrcmykrcmyk')

    fig = plt.figure(count)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('step = ' + str(count) + ' p = '+str(acc_p) + ' n = '+str(acc_n))

    for k, color in zip(range(np.min(result), np.max(result)+1), colors):
        class_members = np.array(result) == k
        if k == -1:
            color_current = 'g'
        else:
            color_current = color

        plt.scatter(feature_low.T[0][class_members], feature_low.T[1][class_members], s=200, c=color_current, alpha=0.5,
                    marker='o', label='class ' + str(k))

    plt.scatter(feature_low.T[0][0:50], feature_low.T[1][0:50], c='g', alpha=1,
                marker='+', label='people')
    plt.scatter(feature_low.T[0][50:100], feature_low.T[1][50:100], c='b', alpha=1,
                marker='+', label='background')

    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(savedir + '/' + str(count) + '.jpg')
    if show:
        plt.show()


def draw_feature_graph(feature_source, feature_target, result_source, result_target_sub, result_target_total, savedir, count, show=False):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    result = np.array(result_target_total)
    acc_p = 2 * np.sum((result_target_total == 1)[0:50])
    acc_n = 2 * np.sum((result_target_total == 0)[50:100])

    source_num = np.shape(feature_source)
    target_num = np.shape(feature_target)

    feature_total = np.concatenate([feature_source, feature_target], axis=0)
    feature_total_low = feature_reduce_dimension(feature_total, 2)

    feature_source_low = feature_total_low[0:source_num[0]]
    feature_target_low = feature_total_low[source_num[0]
        :source_num[0]+target_num[0]]
    color1 = cycle('bgr')
    color2 = cycle('bgr')
    colors = cycle('cmykcmykcmykcmyk')

    fig = plt.figure(count)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('step = ' + str(count) + ' p = '+str(acc_p) + ' n = '+str(acc_n))

    for k, c1 in zip(range(np.min(result_source), np.max(result_source)+1), color1):
        class_members = np.array(result_source) == k
        color_current = c1
        plt.scatter(feature_source_low.T[0][class_members], feature_source_low.T[1][class_members], s=300, c=color_current, alpha=0.3,
                    marker='*', label='Sclass ' + str(k))

    for k, c2 in zip(range(np.min(result_target_total), np.max(result_target_total)+1), color2):
        class_members = np.array(result_target_total) == k
        color_current = c2
        plt.scatter(feature_target_low.T[0][class_members], feature_target_low.T[1][class_members], s=300, c=color_current, alpha=0.5,
                    marker='o', label='Tclass ' + str(k))

    for k, color in zip(range(np.min(result_target_sub), np.max(result_target_sub)+1), colors):
        class_members = np.array(result_target_sub) == k
        color_current = color
        plt.scatter(feature_target_low.T[0][class_members], feature_target_low.T[1][class_members], s=150, c=color_current, alpha=0.5,
                    marker='o', label='subclass ' + str(k))

    plt.scatter(feature_target_low.T[0][0:50], feature_target_low.T[1][0:50], c='g', alpha=1,
                marker='+', label='people')
    plt.scatter(feature_target_low.T[0][50:100], feature_target_low.T[1][50:100], c='b', alpha=1,
                marker='+', label='background')

    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(savedir + '/' + str(count) + '.png')
    if show:
        plt.show()


def log_feature_low(feature_source, feature_target, result_source, result_target_total, result_target_sub, SS, savedir, logname, SS_limit=0.2, use_SS=True):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    result_target = np.array(result_target_total)

    target_positive_index = np.where(result_target == 1)
    target_negative_index = np.where(result_target == 0)

    source_positive_index = np.where(result_source == 1)
    source_negative_index = np.where(result_source == 0)

    source_num = np.shape(feature_source)
    target_num = np.shape(feature_target)

    feature_total = np.concatenate([feature_source, feature_target], axis=0)
    feature_total_low = feature_reduce_dimension(feature_total, 2)

    feature_source_low = feature_total_low[0:source_num[0]]
    feature_source_low_p = feature_source_low[source_positive_index]
    feature_source_low_n = feature_source_low[source_negative_index]

    feature_target_low = feature_total_low[source_num[0]
            :source_num[0]+target_num[0]]

    if use_SS:
        SS_limit_index_low = np.where(SS < SS_limit)
        SS_limit_high = SS >= SS_limit
        RTP = result_target == 1
        RTN = result_target == 0

        SS_limit_index_high_p = np.where(RTP * SS_limit_high)
        SS_limit_index_high_n = np.where(RTN * SS_limit_high)

        
        feature_target_low_p = feature_target_low[SS_limit_index_high_p]
        feature_target_low_n = feature_target_low[SS_limit_index_high_n]
        feature_target_useless = feature_target_low[SS_limit_index_low]
        subclasses_target = result_target_sub[SS_limit_index_low]

    else:
        feature_target_low_p = feature_target_low[result_target==1]
        feature_target_low_n = feature_target_low[result_target==0]

    fig = plt.figure()
    # source domain: red
    plt.plot(feature_source_low_p[:, 0],
             feature_source_low_p[:, 1], "r^", alpha=0.5)
    plt.plot(feature_source_low_n[:, 0],
             feature_source_low_n[:, 1], "ro", alpha=0.5)

    # target domain: blue
    plt.scatter(feature_target_low_p[:, 0],
                feature_target_low_p[:, 1], marker="^", c='b')
    plt.scatter(feature_target_low_n[:, 0],
                feature_target_low_n[:, 1], marker="o", c='b')

    if use_SS:
        cm = plt.cm.get_cmap('RdYlBu')
        plt.scatter(feature_target_useless[:, 0], feature_target_useless[:, 1], marker="s", c=subclasses_target, cmap=cm, alpha=0.5)

    plt.axis('off')
    plt.savefig(savedir + "feature" + logname + ".png")


def mstn_trainmodel_noTL(TRAINING_FILE, VAL_FILE, val_file_num=100, epochs_limit=2500, step_log=True, model_name='mstn'):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir):
        os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir):
        os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir):
        os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    x = tf.placeholder(tf.float32, [None, 227, 227, 3], 'x')
    xt = tf.placeholder(tf.float32, [None, 227, 227, 3], 'xt')

    y = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'y')
    yt = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'yt')

    adlamb = tf.placeholder(tf.float32)
    decay_learning_rate = tf.placeholder(tf.float32)
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = AlexNetModel_noTL(num_classes=NUM_CLASSES,
                              dropout_keep_prob=dropout_keep_prob)
    closs = model.closs(
        batch_x_source=x,
        batch_y_source=y
    )

    feature = model.feature
    result = model.result

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    correct_test = 100 * tf.argmax(model.score, 1) + tf.argmax(y, 1)
    correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    dis_positive, dis_negative, sc, tc, smoothness = model.adloss(
        x, xt, y, adlamb)

    target_correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(yt, 1))
    target_correct = tf.reduce_sum(tf.cast(target_correct_pred, tf.float32))
    target_accuracy = tf.reduce_mean(tf.cast(target_correct_pred, tf.float32))

    # optimizer
    train_op = model.optimize(
        decay_learning_rate,
        train_layers,
        adlamb,
        dis_positive,
        dis_negative,
        sc,
        tc,
        smoothness
    )
    D_op = model.adoptimize(decay_learning_rate, train_layers)

    optimizer = tf.group(train_op, D_op)

    train_writer = tf.summary.FileWriter(
        './MSTN_MODEL/MSTN_train_log/tensorboard'+MODEL_NAME)
    train_writer.add_graph(tf.get_default_graph())
    tf.summary.scalar('Testing Accuracy', target_accuracy)
    merged = tf.summary.merge_all()

    print('============================GLOBAL TRAINABLE VARIABLES ============================')
    print(tf.trainable_variables())

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None
    print('==================== MULTI SCALE===================================================')
    print(multi_scale)
    train_preprocessor_source = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                                  output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    train_preprocessor_target = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES,
                                                  output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor_target = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES,
                                                output_size=[227, 227], multi_scale=multi_scale, istraining=False)
    val_preprocessor_source = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                                output_size=[227, 227], multi_scale=multi_scale, istraining=False)
    print('Initialize done.')

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch_source = np.floor(
        len(train_preprocessor_source.labels) / FLAGS.batch_size).astype(np.int16)
    train_batches_per_epoch_target = np.floor(
        len(train_preprocessor_target.labels) / FLAGS.batch_size).astype(np.int16)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_writer.add_graph(sess.graph)
        # Load the pretrained weights
        model.load_original_weights(sess, skip_layers=train_layers)

        print(("{} Start training...".format(datetime.datetime.now())))
        print(("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir)))
        gs = 0
        gd = 0
        exit_flag = False
        for epoch in range(FLAGS.num_epochs):
            if exit_flag:
                break

            print("Epoch number: {}".format(epoch+1))
            step = 1
            # Start training
            while step < train_batches_per_epoch_source:
                gd += 1
                print(gd)
                lamb = adaptation_factor(gd*1.0/MAX_STEP)
                rate = decay(FLAGS.learning_rate, gd, MAX_STEP)

                for it in range(1):
                    gs += 1
                    if gs % train_batches_per_epoch_target == 0:
                        train_preprocessor_target.reset_pointer()
                    if gs % train_batches_per_epoch_source == 0:
                        train_preprocessor_source.reset_pointer()

                    batch_x_source, batch_y_source = train_preprocessor_source.next_batch(
                        FLAGS.batch_size)
                    batch_x_target, batch_y_target = train_preprocessor_target.next_batch(
                        FLAGS.batch_size)

                    summary, _ = sess.run(
                        [merged, optimizer],
                        feed_dict={
                            x: batch_x_source,
                            xt: batch_x_target,
                            yt: batch_y_target,
                            adlamb: lamb,
                            decay_learning_rate: rate,
                            y: batch_y_source,
                            dropout_keep_prob: 0.5
                        }
                    )
                    train_writer.add_summary(summary, gd)

                closs, gloss, dloss, gregloss, dregloss, floss, totaldis, semloss, smoothness = sess.run(
                    [
                        model.loss,
                        model.G_loss,
                        model.D_loss,
                        model.Gregloss,
                        model.Dregloss,
                        model.F_loss,
                        model.total_distance,
                        model.Semanticloss,
                        model.smoothness
                    ],
                    feed_dict={
                        x: batch_x_source,
                        xt: batch_x_target,
                        adlamb: lamb,
                        decay_learning_rate: rate,
                        y: batch_y_source,
                        dropout_keep_prob: 0.5
                    }
                )

                step += 1

                if gd % 20 == 0 and step_log == True:
                    print(
                        '=================== Step {} ================='.format(gs))
                    print('Epoch {} Step {} \nCloss {} \nGloss {} \nDloss {} \nTotal_Loss {} \nGregloss {} \nDregloss {} \ntotal distance {}\nsemloss {}'.format(
                        epoch, step, closs, gloss, dloss, floss, gregloss, dregloss, totaldis, semloss))
                    print('lambda: ', lamb)
                    print('rate: ', rate)

                    # Epoch completed, start validation
                    print(("{} Start validation".format(datetime.datetime.now())))

                    # Draw feature space and calculate score
                    _result_total = np.zeros([val_file_num], dtype=np.int32)
                    batch_x_target, batch_y_target = val_preprocessor_target.next_batch(
                        val_file_num)
                    batch_x_source, batch_y_source = val_preprocessor_source.next_batch(
                        200)

                    _result_graph_sub, _result_graph_total, _feature_source, _feature_target, _result_source, _SS_s = sess.run(
                        [
                            model.result_graph,
                            model.target_result_new,
                            model.feature_source,
                            model.feature_target,
                            model.result_source,
                            model.SemanticScore_source,

                        ],
                        feed_dict={
                            x: batch_x_source,
                            xt: batch_x_target,
                            y: batch_y_source,
                            dropout_keep_prob: 1.0
                        }
                    )

                    print(_result_graph_sub, _result_graph_total)
                    print(_SS_s)

                    draw_feature_graph(_feature_source, _feature_target, _result_source, _result_graph_sub,
                                       _result_graph_total, './MSTN_MODEL/MSTN_train_log/' + train_dir_name, gd)

                    # Reset the dataset pointers
                    val_preprocessor_target.reset_pointer()
                    val_preprocessor_source.reset_pointer()

                if gd % 500 == 0 and gd > 0:
                    saver.save(sess, './MSTN_MODEL/MSTN_train_log/' + train_dir_name + '/' +
                               model_name+str(gd)+'.ckpt')
                    print(("{} Saving checkpoint of model...".format(
                        datetime.datetime.now())))

                if gd >= epochs_limit:
                    exit_flag = True

                    # Draw feature space and calculate score
                    _result_total = np.zeros([val_file_num], dtype=np.int32)
                    batch_x_target, batch_y_target = val_preprocessor_target.next_batch(
                        val_file_num)
                    batch_x_source, batch_y_source = val_preprocessor_source.next_batch(
                        200)

                    _result_graph_sub, _result_graph_total, _feature_source, _feature_target, _result_source, _SS_s = sess.run(
                        [
                            model.result_graph,
                            model.target_result_new,
                            model.feature_source,
                            model.feature_target,
                            model.result_source,
                            model.SemanticScore_source
                        ],
                        feed_dict={
                            x: batch_x_source,
                            xt: batch_x_target,
                            y: batch_y_source,
                            dropout_keep_prob: 1.0
                        }
                    )

                    print(_result_graph_sub, _result_graph_total)
                    print(_SS_s)

                    draw_feature_graph(_feature_source, _feature_target, _result_source, _result_graph_sub,
                                       _result_graph_total, './MSTN_MODEL/MSTN_train_log/' + train_dir_name, gd)

                    break

        last_model_path = './MSTN_MODEL/trained_models/' + \
            model_name + str(gd)+'.ckpt'
        saver.save(sess, last_model_path)
        print(("{} Saving checkpoint of model...".format(datetime.datetime.now())))

    return last_model_path, _result_graph_total, [_SS_s]


def mstn_label_with_model_noTL(MODEL_PATH, TRAINING_FILE, VAL_FILE, val_file_num=100, model_name='mstn'):
    """
    使用训练好的mstn模型给hard样本打标签

    返回值：
    [labellist_negetive, labellist_positive]:
        labellist_negetive：标记为0的图片引索
        labellist_positive：标记为1的图片引索
    feature_total:
        所有hard样本特征向量组成的array

    """
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir):
        os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir):
        os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir):
        os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    x = tf.placeholder(tf.float32, [None, 227, 227, 3], 'x')
    xt = tf.placeholder(tf.float32, [None, 227, 227, 3], 'xt')

    y = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'y')
    yt = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'yt')

    adlamb = tf.placeholder(tf.float32)
    decay_learning_rate = tf.placeholder(tf.float32)
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = AlexNetModel_noTL(num_classes=NUM_CLASSES,
                              dropout_keep_prob=dropout_keep_prob)
    closs = model.closs(
        batch_x_source=x,
        batch_y_source=y
    )

    feature = model.feature
    result = model.result

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    correct_test = 100 * tf.argmax(model.score, 1) + tf.argmax(y, 1)
    correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    dis_positive, dis_negative, sc, tc, smoothness = model.adloss(
        x, xt, y, adlamb)

    target_correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(yt, 1))
    target_correct = tf.reduce_sum(tf.cast(target_correct_pred, tf.float32))
    target_accuracy = tf.reduce_mean(tf.cast(target_correct_pred, tf.float32))

    # optimizer
    train_op = model.optimize(
        decay_learning_rate,
        train_layers,
        adlamb,
        dis_positive,
        dis_negative,
        sc,
        tc,
        smoothness
    )
    D_op = model.adoptimize(decay_learning_rate, train_layers)

    optimizer = tf.group(train_op, D_op)

    train_writer = tf.summary.FileWriter(
        './MSTN_MODEL/MSTN_train_log/tensorboard'+MODEL_NAME)
    train_writer.add_graph(tf.get_default_graph())
    tf.summary.scalar('Testing Accuracy', target_accuracy)
    merged = tf.summary.merge_all()

    #print('============================GLOBAL TRAINABLE VARIABLES ============================')
    # print(tf.trainable_variables())

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None
    #print('==================== MULTI SCALE===================================================')
    # print(multi_scale)
    train_preprocessor_source = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                                  output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    train_preprocessor_target = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES,
                                                  output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor_target = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES, output_size=[
        227, 227], multi_scale=multi_scale, istraining=False)
    val_preprocessor_source = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                                output_size=[227, 227], multi_scale=multi_scale, istraining=False)
    print('Initialize done.')

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch_source = np.floor(
        len(train_preprocessor_source.labels) / FLAGS.batch_size).astype(np.int16)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_writer.add_graph(sess.graph)
        # Load the pretrained weights
        #model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, MODEL_PATH)

        #print(("{} Start training...".format(datetime.datetime.now())))
        #print(("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir)))
        gs = 0
        gd = 0
        labellist_positive = []
        labellist_negetive = []
        for epoch in range(FLAGS.num_epochs):
            #print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1
            # Start training
            while step < train_batches_per_epoch_source:
                gd += 1
                lamb = adaptation_factor(gd*1.0/MAX_STEP)
                rate = decay(FLAGS.learning_rate, gd, MAX_STEP)
                if gd % 1 == 0:
                    print(("{} Start validation".format(datetime.datetime.now())))
                    batch_max = BATCH_MAX

                    result_graph_total = []
                    SS_s = []

                    for batch_start_num in range(1, val_file_num, batch_max):
                        _result_total = np.zeros(
                            [val_file_num], dtype=np.int32)

                        if val_file_num - batch_start_num < batch_max - 1:
                            batch_num_current = val_file_num - batch_start_num + 1
                        else:
                            batch_num_current = batch_max

                        batch_x_target, batch_y_target = val_preprocessor_target.next_batch(
                            batch_num_current)
                        batch_x_source, batch_y_source = val_preprocessor_source.next_batch(
                            200)

                        _result_graph_sub, _result_graph_total, _feature_source, _feature_target, _result_source, _SS_s = sess.run(
                            [
                                model.result_graph,
                                model.target_result_new,
                                model.feature_source,
                                model.feature_target,
                                model.result_source,
                                model.SemanticScore_source
                            ],
                            feed_dict={
                                x: batch_x_source,
                                xt: batch_x_target,
                                y: batch_y_source,
                                dropout_keep_prob: 1.0
                            }
                        )

                        #print(_result_graph_sub, _result_graph_total)
                        #print(_SS_s, _SS_t)

                        #draw_feature_graph(_feature_source, _feature_target, _result_source, _result_graph_sub, _result_graph_total, './MSTN_MODEL/MSTN_train_log/' + train_dir_name , gd, show=True)

                        result_graph_total.append(_result_graph_total)
                        SS_s.append(_SS_s)

                        # feature_source, feature_target, result_source, result_target_total, SS, savedir, logname, SS_limit
                        log_feature_low(
                            _feature_source, 
                            _feature_target, 
                            _result_source, 
                            _result_graph_total,
                            SS=SS_s,
                            savedir='./MSTN_MODEL/MSTN_train_log/' + train_dir_name + model_name + '/',
                            logname=str(batch_start_num),
                            use_SS=False
                        )

                        print("Validation %.2f %% ..." %
                              (100*val_preprocessor_target.pointer/val_file_num))

                        # Reset the dataset pointers
                        # val_preprocessor_target.reset_pointer()
                        val_preprocessor_source.reset_pointer()

                    print("Validation done.")
                    result_graph_total = np.array(
                        [i for item in result_graph_total for i in item])
                    SS_s = np.array([i for item in SS_s for i in item])

                    return result_graph_total, [SS_s]


def main(_):
    pass


if __name__ == '__main__':
    tf.app.run()
