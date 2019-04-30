# coding=utf-8
from helpmethods import *
import numpy as np
import sys
import os


def main():
    yolo_test_dir = "./YOLO_MODEL/yolo_test/"  # 测试换图片集更改此处
    yolo_train_dir_original = "./YOLO_MODEL/yolo_train/"
    yolo_result_dir_base = "./YOLO_MODEL/yolo_result/"

    yolo_model_cfg = "./YOLO_MODEL/cfg/yolo.cfg"
    yolo_model_original = "./YOLO_MODEL/bin/yolo.weights"

    mstn_result_dir_base = "./MSTN_MODEL/MSTN_result/"
    mstn_train_img_dir = "./MSTN_MODEL/MSTN_train_images/"
    mstn_source_train_label = "./MSTN_MODEL/MSTN_train_images/source.txt"
    mstn_target_train_label = "./MSTN_MODEL/MSTN_train_images/target_with_label.txt"

    # ------------------config training parameters------------------
    #train_yolo_picture_number = 436
    test_yolo_picture_number = 436
    test_pictuce_start_number = 0

    beta = 0.0
    theta0 = 0.8
    run_time = 5
    mstn_train_step_range = [500, 1500]

    train_yolo = np.ones([run_time], dtype=np.int)
    label_with_yolo = np.ones([run_time], dtype=np.int)
    label_hard = np.ones([run_time], dtype=np.int)
    gpu_use = 0.0
    model_name = 'crowdNEW-0429'

    #train_yolo[0:4] = 0
    #label_with_yolo[1:4] = 0
    #label_hard[0:3] = 0

    # --------------------------------------------------------------
    #theta = 0.66853
    #theta = 0.5010323868913487
    #theta = 1.473481240236558243e-01
    for i in range(0, run_time):
        mstntrain = 1

        if i == 0:
            yolomodel = yolo_model_original
            yolomodel_prev = yolo_model_original
        else:
            yolomodel_prev = yolomodel
            yolomodel = -1

        if i == 0:
            theta = theta0

        if not run_time == 1:
            mstn_train_step = mstn_train_step_range[1] - i * (
                mstn_train_step_range[1] - mstn_train_step_range[0])/(run_time-1)
        else:
            mstn_train_step = mstn_train_step_range[1]

        clow = 0.2
        chigh = beta + theta

        yolo_result_dir = yolo_result_dir_base + model_name + str(i) + '/'
        yolo_train_dir = yolo_train_dir_original + model_name + str(i) + '/'
        yolo_train_dir_prev = yolo_train_dir_original + \
            model_name + str(i-1) + '/'
        mstn_result_dir = mstn_result_dir_base + model_name + str(i) + '/'
        mstn_target_test_file = yolo_result_dir + "picture_labels.txt"

        dir_init(yolo_train_dir)
        dir_init(yolo_result_dir)
        dir_init(mstn_result_dir_base)
        dir_init(mstn_result_dir)
        dir_init(mstn_train_img_dir)

        train_yolo_init(
            model=yolo_model_cfg,
            weights=str(yolomodel_prev),                        # 训练yolo的基础权重文件
            annotation=yolo_train_dir_prev + "annotation",
            dataset=yolo_train_dir_prev + "image",
            learningrate="0.000005",
            batchsize="1",
            epoch="3",
            save="250",
            train_yolo=train_yolo[i]*i,
            gpu_use=gpu_use,
            DO_NOT_DEL_TRAIN_PIC=True,      # 选择是否在训练yolo结束后清空yolo训练集 改为False时请谨慎
        )

        label_with_YOLO(
            yolo_test_dir,
            test_yolo_picture_number,
            yolo_result_dir,
            clow,
            chigh,
            model=yolo_model_cfg,
            load=yolomodel,                        # 检测权重
            use_gpu=gpu_use,                                       # 选择是否使用gpu加速检测
            start_number=test_pictuce_start_number,             # 待检测图片的起始编号（文件名需要0填充共6位）
            # 选择是否保存检测结果的完整图片到yolo_result_dir
            save_picture_with_box=True,
            label_image=label_with_yolo[i],
            already_labeled=False
        )

        MSTN_train_set_init(
            yolo_result_dir=yolo_result_dir,
            yolo_test_dir=yolo_test_dir,
            MSTN_train_img_dir=mstn_train_img_dir,
            pic_num_for_train_MSTN=test_yolo_picture_number,    # 背景建模使用的视频帧数量
            positive_score_limit=0.2,                           # 背景建模中的检测结果得分阈值
            background_modeling=label_hard[i]*0
        )

        theta = label_hard_pic_with_MSTN(
            theta,
            beta,
            yolo_train_dir,
            yolo_test_dir,
            yolo_result_dir,
            mstn_result_dir,
            mstn_train_img_dir,
            mstn_source_train_label,
            mstn_target_test_file,
            mstn_target_train_label,
            SS_limit=0.5,
            # 选择是否训练困难样本分类器，若为False则直接使用/MSTN_MODEL/trained_models/中的现有权重
            mstn_train=True*mstntrain,
            mstn_test=True*mstntrain,
            step_log=False,                  # 选择是否计算每20步训练的结果
            add_to_trainset=True,          # 选择是否将分类结果制作为yolo训练图片
            model_name=model_name+'_yolo'+str(int(i)),          # 训练/使用的模型名称
            train_epoch=mstn_train_step,                  # 训练迭代次数（四个数据集均在500左右较为合适）
            label_hard_image=label_hard[i]
        )

        t = np.ones([1], dtype=np.float)
        np.savetxt('./theta'+str(i)+'.txt', t*theta)


def dir_init(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    sys.exit(main())
