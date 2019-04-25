# coding=utf-8
from helpmethods import *
import sys


def main():
    yolo_dir = './YOLO_MODEL/'
    mstn_dir = './MSTN_MODEL/'

    yolo_test_dir = "./YOLO_MODEL/yolo_test/"  # 测试换图片集更改此处
    yolo_train_dir = "./YOLO_MODEL/yolo_train/"
    yolo_result_dir = "./YOLO_MODEL/yolo_result/"

    yolo_model_cfg = "./YOLO_MODEL/cfg/yolo.cfg"
    yolo_model_original = "./YOLO_MODEL/bin/yolo.weights"

    mstn_result_dir = "./MSTN_MODEL/MSTN_result/"
    mstn_train_img_dir = "./MSTN_MODEL/MSTN_train_images/"
    mstn_source_train_label = "./MSTN_MODEL/MSTN_train_images/source.txt"
    mstn_target_train_label = "./MSTN_MODEL/MSTN_train_images/target_with_label.txt"
    mstn_target_test_file = yolo_result_dir + "picture_labels.txt"

    train_yolo_picture_number = 351
    test_yolo_picture_number = 351
    test_pictuce_start_number = 0

    clow = 0.08
    chigh = 0.6

    train_yolo_init(
        yolo_train_dir,
        yolo_result_dir,
        mstn_result_dir,
        model=yolo_model_cfg,
        weights=yolo_model_original,                        # 训练yolo的基础权重文件
        annotation=yolo_train_dir + "annotation",
        dataset=yolo_train_dir + "image",
        learningrate="0.00001",
        batchsize="1",
        epoch="100",
        save="500",
        train_yolo=False,               # 选择是否运行此函数
        DO_NOT_DEL_TRAIN_PIC=True,      # 选择是否在训练yolo结束后清空yolo训练集 改为False时请谨慎
    )

    label_with_YOLO(
        yolo_test_dir,
        test_yolo_picture_number,
        yolo_result_dir,
        clow,
        chigh,
        model=yolo_model_cfg,
        load=yolo_model_original,                        # 检测权重
        # pb="./YOLO_MODEL/built_graph/yolo_test2.pb",      # 可选择直接使用pb文件，不能和model, weights同时赋值
        # meta="./YOLO_MODEL/built_graph/yolo_test2.meta",  # 可选择直接使用meta文件，不能和model, weights同时赋值
        use_gpu=False,                                       # 选择是否使用gpu加速检测
        start_number=test_pictuce_start_number,             # 待检测图片的起始编号（文件名需要0填充共6位）
        save_picture_with_box=True,                         # 选择是否保存检测结果的完整图片到yolo_result_dir
        label_image=True,                            # 选择是否运行此函数
        already_labeled=False
    )

    MSTN_train_set_init(
        yolo_result_dir=yolo_result_dir,
        yolo_test_dir=yolo_test_dir,
        MSTN_train_img_dir=mstn_train_img_dir,
        pic_num_for_train_MSTN=150, #test_yolo_picture_number,    # 背景建模使用的视频帧数量
        positive_score_limit=0.2,                           # 背景建模中的检测结果得分阈值
        background_modeling=False                           # 选择是否运行此函数
    )

    label_hard_pic_with_MSTN(
        yolo_dir,
        mstn_dir,
        mstn_source_train_label,
        mstn_target_test_file,
        mstn_target_train_label,
        SS_limit=0.3,
        mstn_train=False,               # 选择是否训练困难样本分类器，若为False则直接使用/MSTN_MODEL/trained_models/中的现有权重
        mstn_test=True,
        step_log=True,                  # 选择是否计算每20步训练的结果
        add_to_trainset=True,          # 选择是否将分类结果制作为yolo训练图片
        model_name="cuhk352",              # 训练/使用的模型名称
        train_epoch=500,                  # 训练迭代次数（四个数据集均在500左右较为合适）
        label_hard_image=True           # 选择是否运行此函数
    )



if __name__ == "__main__":
    sys.exit(main())
