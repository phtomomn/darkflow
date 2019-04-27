# coding=utf-8
import datetime
import os
import shutil
import sys
import xml.dom.minidom
from xml.dom.minidom import Document

import numpy as np
from PIL import Image

import AutoBackground.gt as autoback
import cv2
from darkflow.cli import cliHandler
from darkflow.net.build import TFNet
from MSTN_MODEL.MSTN_train import mstn_label_with_model, mstn_trainmodel
from MSTN_MODEL.MSTN_train import mstn_label_with_model_noTL, mstn_trainmodel_noTL


def test_with_yolo(
        yolomodel, 
        yolo_test_dir,
        yolo_result_dir, 
        picture_number, 
        confidence_limit=0.5, 
        start_number=0
    ):
    """
    使用yolo模型测试指定数量的图片
    图片名称要求：0填充6位
    结果保存：
        predict_box：
            第一维：图片引索
            第二维：与原图同尺寸的box分布
        predict_box_list：
            第一维为图片引索
            第二维为的box位置与置信度
    """
    predict_result = []
    
    for i in range(start_number, start_number + picture_number):
        print('detecting {}/{}...'.format(str(i-start_number+1), str(picture_number)))
        imgcv = cv2.imread(yolo_test_dir + "image/" + str(i).zfill(6) + ".jpg")
        img_shape = imgcv.shape[0:2]
        result = yolomodel.return_predict(imgcv)
        person_cnt = 0
        box_number = len(result)
        for j in range(box_number):
            current_box = result[j]
            if current_box['label'] == 'person':
                topleft = current_box['topleft']
                bottomright = current_box['bottomright']
                confidence = current_box['confidence']

                if bottomright['x'] - topleft['x'] >= 0.4*img_shape[1] or bottomright['y'] - topleft['y'] >= 0.4*img_shape[0]:
                    continue
                
                if confidence >= confidence_limit:
                    predict_result.append([i, j, topleft['x'], topleft['y'], bottomright['x'], bottomright['y'], confidence])
                    person_cnt += 1
        
        print('\tDone. {} boxes found.'.format(str(person_cnt)))

    np.savetxt(yolo_result_dir+'predict_result.txt', np.array(predict_result))



def save_predict_picture_with_box(
        img_dir,
        result_file, 
        yolo_predict_whole_pic_dir, 
        picture_number,  
        confidence_limit=0.5, 
        start_number=0,
        show_confidence=True
    ):
    """
    根据yolo预测结果在原图上加box，并保存
    """
    if not os.path.exists(yolo_predict_whole_pic_dir):
        os.makedirs(yolo_predict_whole_pic_dir)

    predict_result = np.loadtxt(result_file)

    for i in range(start_number, start_number + picture_number):
        current_predict = predict_result[np.where(predict_result.T[0]==i)]
        box_number = current_predict.shape[0]

        img = cv2.imread(img_dir +str(i).zfill(6) + ".jpg")
        for j in range(box_number):
            current_box_and_confidence = current_predict[j]
            current_box = current_box_and_confidence[2:6].astype(np.int32)

            if show_confidence:
                confidence = current_box_and_confidence[6]
                if confidence >= confidence_limit:
                    cv2.rectangle(
                        img, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 255, 0), 8)
                    cv2.putText(img, str(round(confidence, 2)), (current_box[0], max(
                        current_box[1]-10, 0)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 5)
                    cv2.putText(img, str(round(confidence, 2)), (current_box[0], max(
                        current_box[1]-10, 0)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
                else:
                    cv2.rectangle(
                        img, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (255, 224, 18), 5)
                    cv2.putText(img, str(round(confidence, 2)), (current_box[0], max(
                        current_box[1]-10, 0)), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 5)
                    cv2.putText(img, str(round(confidence, 2)), (current_box[0], max(
                        current_box[1]-10, 0)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
            
            else:
                cv2.rectangle(img, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 255, 0), 8)

        cv2.imwrite(yolo_predict_whole_pic_dir + str(i).zfill(6) + ".jpg", img)
        print('saving whole image {}/{}...'.format(str(i-start_number+1), str(picture_number)))

    print("{} Pictures with boxes are saved at {}".format(datetime.datetime.now(), yolo_predict_whole_pic_dir))



def save_class_predict_box_sub_picture(
        yolo_test_dir, 
        yolo_result_dir, 
        picture_number, 
        start_number = 0, 
        low_limit=0.1, 
        high_limit=0.5, 
        save=True
    ):
    """
    根据yolo预测结果与两个阈值，取出hard样本对应box，并另存为新图片。
    并将预测结果中的图片数量写入picture_numbers.txt，同时保存适合于mstn模型训练的标签picture_labels.txt
    返回值：
        predict_box_hard_list：
            hard样本的predict_box_list.
            第一维：图片引索
            第二纬：hard样本的box位置与置信度
    """
    if save:
        if not os.path.exists(yolo_result_dir):
            os.makedirs(yolo_result_dir)

        if not os.path.exists(yolo_result_dir + "hard/"):
            os.makedirs(yolo_result_dir + "hard/")

        if not os.path.exists(yolo_result_dir + "positive/"):
            os.makedirs(yolo_result_dir + "positive/")

        if not os.path.exists(yolo_result_dir + "negative/"):
            os.makedirs(yolo_result_dir + "negative/")

    predict_box_hard_list = []
    file_name_list_hard = []
    file_name_list_pos = []

    box_count_n = 0
    box_count_p = 0
    box_count_h = 0

    cnt = 0

    for _ in range(start_number + picture_number):
        predict_box_hard_list.append([])

    predict_result = np.loadtxt(yolo_result_dir+'predict_result.txt')
    for i in range(start_number, start_number + picture_number):
        current_predict = predict_result[np.where(predict_result.T[0]==i)]
        box_number = current_predict.shape[0]

        with Image.open(yolo_test_dir + "image/" + str(i).zfill(6) + ".jpg") as img:
            for j in range(box_number):
                print('saving box {}/{} in image {}/{}'.format(str(j+1), str(box_number+1), str(start_number-i+1), str(picture_number)))
                current_box_and_confidence = current_predict[j]
                current_box = current_box_and_confidence[2:6]
                confidence = current_box_and_confidence[6]

                if current_box[0] == current_box[2] or current_box[1] == current_box[3]:
                    continue

                if confidence < low_limit:
                    if save:
                        img_temp = img.crop(current_box)
                        img_temp.save(yolo_result_dir +
                                    "negative/" + "n" + str(box_count_n) + ".jpg")
                    box_count_n += 1

                if confidence >= low_limit and confidence <= high_limit:
                    predict_box_hard_list[i].append(current_box_and_confidence)
                    if save:
                        img_temp = img.crop(current_box)
                        img_temp.save(yolo_result_dir + "hard/" + "h" +
                                    str(box_count_h) + ".jpg")

                    file_name_list_hard.append(yolo_result_dir + "hard/" + "h" +
                                    str(box_count_h) + ".jpg "+ str(i).zfill(6)+'.jpg' + '\n')
                    box_count_h += 1

                if confidence > high_limit:
                    if save:
                        img_temp = img.crop(current_box)
                        img_temp.save(yolo_result_dir + "positive/" + "p" +
                                    str(box_count_p) + ".jpg")
                   
                    file_name_list_pos.append(yolo_result_dir + "positive/" + "p" +
                                    str(box_count_p) + ".jpg "+ str(i).zfill(6)+'.jpg'+'\n')
                    box_count_p += 1
            
                cnt += 1

    with open(yolo_result_dir + "picture_numbers.txt", 'w') as f:
        f.write(str(box_count_n)+" "+str(box_count_h) +
                " "+str(box_count_p)+"\n")
        f.write(str(low_limit)+"\n")
        f.write(str(high_limit)+"\n")
        print("{} Number of neg, hard, pos pictures are saved at {}.".format(datetime.datetime.now(), yolo_result_dir + "picture_numbers.txt"))

    with open(yolo_result_dir + "picture_labels.txt", 'w') as f:
        for i in range(box_count_h):
            f.write(yolo_result_dir + "hard/" +
                    "h" + str(i) + ".jpg" + " 1\n")
    
    with open(yolo_result_dir + "original_path_hard.txt", 'w') as f:
        f.writelines(file_name_list_hard)

    with open(yolo_result_dir + "original_path_pos.txt", 'w') as f:
        f.writelines(file_name_list_pos)

    print("{} {} boxes are detected from {} test pictures. They are labeled into:\n\t{} pos samples\n\t{} neg samples\n\t{} hard samples".format(
        datetime.datetime.now(), str(box_count_h+box_count_n+box_count_p), str(picture_number), str(box_count_p), str(box_count_n), str(box_count_h)))
    print("and saved into {}.".format(yolo_result_dir))
    return predict_box_hard_list


def save_labeled_hard_predict_box_sub_picture(yolo_result_dir, mstn_result_dir, hard_predict_box_label):
    """
    使用hard样本经过mstn模型后得到的label，分类保存正负样本图片，并保存+-样本的数量至hard_label_num.txt
    """
    if not os.path.exists(mstn_result_dir):
        os.makedirs(mstn_result_dir)

    if not os.path.exists(mstn_result_dir + "hard-positive/"):
        os.makedirs(mstn_result_dir + "hard-positive/")

    if not os.path.exists(mstn_result_dir + "hard-negative/"):
        os.makedirs(mstn_result_dir + "hard-negative/")

    pos_picture = hard_predict_box_label[1]
    neg_picture = hard_predict_box_label[0]

    for i in range(len(pos_picture)):
        source_path = yolo_result_dir + \
            "hard/" + "h" + str(pos_picture[i]) + ".jpg"
        target_path = mstn_result_dir + \
            "hard-positive/" + "hp" + str(i) + ".jpg"
        shutil.copyfile(source_path, target_path)

    for i in range(len(neg_picture)):
        source_path = yolo_result_dir + \
            "hard/" + "h" + str(neg_picture[i]) + ".jpg"
        target_path = mstn_result_dir + \
            "hard-negative/" + "hn" + str(i) + ".jpg"
        shutil.copyfile(source_path, target_path)

    with open(mstn_result_dir + "hard_label_num.txt", 'w') as f:
        f.write(str(len(neg_picture))+" "+str(len(pos_picture)))

    print("{} All {} hard samples are labeled into:\n\t{} pos samples\n\t{} neg samples".format(
        datetime.datetime.now(), str(len(pos_picture)+len(neg_picture)), str(len(pos_picture)), str(len(neg_picture))))
    print("and saved into {}.".format(mstn_result_dir + "hard-*/"))
    print("{} Classed picture numbers are saved at {}.".format(datetime.datetime.now(), mstn_result_dir + "hard_label_num.txt"))




def train_yolo_model(
        filename="predict.py", 
        model="cfg/yolo_test2.cfg", 
        weights="bin/yolov2-tiny-voc.weights", 
        annotation="train/annotation", 
        dataset="train/image", 
        learningrate="0.001", 
        config="./YOLO_MODEL/cfg/",
        batchsize="64", 
        epoch="100", 
        save="200",
        pb='null',
        meta='null'
    ):

    if pb == 'null':
        config = filename \
            + " --model " + model \
            + " --load " + weights \
            + " --train" \
            + " --annotation " + annotation \
            + " --dataset " + dataset \
            + " --lr " + learningrate \
            + " --gpu 0.0" \
            + " --batch " + batchsize \
            + " --epoch " + epoch \
            + " --save " + save \
            + " --savepb" \
            + " --config " + config \
            + " --gpu 1.0"
    else:
        config = filename \
            + " --pbLoad " + pb \
            + " --metaLoad " + meta \
            + " --train" \
            + " --annotation " + annotation \
            + " --dataset " + dataset \
            + " --lr " + learningrate \
            + " --gpu 1.0" \
            + " --batch " + batchsize \
            + " --epoch " + epoch \
            + " --save " + save \
            + " --savepb" \
            + " --gpu 1.0"
    argv = config.split(" ")
    cliHandler(argv)


def train_yolo_init(
        yolo_train_dir, 
        yolo_result_dir, 
        mstn_result_dir, 
        DO_NOT_DEL_TRAIN_PIC=True, 
        train_yolo=False, 
        filename="predict.py", 
        model="cfg/yolo_test2.cfg", 
        weights="bin/yolov2-tiny-voc.weights", 
        annotation="train/annotation", 
        dataset="train/image", 
        learningrate="0.001", 
        batchsize="8", 
        epoch="100", 
        save="200"
    ):
    """
    初始化训练文件夹，并按照指定选项训练yolo
    """

    """
    if os.path.exists(yolo_result_dir) and not DO_NOT_DEL_TRAIN_PIC:
        shutil.rmtree(yolo_result_dir)
        os.mkdir(yolo_result_dir)

    if os.path.exists(mstn_result_dir) and not DO_NOT_DEL_TRAIN_PIC:
        shutil.rmtree(mstn_result_dir)
        os.mkdir(mstn_result_dir)
    """


    if train_yolo:
        train_yolo_model(
            filename=filename, 
            model=model,
            weights=weights, 
            annotation=annotation, 
            dataset=dataset, 
            learningrate=learningrate, 
            batchsize=batchsize, 
            epoch=epoch, 
            save=save
        )



def save_yolo_predict_result(yolo_result_dir, low_limit=0.1, high_limit=0.5):
    predict_box_array = []
    predict_box_array_pos = []
    predict_box_array_hard = []

    predict_result = np.loadtxt(yolo_result_dir+'predict_result.txt')
    predict_pic_num = (np.max(predict_result.T[0]) - np.min(predict_result.T[0]) + 1).astype(np.int32)

    for pic in range(predict_pic_num):
        current_predict = predict_result[np.where(predict_result.T[0]==pic)]
        box_number = current_predict.shape[0]

        if box_number == 0:
            continue

        for box in range(box_number):
            predict_box_array.append([box, pic, current_predict[box][2], current_predict[box][3], current_predict[box][4], current_predict[box][5], current_predict[box][6]])
            if current_predict[box][6] >= low_limit and current_predict[box][6] <= high_limit:
                predict_box_array_hard.append([box, pic, current_predict[box][2], current_predict[box][3], current_predict[box][4], current_predict[box][5], current_predict[box][6]])
            
            if current_predict[box][6] >= high_limit:
                predict_box_array_pos.append([box, pic, current_predict[box][2], current_predict[box][3], current_predict[box][4], current_predict[box][5], current_predict[box][6]])
            
    np.savetxt(yolo_result_dir + "result.txt", np.array(predict_box_array))
    np.savetxt(yolo_result_dir + "position_hard.txt", np.array(predict_box_array_hard))
    np.savetxt(yolo_result_dir + "position_pos.txt", np.array(predict_box_array_pos))


def label_with_YOLO(
        yolo_test_dir, 
        picture_number, 
        yolo_result_dir, 
        confidence_limit_low, 
        confidence_limit_high, 
        save_picture_with_box=False, 
        start_number=0,
        pb="./YOLO_MODEL/built_graph/yolo_test2.pb",
        meta="./YOLO_MODEL/built_graph/yolo_test2.meta",
        model='null',
        load='null',
        use_gpu=False,
        label_image=True,
        yolo_limit=0.08,
        already_labeled=False
    ):
    """
    使用训练好的yolo模型分类测试图片为正、负、hard三类，并将测试结果box储存为新图片
    返回值：
        IoU：每张测试图片的IoU
    """
    if not label_image:
        return

    if already_labeled == False:
        options = {
            "config": "./YOLO_MODEL/cfg/",
            "threshold": yolo_limit
        }

        if model == 'null':
            options.update({"pbLoad": pb, "metaLoad": meta})
        else:
            options.update({"model": model, "load": load})

        if use_gpu > 0:
            options.update({"gpu":use_gpu})

        tfnet = TFNet(options)
        test_with_yolo(
            tfnet, 
            yolo_test_dir, 
            yolo_result_dir,
            picture_number, 
            start_number=start_number, 
            confidence_limit=confidence_limit_low
        )

    save_class_predict_box_sub_picture(
        yolo_test_dir, 
        yolo_result_dir, 
        picture_number, 
        start_number = start_number,
        low_limit=confidence_limit_low, 
        high_limit=confidence_limit_high
    )

    save_yolo_predict_result(
        yolo_result_dir, 
        low_limit=confidence_limit_low, 
        high_limit=confidence_limit_high
    )

    if save_picture_with_box:
        save_predict_picture_with_box(
            yolo_test_dir + "image/",
            yolo_result_dir + 'predict_result.txt', 
            yolo_result_dir + "whole_pic/", 
            picture_number, 
            start_number=start_number, 
            confidence_limit=confidence_limit_high
        )
    

def MSTN_train_set_init(yolo_result_dir, yolo_test_dir, pic_num_for_train_MSTN, MSTN_train_img_dir, positive_score_limit=0.5, background_modeling=True):
    if not background_modeling:
        return

    gt = autoback.ReadGT(
        GT_file=yolo_result_dir + "result.txt", 
        pic_num=pic_num_for_train_MSTN
    )

    autoback.MakeMeanBackground(
        pic_dir=yolo_test_dir + "image/", 
        GT=gt, 
        result_save_path=MSTN_train_img_dir + "target_backgroung_whole.jpg", 
        score_limit=positive_score_limit,
        step=2, 
        log=False,
        #log_dir="./AutoBackground/log/",
        motion_weight=0.5,
        gt_weight=0.5
    )

    target_pos_result = autoback.MakePositiveSamples_fromResult(yolo_result_dir, string="p")

    target_neg_result = autoback.CutBackgroundPic(
        back_pic=MSTN_train_img_dir + "target_backgroung_whole.jpg",
        result_save_path=MSTN_train_img_dir + "target_background/", 
        block_size=[300,150], 
        step_per_block=2,
        string="n"
    )

    autoback.MakeLabel(
        positive_result=target_pos_result,
        negative_result=target_neg_result,
        result_file=MSTN_train_img_dir + "target_with_label.txt"
    )


def MSTN_result_process(mstn_predict_result, score_source, score_target, score_weight=[1.0, 1.0], score_threshold=0.4):
    final_source = score_source * score_weight[0] + score_target * score_weight[1]
    high_score_image_flage = final_source >= score_threshold
    low_score_image_flage = final_source < score_threshold
    return high_score_image_flage, low_score_image_flage


def make_yolo_xml_label(whole_pic_file_name, bwidth, bheight, bdepth, Xmin, Ymin, Xmax, Ymax, box_count, xml_path):
    #with Document() as doc:
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_text = doc.createTextNode("VOC2007")
    folder.appendChild(folder_text)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_str = whole_pic_file_name
    filename_text = doc.createTextNode(filename_str)
    filename.appendChild(filename_text)
    annotation.appendChild(filename)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width_text = doc.createTextNode(str(bwidth))
    width.appendChild(width_text)
    size.appendChild(width)

    height = doc.createElement('height')
    height_text = doc.createTextNode(str(bheight))
    height.appendChild(height_text)
    size.appendChild(height)

    depth = doc.createElement('depth')
    depth_text = doc.createTextNode(str(bdepth))
    depth.appendChild(depth_text)
    size.appendChild(depth)

    segmented = doc.createElement('segmented')
    segmented_text = doc.createTextNode(str(0))
    segmented.appendChild(segmented_text)
    annotation.appendChild(segmented)

    for box in range(box_count):
            

        object1 = doc.createElement('object')
        annotation.appendChild(object1)

        name = doc.createElement('name')
        name_text = doc.createTextNode("person")
        name.appendChild(name_text)
        object1.appendChild(name)

        bndbox = doc.createElement('bndbox')
        object1.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        xmin_text = doc.createTextNode(str(Xmin[box]))
        xmin.appendChild(xmin_text)
        bndbox.appendChild(xmin)

        ymin = doc.createElement('ymin')
        ymin_text = doc.createTextNode(str(Ymin[box]))
        ymin.appendChild(ymin_text)
        bndbox.appendChild(ymin)

        xmax = doc.createElement('xmax')
        xmax_text = doc.createTextNode(str(Xmax[box]))
        xmax.appendChild(xmax_text)
        bndbox.appendChild(xmax)

        ymax = doc.createElement('ymax')
        ymax_text = doc.createTextNode(str(Ymax[box]))
        ymax.appendChild(ymax_text)
        bndbox.appendChild(ymax)

        with open(xml_path, 'w') as f:
            doc.writexml(f, indent='\t', newl='\n',
                         addindent='\t', encoding='utf-8')


def hard_and_pos_processing(hard_result_classify, hard_result_position, hard_result_path, pos_result_position, pos_result_path, high_score_image_flage):
    final_flag = high_score_image_flage * hard_result_classify.T[0]
    final_index = np.where(final_flag)[0]

    hard_result_position_f = hard_result_position[final_index]
    #hard_result_path_f = hard_result_path[final_index]
    hard_result_path_f = []
    for i in range(final_index.shape[0]):
        hard_result_path_f.append(hard_result_path[final_index[i]])

    if pos_result_position.shape[0]:
        result_position = np.concatenate([hard_result_position_f, pos_result_position], axis=0)
        result_path = hard_result_path_f + pos_result_path
    
    else:
        result_position = hard_result_position_f
        result_path = hard_result_path_f

    img_cnt = result_position.T[1]
    img_order_sort_index = np.argsort(img_cnt)

    result_position_sort = result_position[img_order_sort_index]
    result_path_sort = []

    for i in range(img_order_sort_index.shape[0]):
        result_path_sort.append(result_path[img_order_sort_index[i]])

    return result_position_sort, result_path_sort




def make_label_new_postive_sub_pic(mstn_result_dir, yolo_train_dir, yolo_result_dir, mstn_train_img_dir, score_weight):
    """
    将从mstn网络输出的hard样本中得到的正样本加入yolo训练集，同时制作yolo训练使用的标签
    """

    hard_result_classify = np.loadtxt(mstn_result_dir + "result+score.txt")
    hard_result_position = np.loadtxt(yolo_result_dir + "position_hard.txt")
    hard_result_position_final = np.column_stack(
        [
            hard_result_position.T[0], 
            hard_result_position.T[1], 
            hard_result_position.T[2], 
            hard_result_position.T[3], 
            hard_result_position.T[4], 
            hard_result_position.T[5],
            hard_result_classify.T[1]+hard_result_classify.T[2],
        ]
    )
    pos_result_position = np.loadtxt(yolo_result_dir + "position_pos.txt") 
    
    
    with open(yolo_result_dir + "original_path_hard.txt", 'r') as f:
        hard_result_path = f.readlines()
    
    with open(yolo_result_dir + "original_path_pos.txt", 'r') as f:
        pos_result_path = f.readlines()

    high_score_image_flage = np.loadtxt(mstn_result_dir + "high_score_index.txt")

    final_position, final_path = hard_and_pos_processing(hard_result_classify, hard_result_position_final, hard_result_path, pos_result_position, pos_result_path, high_score_image_flage)
    np.savetxt(yolo_train_dir + 'labels.txt', np.column_stack(
        [
            hard_result_position.T[1], 
            hard_result_position.T[0], 
            hard_result_position.T[2], 
            hard_result_position.T[3], 
            hard_result_position.T[4], 
            hard_result_position.T[5],
            hard_result_classify.T[1]*score_weight[0]+hard_result_classify.T[2]*score_weight[1],
        ]
    ))

    xml_save_dir = yolo_train_dir + "annotation/"
    train_picture_dir = yolo_train_dir + "image/"

    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir)
    if not os.path.exists(train_picture_dir):
        os.makedirs(train_picture_dir)

    box_count = 0
    for pic in range(0, np.max(final_position.T[1]).astype(np.int32)+1):
        print('labeling {}/{}...'.format(str(pic+1), str( np.max(final_position.T[1]).astype(np.int32)+1)))
        current_pic_index = np.where(final_position.T[1] == pic)[0]

        if current_pic_index.shape[0] == 0:
            continue
        
        current_img = cv2.imread(mstn_train_img_dir + "target_backgroung_whole.jpg")
        current_position = final_position[current_pic_index]
        for box in range(current_position.shape[0]):
            current_box_position = current_position[box][2:6].astype(np.int32)
            current_box_path = final_path[box_count].split('\n')[0]
            current_box = cv2.imread(current_box_path.split(' ')[0])
            [X, Y] = current_box.shape[0:2]
            current_img[current_box_position[1]:current_box_position[3], current_box_position[0]:current_box_position[2], :] = current_box
            box_count += 1
        
        current_pic_name = str(pic).zfill(6) + ".jpg"
        cv2.imwrite(train_picture_dir + current_pic_name, current_img)

        [bheight, bwidth, bdepth] = current_img.shape
        current_xml_name = str(pic).zfill(6) + ".xml"
        Xmin = final_position[current_pic_index, 2]
        Ymin = final_position[current_pic_index, 3]
        Xmax = final_position[current_pic_index, 4]
        Ymax = final_position[current_pic_index, 5]

        make_yolo_xml_label(
            whole_pic_file_name=current_pic_name, 
            bwidth=bwidth,
            bheight=bheight,
            bdepth=bdepth,
            Xmin=Xmin,
            Ymin=Ymin,
            Xmax=Xmax,
            Ymax=Ymax,
            box_count=current_pic_index.shape[0],
            xml_path=xml_save_dir + current_xml_name
        )
        
    return final_position


def caculate_theta(theta0, beta, nu, result, original_score):
    zeta = np.sum((original_score-beta) * np.sign(result-0.5)) / np.sum(np.abs(original_score - beta))
    theta = 1 - nu * zeta
    return theta


def class_low_result_into_subclasses():
    pass


def label_hard_pic_with_MSTN(
        theta,
        beta,
        yolo_dir,
        mstn_dir,
        mstn_source_train_label, 
        mstn_target_test_file, 
        mstn_target_train_label, 
        mstn_train=False, 
        mstn_test=False,
        add_to_trainset=True, 
        step_log=True, 
        model_name='mstn', 
        train_epoch=400, 
        SS_limit=0.2,
        label_hard_image=True
    ):
    yolo_train_dir = yolo_dir + 'yolo_train/'
    yolo_test_dir = yolo_dir + 'yolo_test/'
    yolo_result_dir = yolo_dir + 'yolo_result/'

    mstn_train_img_dir = mstn_dir + 'MSTN_train_images/'
    mstn_result_dir = mstn_dir + 'MSTN_result/'


    """
    使用MSTN模型训练分类hard样本，并将分类好的正样本加入yolo训练集中并分别制作标签
    """  
    if not label_hard_image:
        return
    
    with open(yolo_result_dir + "picture_numbers.txt", 'r') as f:
        number_list = f.readline().split(" ")
        neg_box_number = int(number_list[0])
        hard_box_number = int(number_list[1])
        pos_box_number = int(number_list[2])
    
    print("{} Start classify {} hard samples.".format(datetime.datetime.now(), str(hard_box_number)))

    training_img_number = min(hard_box_number, 200)
    val_img_number = hard_box_number
    
    if mstn_train == True:
        last_model_path, result_total, SS = mstn_trainmodel(
            TRAINING_FILE=mstn_source_train_label, 
            VAL_FILE=mstn_target_test_file, 
            TARGET_LABEL_FILE=mstn_target_train_label, 
            epochs_limit=train_epoch, 
            step_log=step_log, 
            val_file_num=training_img_number, 
            model_name=model_name
        )
        return

    elif mstn_test == True:
        result_total, SS, result_sub = mstn_label_with_model(
            TRAINING_FILE=mstn_source_train_label, 
            VAL_FILE=mstn_target_test_file, 
            TARGET_LABEL_FILE=mstn_target_train_label, 
            val_file_num=val_img_number,
            SS_limit=SS_limit,
            model_name=model_name,
            train_epoch=train_epoch
        )

        original_hard_score = np.loadtxt(yolo_result_dir+'position_hard.txt')
        log_array = np.column_stack([result_total, SS[0], SS[1], original_hard_score.T[6], result_sub])
        np.savetxt(mstn_result_dir + "result+score.txt", log_array)


        

        print("Hard samples are classified.")
    
    if add_to_trainset:
        res = np.loadtxt(mstn_result_dir + "result+score.txt")
        high_score_image_flage, low_score_image_flage = MSTN_result_process(res.T[0], res.T[1], res.T[2], score_weight=[1.0, 3.0], score_threshold=0.8)
        np.savetxt(mstn_result_dir + "high_score_index.txt", high_score_image_flage)
        np.savetxt(mstn_result_dir + "low_score_index.txt", low_score_image_flage)

        theta_new = caculate_theta(theta, beta, nu=1.0, result=res.T[0], original_score=res.T[3])
        print(theta_new)

        final_position = make_label_new_postive_sub_pic(mstn_result_dir, yolo_train_dir, yolo_result_dir, mstn_train_img_dir, score_weight=[1.0, 3.0])
        picture_number = (np.max(final_position.T[1]) - np.min(final_position.T[1]) + 1).astype(np.int32)
        save_predict_picture_with_box(
            yolo_test_dir + 'image/',
            yolo_train_dir + 'labels.txt',
            yolo_train_dir + 'image_with_box/',
            picture_number=picture_number,
            start_number=np.min(final_position.T[0]).astype(np.int32),
            confidence_limit=0.8
        )
        print("Classified hard samples and their labels are add to yolo train set.")


def label_hard_pic_with_MSTN_noTL(
        yolo_result_dir, 
        mstn_train_img_dir,
        mstn_result_dir, 
        yolo_train_dir, 
        mstn_source_train_label, 
        mstn_target_test_file, 
        mstn_target_train_label, 
        mstn_train=False, 
        add_to_trainset=True, 
        step_log=True, 
        model_name='mstn', 
        train_epoch=400, 
        SS_limit=0.2,
        label_hard_image=True
    ):
    """
    使用MSTN模型训练分类hard样本，并将分类好的正样本加入yolo训练集中并分别制作标签
    **不使用目标域有标签样本，仅供测试使用！**
    """  
    if not label_hard_image:
        return
    
    with open(yolo_result_dir + "picture_numbers.txt", 'r') as f:
        number_list = f.readline().split(" ")
        neg_box_number = int(number_list[0])
        hard_box_number = int(number_list[1])
        pos_box_number = int(number_list[2])

    training_img_number = min(hard_box_number, 200)
    val_img_number = hard_box_number
    
    
    if mstn_train == True:
        last_model_path, result_total, SS = mstn_trainmodel_noTL(
            TRAINING_FILE=mstn_source_train_label, 
            VAL_FILE=mstn_target_test_file, 
            epochs_limit=train_epoch, 
            step_log=step_log, 
            val_file_num=training_img_number, 
            model_name=model_name
        )
   
    result_total, SS = mstn_label_with_model_noTL(
        MODEL_PATH="./MSTN_MODEL/trained_models/" + model_name + str(train_epoch) + ".ckpt", 
        TRAINING_FILE=mstn_source_train_label, 
        VAL_FILE=mstn_target_test_file, 
        val_file_num=val_img_number,
        model_name=model_name
    )

    
    #log_array = np.column_stack([result_total, SS[0], SS[1]])
    #np.savetxt(mstn_result_dir + "result+score.txt", log_array)
    

    #high_score_image_flage = MSTN_result_process(result_total, SS, score_weight=[1.0, 1.0], score_threshold=0.4)
    #np.savetxt(mstn_result_dir + "high_score_index.txt", high_score_image_flage)

    print("Hard samples are classified.")

    #if add_to_trainset:
    #    make_label_new_postive_sub_pic(mstn_result_dir, yolo_train_dir, yolo_result_dir, mstn_train_img_dir)
    #    print("Classified hard samples and their labels are add to yolo train set.")




def model_summary(mstn_result_dir, yolo_result_dir, yolo_test_dir):
    print("{} Summary of model:".format(datetime.datetime.now()))

    with open(yolo_test_dir + "box_number.txt", 'r') as f:
        total_box_number = int(f.readline())

    with open(yolo_result_dir + "picture_numbers.txt", 'r') as f:
        number_list = f.readline().split(" ")
        neg_box_number = int(number_list[0])
        hard_box_number = int(number_list[1])
        pos_box_number = int(number_list[2])
        clow = float(f.readline())
        chigh = float(f.readline())
    
    print("\tYOLO model:")
    print("\t\ttotal person box: {}".format(str(total_box_number)))
    print("\t\tnegetive box: {}".format(str(neg_box_number)))
    print("\t\thard box: {}".format(str(hard_box_number)))
    print("\t\tpositive box: {}".format(str(pos_box_number)))
    
    with open(mstn_result_dir + "hard_label_num.txt", 'r') as f:
        number_list = f.readline().split(" ")
        hard_neg_box_number = int(number_list[0])
        hard_pos_box_number = int(number_list[1])

    print("\tMSTN model:")
    print("\t\ttotal hard box: {}".format(str(hard_box_number)))
    print("\t\tnegetive-hard box: {}".format(str(hard_neg_box_number)))
    print("\t\tpositive-hard box: {}".format(str(hard_pos_box_number)))
