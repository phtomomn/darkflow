# coding=utf-8
import numpy as np
import cv2
import os
import sys

def ReadGT(GT_file, pic_num=50):
    """
    GT 格式：
        帧内编号
        帧数
        lefttop_x
        lefttop_y
        rightbuttom_x
        rightbuttom_y 
    """
    GT_full = np.loadtxt(GT_file)
    if np.sum(GT_full[:, 1]==pic_num) == 0:
        return GT_full
    else:
        GT = GT_full[0:np.min(np.where(GT_full[:, 1]==pic_num))]
        return GT



def MakeMeanBackground(pic_dir, GT, result_save_path, t=0, step=1, score_limit=0.6, motion_weight=0.6, gt_weight=0.4, log=False, log_dir="./"):
    first_frame = cv2.imread(pic_dir + "000000.jpg")
    back = np.zeros_like(first_frame)

    pic_num = (GT[GT.shape[0]-1][1] + 1).astype(int)

    if t == 0:
        t = min(1.0 - 1/(pic_num/step), 0.9)

    frame_previous = cv2.imread(pic_dir + str(0).zfill(6) + ".jpg")
    for pic in range(0, pic_num, step):
        mask_groundtruth = np.ones_like(first_frame, dtype=np.int32)
        frame_current = cv2.imread(pic_dir + str(pic).zfill(6) + ".jpg")

        mask_motion_3d = np.exp(-0.2*np.abs(frame_current - frame_previous))
        mask_motion_2d = (mask_motion_3d[:, :, 0] + mask_motion_3d[:, :, 1] + mask_motion_3d[:, :, 2]) / 3

        mask_motion_3d[:, :, 0] = mask_motion_2d
        mask_motion_3d[:, :, 1] = mask_motion_2d
        mask_motion_3d[:, :, 2] = mask_motion_2d

        GT_current = GT[np.where(GT[:, 1]==pic)]

        for box in range(GT_current.shape[0]):
            [lefttop_x, lefttop_y, rightbuttom_x, rightbuttom_y] = GT_current[box, 2:6].astype(np.int32)
            if GT_current[box, 6] >= score_limit:
                mask_groundtruth[lefttop_y:rightbuttom_y, lefttop_x:rightbuttom_x, :] = 0
        
        if log:
            cv2.imwrite(log_dir + str(pic).zfill(6) + ".jpg", frame_current * mask_motion_3d)

        back_weight_groundtruth = t * mask_groundtruth + 1.0 * (1.0 - mask_groundtruth)
        back_weight_motion = t * mask_motion_3d + 1.0 * (1.0 - mask_motion_3d)

        back_weight = gt_weight * back_weight_groundtruth + motion_weight * back_weight_motion
        mask_front = gt_weight * mask_groundtruth + motion_weight * mask_motion_3d

        back = back_weight * back + (1-t) * (frame_current * mask_front)
        
        
        frame_previous = frame_current
    
    cv2.imwrite(result_save_path, back)


def MakePositiveSamples_fromResult(yolo_result_dir, string="p"):
    with open(yolo_result_dir + "picture_numbers.txt", 'r') as f:
        number_list = f.readline().split(" ")
        pos_box_number = int(number_list[2])
    
    path = yolo_result_dir + "positive/"

    return [pos_box_number, path, string]
    



def CutBackgroundPic(back_pic, result_save_path, block_size=[200, 100], step_per_block=3, string="n"):
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
        
    img = cv2.imread(back_pic)

    (x, y, _) = img.shape
    
    x_size = block_size[0]
    y_size = block_size[1]

    x_step = block_size[0] // step_per_block
    y_step = block_size[1] // step_per_block

    back_count = 0
   
    for xx in range(0, x, x_step):
        for yy in range(0, y, y_step):
            subpic_current = img[xx : xx + x_size, yy : yy + y_size, :]
            cv2.imwrite(result_save_path + string + str(back_count)+".jpg", subpic_current)
            back_count += 1
    
    return [back_count, result_save_path, string]


def MakeLabel(positive_result, negative_result, result_file):
    [p_count, positive_path, string_p] = positive_result
    [n_count, negative_path, string_n] = negative_result

    with open(result_file, 'w') as f:
        for p_count in range(p_count):
            f.write(positive_path + string_p + str(p_count) + ".jpg 1\n")
        
        for n_count in range(n_count):
            f.write(negative_path + string_n + str(n_count) + ".jpg 0\n")
                    

def main():
    gt = ReadGT(GT_file="./GT100.txt", pic_num=100)
    #MakeMeanBackground(pic_dir="./VideoFrames/", GT=gt, result_save_path="./result.jpg", step=2, pic_num=100) #, log=True, log_dir="./log/")
    person_count = MakePositiveSamples(pic_dir="./VideoFrames/", GT=gt, result_save_path="./target_person/", pic_num=100, step=3)
    back_count = CutBackgroundPic(back_pic="./result.jpg", result_save_path="./target_background/", block_size=[200,100], step_per_block=2)
    MakeLabel(positive_path="./target_person/", negative_path="./target_background/", positive_count=person_count, negative_count=back_count, result_save_path="./target_with_label.txt")

    print(person_count, back_count)

if __name__ == "__main__":
    sys.exit(main())