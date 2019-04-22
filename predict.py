from helpmethods import *
import sys

def main():
    yolo_test_dir = "./YOLO_MODEL/yolo_test/"       #测试换图片集更改此处
    yolo_train_dir = "./YOLO_MODEL/yolo_train/"
    yolo_result_dir = "./YOLO_MODEL/yolo_result/"

    mstn_result_dir = "./MSTN_MODEL/MSTN_result/"
    mstn_train_img_dir = "./MSTN_MODEL/MSTN_train_images/"
    mstn_source_train_label = "./MSTN_MODEL/MSTN_train_images/source.txt"
    mstn_target_train_label = "./MSTN_MODEL/MSTN_train_images/target_with_label.txt"
    #mstn_target_test_file = "./MSTN_MODEL/MSTN_train_images/target_for_test.txt"    #yolo_result_dir + "picture_labels.txt"     #测试换图片集更改此处
    mstn_target_test_file = yolo_result_dir + "picture_labels.txt"

    train_yolo_picture_number = 100
    test_yolo_picture_number = 100
    test_pictuce_start_number = 0

    clow = 0.08
    chigh = 0.6

    """
    train_yolo_init(
        yolo_train_dir, 
        yolo_result_dir,
        mstn_result_dir, 
        train_yolo_picture_number, 
        train_yolo=True, 
        first_train_yolo=False,
        DO_NOT_DEL_TRAIN_PIC=True, 
        model="./YOLO_MODEL/cfg/yolo.cfg", 
        weights="./YOLO_MODEL/bin/yolo.weights",
        #pb="./YOLO_MODEL/built_graph/yolo_test2.pb",
        #meta="./YOLO_MODEL/built_graph/yolo_test2.meta", 
        annotation="./YOLO_MODEL/yolo_train/annotation", 
        dataset="./YOLO_MODEL/yolo_train/image", 
        learningrate="0.00001", 
        batchsize="1", 
        epoch="100", 
        save="500"
    )
    """

    """
    label_with_YOLO(
        yolo_test_dir, 
        test_yolo_picture_number,
        yolo_result_dir, 
        clow, 
        chigh, 
        #pb="./YOLO_MODEL/built_graph/yolo_test2.pb",
        #meta="./YOLO_MODEL/built_graph/yolo_test2.meta",
        model="./YOLO_MODEL/cfg/yolo.cfg",
        load="./YOLO_MODEL/bin/yolo.weights",
        start_number=test_pictuce_start_number, 
        save_picture_with_box=True
    )
    

    
    MSTN_train_set_init(
        yolo_result_dir=yolo_result_dir,
        yolo_test_dir=yolo_test_dir,
        MSTN_train_img_dir=mstn_train_img_dir,
        pic_num_for_train_MSTN=100,
        positive_score_limit=0.2
    )
    """
    
    
    
    
    label_hard_pic_with_MSTN(
        yolo_result_dir,
        mstn_train_img_dir,
        mstn_result_dir,
        yolo_train_dir, 
        mstn_source_train_label, 
        mstn_target_test_file,
        mstn_target_train_label, 
        mstn_train=False, 
        step_log=True,
        result_log=True,
        add_to_trainset=True,
        model_name="towncenter",
        train_epoch=400,
    )
    
    
    
    #model_summary(
    #    mstn_result_dir,
    #    yolo_result_dir,
    #    yolo_test_dir
    #)



if __name__ == "__main__":
    sys.exit(main())
