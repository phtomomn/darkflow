## Intro


## Dependencies
Python3, tensorflow 1.0, numpy, opencv 3.

## 图片集准备
一次完整的检测-困难样本分类-训练检测器需要

- 在 `./YOLO_MODEL/yolo_test/image/` 中放入待检测图片；
- 在 `./YOLO_MODEL/cfg/` 中放入yolo配置文件 `XXX.cfg` ；
- 在 `./YOLO_MODEL/bin/` 中放入yolo预训练的权重 `XXX.weights` ；
- 在 `./MSTN_MODEL/MSTN_models/` 中放入预训练的权重 `bvlc_alexnet.npy` 。

## Getting started
修改 `predict.py` 中的注释内容来确定实现功能，之后运行
   
   ```
    python predict.py
   ```

## 结果保存
- yolo检测结果保存在 `./YOLO_MODEL/yolo_result/`;
- 困难样本分类结果保存在 `./MSTN_MODEL/mstn_result/`，制作好的yolo训练图片与其标签保存在 `./YOLO_MODEL/yolo_train/` 。
