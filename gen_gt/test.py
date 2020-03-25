# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/11 下午7:39
@File : test.py.py
'''
import os
import json
import cv2 as cv


img_path = '/home/charleswu/deeplearning/data/表格数据集/ICDAR2013_SPLERGE_train_data/table_img/0b16bc138c4741d0b52e8986dfa6a568.png'
json_path = '/home/charleswu/deeplearning/data/表格数据集/ICDAR2013_SPLERGE_train_data/table_img/0b16bc138c4741d0b52e8986dfa6a568.json'
img = cv.imread(img_path)
with open(json_path, 'r', encoding='utf-8') as f:
    res = json.load(f)
for r in res:
    cell_coord = r['cell_coord']
    content = r['content']
    crop_img = img[cell_coord[2]:cell_coord[3], cell_coord[0]:cell_coord[1]]
    print(content)
    cv.imshow('1', crop_img)
    break
cv.waitKey()