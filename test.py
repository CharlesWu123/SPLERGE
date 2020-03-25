# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/12 下午4:09
@File : test.py
'''
import time

import torch
import cv2 as cv
import numpy as np
from config import params
from model.split import Split

threshold = 0.9

model_path = './trained_models/split_295.pth'
split = Split()
if torch.cuda.is_available():
    split = split.cuda()
print('loading pretrained model from {0}'.format(model_path))
split.load_state_dict(torch.load(model_path, map_location=None if torch.cuda.is_available() else 'cpu'))


def table_reg(image, model):
    image = image.transpose(2, 0, 1)  # h,w,c -> c,h,w
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image)
    image.sub_(params.mean).div_(params.std)

    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    model.eval()
    _, _, pred_row, _, _, pred_col = model(image)
    pred_row = pred_row.cpu().detach().numpy()[0]
    pred_col = pred_col.cpu().detach().numpy()[0]
    pred_row[pred_row >= threshold] = 1
    pred_row[pred_row < threshold] = 0
    pred_col[pred_col >= threshold] = 1
    pred_col[pred_col < threshold] = 0
    row_line_index = np.nonzero(pred_row)[0]
    col_line_index = np.nonzero(pred_col)[0]
    return row_line_index, col_line_index


def draw_line(image, row_line_index, col_line_index, save_path, ratio):
    h, w, c = image.shape
    # 找出中点
    row_line_index = center_line(row_line_index)
    col_line_index = center_line(col_line_index)
    # 解决两条线离得很近的情况（实际上有一条不是表格线）
    # todo
    for i in row_line_index:
        cv.line(image, (0, i), (w, i), color=(0, 0, 255), thickness=1)
    for j in col_line_index:
        cv.line(image, (j, 0), (j, h), color=(0, 255, 0), thickness=1)
    # image = cv.resize(image, None, fx=1/ratio, fy=1/ratio, interpolation=cv.INTER_CUBIC)
    cv.imwrite(save_path, image)


def center_line(line_index):
    # 通过表格线区域找出此区域的中点
    res = []
    tmp_index = [line_index[0]]
    for i in range(1, len(line_index)):
        if line_index[i] == line_index[i-1] + 1:
            tmp_index.append(line_index[i])
        else:
            res.append(int(np.median(tmp_index)))
            tmp_index = [line_index[i]]
    if tmp_index:
        res.append(int(np.median(tmp_index)))
    return res


def image_resize(size, new_size):
    h, w = size
    new_h, new_w = new_size
    ratio = 1
    if h > new_h:
        ratio = new_h / h
        new_w = int(w * ratio)
    elif w > new_w:
        ratio = new_w / w
        new_h = int(h * ratio)
    else:
        new_w, new_h = w, h
    return new_w, new_h, ratio


if __name__ == '__main__':
    begin_time = time.time()
    image_path = './data_sample/web.jpg'
    save_path = './data_sample/web_pred2.jpg'
    image = cv.imread(image_path)
    h, w, c = image.shape
    new_h, new_w = 600, 600     # 限制高和宽最大为 600
    new_w, new_h, ratio = image_resize((h, w), (new_h, new_w))
    image_split = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    row_line_index, col_line_index = table_reg(image_split, split)
    draw_line(image_split, row_line_index, col_line_index, save_path, ratio)
    end_time = time.time()
    print('Use Time : {}'.format(end_time - begin_time))
