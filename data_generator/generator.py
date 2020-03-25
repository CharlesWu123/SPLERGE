# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/12 上午10:14
@File : generator.py.py
'''
import json
import os
import cv2 as cv
import numpy as np
import torch
from config import params
from torch.utils.data import Dataset, DataLoader


class TableDataset(Dataset):
    def __init__(self, images_dir, json_dir):
        self.images_dir = images_dir
        self.json_dir = json_dir
        self.labels = []
        self._get_labels()

    def _get_labels(self):
        json_list = os.listdir(self.json_dir)
        for json_name in json_list:
            name, ext = os.path.splitext(json_name)
            json_path = os.path.join(self.json_dir, json_name)
            image_path = os.path.join(self.images_dir, name + '.png')
            img = cv.imread(image_path)
            h, w, c = img.shape
            self.labels.append((image_path, self._get_label(json_path, h, w)))      # (image_path, label)

    @staticmethod
    def _get_label(json_path, h, w):
        mask_img_row = np.ones((h, w), dtype=np.uint8)
        mask_img_col = np.ones((h, w), dtype=np.uint8)
        with open(json_path, 'r', encoding='utf-8') as f:
            cells = json.load(f)
        # 行，排除跨行单元格
        for cell in cells:
            start_row, end_row, start_col, end_col = cell['row_column']
            if end_row != -1: continue
            x_min, x_max, y_min, y_max = cell['cell_coord']
            mask_img_row[y_min:y_max, x_min:x_max] = 0
        # 列，排除跨列单元格
        for cell in cells:
            start_row, end_row, start_col, end_col = cell['row_column']
            if end_col != -1: continue
            x_min, x_max, y_min, y_max = cell['cell_coord']
            mask_img_col[y_min:y_max, x_min:x_max] = 0
        row_label = np.all(mask_img_row, axis=1).astype(np.float32)
        col_label = np.all(mask_img_col, axis=0).astype(np.float32)
        return torch.from_numpy(row_label), torch.from_numpy(col_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path, label = self.labels[item]
        image = cv.imread(image_path)
        image = image.transpose(2, 0, 1)  # h,w,c -> c,h,w
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image)
        image.sub_(params.mean).div_(params.std)
        return image, label


if __name__ == '__main__':
    # 0 黑  255 白
    img_path = '/home/charleswu/deeplearning/data/表格数据集/ICDAR2013_SPLERGE_train_data/table_img/images/1b6273a89d78430bb54d59fa21c9fc31.png'
    json_path = '/home/charleswu/deeplearning/data/表格数据集/ICDAR2013_SPLERGE_train_data/table_img/json/1b6273a89d78430bb54d59fa21c9fc31.json'

    img = cv.imread(img_path)
    h, w, c = img.shape
    mask_img_row = np.ones((h, w), dtype=np.uint8)
    mask_img_col = np.ones((h, w), dtype=np.uint8)
    with open(json_path, 'r', encoding='utf-8') as f:
        cells = json.load(f)
    for cell in cells:
        start_row, end_row, start_col, end_col = cell['row_column']
        if end_row != -1: continue
        x_min, x_max, y_min, y_max = cell['cell_coord']
        mask_img_row[y_min:y_max, x_min:x_max] = 0
    for cell in cells:
        start_row, end_row, start_col, end_col = cell['row_column']
        if end_col != -1: continue
        x_min, x_max, y_min, y_max = cell['cell_coord']
        mask_img_col[y_min:y_max, x_min:x_max] = 0
    row_label = np.all(mask_img_row, axis=1).astype(np.uint8)
    col_label = np.all(mask_img_col, axis=0).astype(np.uint8)
    # row_mask = row_label.reshape((-1, 1)).repeat(w, axis=1)
    # col_mask = col_label.reshape((1, -1)).repeat(h, axis=0)
    # show_img = mask_img * 255
    # row_mask = row_mask * 255
    # col_mask = col_mask * 255
    # cv.imshow('1', show_img)
    # cv.imshow('2', row_mask)
    # cv.imshow('3', col_mask)

    row_line_index = np.nonzero(row_label)[0]
    col_line_index = np.nonzero(col_label)[0]
    for i in row_line_index:
        cv.line(img, (0, i), (w, i), color=(0, 0, 255), thickness=1)
    for j in col_line_index:
        cv.line(img, (j, 0), (j, h), color=(0, 255, 0), thickness=1)
    cv.imshow('1', img)
    cv.waitKey()

    # images_dir = '/home/charleswu/deeplearning/data/表格数据集/ICDAR2013_SPLERGE_train_data/table_img/images'
    # json_dir = '/home/charleswu/deeplearning/data/表格数据集/ICDAR2013_SPLERGE_train_data/table_img/json'
    # dataset = TableDataset(images_dir, json_dir)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for i_batch, (image, label) in enumerate(dataloader):
    #     print(i_batch, image.shape, label[0].shape, label[1].shape)
