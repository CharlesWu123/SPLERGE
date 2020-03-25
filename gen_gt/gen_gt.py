# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/11 下午2:13
@File : gen_gt.py
'''
import json
import os
import cv2 as cv
from uuid import uuid4
from pdf2img import ICDAR2013
from read_xml import ReadXml


def gen_table_gt(dir_path, pdf_img_save_dir, table_img_save_dir):
    # 将pdf转换为图片，并且截取其中的表格 获得gt信息
    table_img_list = []
    for cur_dir, next_dir, file_names in os.walk(dir_path):
        for file_name in file_names:
            name, ext = os.path.splitext(file_name)
            if ext != '.pdf': continue
            pdf_path = os.path.join(cur_dir, file_name)
            print('process {}'.format(pdf_path))
            cur_save_dir = os.path.join(pdf_img_save_dir, name)
            os.makedirs(cur_save_dir, exist_ok=True)
            reg_xml_path = os.path.join(cur_dir, name + '-reg.xml')
            str_xml_path = os.path.join(cur_dir, name + '-str.xml')
            # 将 pdf 转换为图片
            images_path_list = ICDAR2013(pdf_path=pdf_path, save_dir=cur_save_dir).gen_img()
            # 提取 xml gt 中的信息
            xml_result = ReadXml(reg_xml_path=reg_xml_path, str_xml_path=str_xml_path).get_result()
            for dict_res in xml_result:
                image_path = images_path_list[dict_res['page']]
                img_path, img_gt = extract_table(image_path, dict_res, table_img_save_dir)
                table_img_list.append(img_path)


def extract_table(image_path, dict_res, save_dir):
    save_name = uuid4().hex
    # 提取 table 图片，并且生成对应的表格结构gt
    table_coord = dict_res['table_coord']   # x1, x2, y1, y2
    img = cv.imread(image_path)
    h, w, c = img.shape
    table_x_min, table_x_max, table_y_min, table_y_max = table_coord[0], table_coord[1], h-table_coord[3], h-table_coord[2]
    table_img = img[table_y_min:table_y_max, table_x_min:table_x_max]
    save_path = os.path.join(save_dir, save_name + '.png')
    cv.imwrite(save_path, table_img)
    # 生成该 table 的gt
    table_gt = []
    cells = dict_res['cells']
    for cell in cells:
        row_column = cell['row_column']
        cell_coord = cell['cell_coord']
        # 转换成图片的坐标，以左上为原点
        cell_x_min, cell_x_max, cell_y_min, cell_y_max = cell_coord[0] - table_x_min, cell_coord[1] - table_x_min, \
                                                         table_coord[3] - cell_coord[3], table_coord[3] - cell_coord[2]
        content = cell['content']
        table_gt.append({
            "row_column": row_column,
            "cell_coord": [cell_x_min, cell_x_max, cell_y_min, cell_y_max],
            "content": content
        })
    with open(os.path.join(save_dir, save_name + '.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(table_gt, ensure_ascii=False))
    return save_path, table_gt


if __name__ == '__main__':
    dir_path = '/home/charleswu/deeplearning/data/表格数据集/ICDAR2013/us-gov-dataset'
    pdf_img_save_dir = '/home/charleswu/deeplearning/data/表格数据集/ICDAR2013_SPLERGE_train_data/pdf2img'
    table_img_save_dir = '/home/charleswu/deeplearning/data/表格数据集/ICDAR2013_SPLERGE_train_data/table_img'
    gen_table_gt(dir_path, pdf_img_save_dir, table_img_save_dir)