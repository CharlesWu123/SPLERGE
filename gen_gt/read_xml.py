# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/11 下午2:13
@File : read_xml.py
'''
# -*- coding: utf-8 -*-
import json
from xml.dom.minidom import parse, parseString


class ReadXml:
    '''
    xml处理类: 读 XML
    '''
    def __init__(self, reg_xml_path, str_xml_path):
        '''
        :param xml_path:
        :param xml_char:
        '''
        self.result = []
        self.reg_objs = parseString(self._bytes_decode(reg_xml_path))
        self.str_objs = parseString(self._bytes_decode(str_xml_path))
        self.get_table_coord()

    def get_result(self):
        return self.result

    def _bytes_decode(self, file_path):
        '''
        防止编码问题，首先使用二进制方式读取, 以字符串方式解析
        :param file_path: 文件路径
        :return:
        '''
        with open(file_path, 'rb') as f:
            content = f.read()
        return bytes.decode(content, encoding='utf-8')

    def get_table_coord(self):
        table_reg = self.reg_objs.getElementsByTagName('table')
        table_str = self.str_objs.getElementsByTagName('table')
        for tr, ts in zip(table_reg, table_str):
            # 处理表格
            region = tr.getElementsByTagName('region')
            if len(region) > 1: continue                # 大于 1 的为一个表格跨页情况，不处理
            page = region[0].getAttribute('page')       # 获取表格所在的页数
            table_bbox = tr.getElementsByTagName('bounding-box')[0]
            # 获取表格的坐标
            table_x1, table_x2, table_y1, table_y2 = table_bbox.getAttribute('x1'), table_bbox.getAttribute('x2'), table_bbox.getAttribute('y1'), table_bbox.getAttribute('y2')
            table_x1, table_x2, table_y1, table_y2 = map(int, [table_x1, table_x2, table_y1, table_y2])
            # 处理单元格
            cells = ts.getElementsByTagName('cell')
            cells_list = []
            for cell in cells:
                start_col, start_row, end_col, end_row = cell.getAttribute('start-col'), cell.getAttribute('start-row'), cell.getAttribute('end-col'), cell.getAttribute('end-row')
                if not end_row: end_row = -1
                if not end_col: end_col = -1
                bbox = cell.getElementsByTagName('bounding-box')[0]
                cell_x1, cell_x2, cell_y1, cell_y2 = bbox.getAttribute('x1'), bbox.getAttribute('x2'), bbox.getAttribute('y1'), bbox.getAttribute('y2')
                content = cell.getElementsByTagName('content')[0]
                if content.childNodes:
                    content = content.childNodes[0].data
                else:
                    content = ""
                start_col, start_row, end_col, end_row = map(int, [start_col, start_row, end_col, end_row])
                cell_x1, cell_x2, cell_y1, cell_y2 = map(int, [cell_x1, cell_x2, cell_y1, cell_y2])
                # 存储单元格信息
                cells_list.append({
                    "row_column": [start_row, end_row, start_col, end_col],
                    "cell_coord": [cell_x1, cell_x2, cell_y1, cell_y2],
                    "content": content
                })
            # 存储表格所有信息
            self.result.append({
                "page": int(page) - 1,
                "table_coord": [table_x1, table_x2, table_y1, table_y2],
                "cells": cells_list
            })


if __name__ in "__main__":
    objs = ReadXml(reg_xml_path='/home/charleswu/deeplearning/data/表格数据集/ICDAR2013/eu-dataset/eu-001-reg.xml',
                   str_xml_path='/home/charleswu/deeplearning/data/表格数据集/ICDAR2013/eu-dataset/eu-001-str.xml')
    result = objs.get_result()
    with open('2.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))

