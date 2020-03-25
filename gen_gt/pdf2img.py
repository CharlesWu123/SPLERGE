# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/11 下午1:46
@File : pdf2img.py
'''
import os

import fitz
# https://pymupdf.readthedocs.io/en/latest/faq/


class ICDAR2013:
    def __init__(self, pdf_path=None, pdf_dir=None, save_dir=None):
        self.pdf_path = pdf_path
        self.pdf_dir = pdf_dir
        self.save_dir = save_dir

    def gen_img(self):
        if not self.save_dir:
            raise Exception("请传入图片保存文件夹参数")
        if self.pdf_path:
            return self._process_single(self.pdf_path, self.save_dir)
        elif self.pdf_dir:
            return self._process_batch(self.pdf_dir, self.save_dir)
        else:
            raise Exception("请传入PDF文件参数或者PDF所在文件夹参数")

    @staticmethod
    def _process_single(pdf_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        pdf_doc = fitz.open(pdf_path)
        images_path_list = []
        for pg in range(pdf_doc.pageCount):
            page = pdf_doc[pg]
            rotate = int(0)
            # 尺寸的缩放系数
            zoom_x, zoom_y = 1, 1
            mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
            pix = page.getPixmap(matrix=mat, alpha=False)
            image_path = os.path.join(save_dir, str(pg) + '.png')
            pix.writePNG(image_path)
            images_path_list.append(image_path)
        return images_path_list

    def _process_batch(self, pdf_dir, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        images_path_list = []
        for cur_dir, next_dir, file_names in os.walk(pdf_dir):
            for file_name in file_names:
                name, ext = os.path.splitext(file_name)
                if ext != '.pdf': continue
                cur_save_dir = os.path.join(save_dir, name)
                pdf_path = os.path.join(cur_dir, file_name)
                images_path_list.append(self._process_single(pdf_path, cur_save_dir))
        return images_path_list


if __name__ == '__main__':
    pdf_path = '/home/charleswu/deeplearning/data/表格数据集/ICDAR2013/icdar2013-competition-dataset-with-gt/competition-dataset-us/us-014.pdf'
    save_dir = './'
    pdf2img = ICDAR2013(pdf_path=pdf_path, save_dir=save_dir)
    pdf2img.gen_img()