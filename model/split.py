# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/9 上午11:11
@File : split.py
'''
import torch
import torch.nn as nn
import numpy as np


def projection_pooling_row(input):
    b, c, h, w = input.size()
    ave_v = input.mean(dim=3)
    ave_v = ave_v.reshape(b, c, h, -1)
    input[:, :, :, :] = ave_v[:, :, :]
    return input


def projection_pooling_column(input):
    b, c, h, w = input.size()
    input = input.permute(0, 1, 3, 2)
    ave_v = input.mean(dim=3)
    ave_v = ave_v.reshape(b, c, w, -1)
    input[:, :, :, :] = ave_v[:, :, :]
    input = input.permute(0, 1, 3, 2)
    return input


class Block(nn.Module):
    def __init__(self, in_channels, i, row_column=0):
        super(Block, self).__init__()
        self.index = i
        self.row_column = row_column
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=4, dilation=4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=1)
        self.branch1 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels=18, out_channels=1, kernel_size=1)

    def forward(self, input):
        out1 = torch.cat([self.conv1(input), self.conv2(input), self.conv3(input)], dim=1)
        if self.index <= 3:
            if self.row_column == 0:
                out1 = self.pool1(out1)
            else:
                out1 = self.pool2(out1)
        if self.row_column == 0:
            b1 = projection_pooling_row(self.branch1(out1))     # 上分支的投影池化
            b2 = projection_pooling_row(self.branch2(out1))     # 下分支的投影池化
        else:
            b1 = projection_pooling_column(self.branch1(out1))  # 上分支的投影池化
            b2 = projection_pooling_column(self.branch2(out1))  # 下分支的投影池化
        b, c, h, w = b2.size()
        # b2 = b2.squeeze(1)
        b2 = torch.sigmoid(b2)
        output = torch.cat([b1, out1, b2], dim=1)
        return output, b2


class SFCN(nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        cnn = nn.Sequential()
        input_c = [3, 18, 18]
        padding = [3, 3, 6]
        dilation = [1, 1, 2]
        for i in range(3):
            cnn.add_module('sfcn{}'.format(i), nn.Conv2d(input_c[i], 18, 7, padding=padding[i], dilation=dilation[i]))
            cnn.add_module('sfcn_relu{}'.format(i), nn.ReLU(True))
        self.cnn = cnn

    def forward(self, input):
        output = self.cnn(input)
        return output


class Split(nn.Module):

    def __init__(self):
        super(Split, self).__init__()
        self.sfcn = SFCN()
        self.rpn()
        self.cpn()

    def rpn(self):
        self.row_1 = Block(18, 1)
        self.row_2 = Block(37, 2)
        self.row_3 = Block(37, 3)
        self.row_4 = Block(37, 4)
        self.row_5 = Block(37, 5)

    def cpn(self):
        self.column_1 = Block(18, 1, row_column=1)
        self.column_2 = Block(37, 2, row_column=1)
        self.column_3 = Block(37, 3, row_column=1)
        self.column_4 = Block(37, 4, row_column=1)
        self.column_5 = Block(37, 5, row_column=1)

    def forward(self, input):
        out_fcn = self.sfcn(input)
        r1, rp1 = self.row_1(out_fcn)
        r2, rp2 = self.row_2(r1)
        r3, rp3 = self.row_3(r2)
        r4, rp4 = self.row_4(r3)
        r5, rp5 = self.row_5(r4)
        # print(rp5[0, :, :, 0].size())

        c1, cp1 = self.column_1(out_fcn)
        c2, cp2 = self.column_2(c1)
        c3, cp3 = self.column_3(c2)
        c4, cp4 = self.column_4(c3)
        c5, cp5 = self.column_5(c4)
        # print(cp5[0, :, 0, :].size())
        return rp3[0, :, :, 0], rp4[0, :, :, 0], rp5[0, :, :, 0], cp3[0, :, 0, :], cp4[0, :, 0, :], cp5[0, :, 0, :]

if __name__ == '__main__':
    a = np.random.randint(0, 255, size=(1, 3, 500, 500))
    a = a.astype(np.float32)
    input = torch.from_numpy(a)
    split = Split()
    split = split.cuda()
    input = input.cuda()
    split(input)