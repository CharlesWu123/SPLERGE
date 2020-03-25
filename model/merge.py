# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/9 上午11:11
@File : merge.py
'''

import torch
import torch.nn as nn
import numpy as np


class GridPooling():
    def __init__(self, grid_struc):
        # 表格线用 1 表示，其他均为 0
        h, w = grid_struc.shape
        self.zone = [[0], [0]]
        for i in range(h):
            if all(grid_struc[i, :]) and i != 0:
                self.zone[0].append(i)
        for j in range(w):
            if all(grid_struc[:, j]) and j != 0:
                self.zone[1].append(j)
        self.zone[0].append(h)
        self.zone[1].append(w)

    def get_zone(self):
        return self.zone

    def forward(self, input):
        b, c, h, w = input.size()
        for ib in range(b):
            for ic in range(c):
                for i in range(len(self.zone[0])-1):
                    row = (self.zone[0][i], self.zone[0][i+1])
                    for j in range(len(self.zone[1])-1):
                        col = (self.zone[1][j], self.zone[1][j+1])
                        grid_mean = torch.mean(input[ib, ic, row[0]:row[1], col[0]:col[1]])
                        input[ib, ic, row[0]:row[1], col[0]:col[1]] = grid_mean
        return input

    def __call__(self, *input, **kwargs):
        return self.forward(*input)


class Block(nn.Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=3, dilation=3)
        self.branch1 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels=18, out_channels=1, kernel_size=1)

    def forward(self, input, grid_pool):
        out1 = torch.cat([self.conv1(input), self.conv2(input), self.conv3(input)], dim=1)
        b1 = grid_pool(self.branch1(out1))     # 上分支的表格池化
        b2 = grid_pool(self.branch2(out1))     # 下分支的表格池化
        b2 = torch.sigmoid(b2)
        output = torch.cat([b1, out1, b2], dim=1)
        return output, b2


class Branch(nn.Module):
    def __init__(self):
        super(Branch, self).__init__()
        self.block1 = Block(18)
        self.block2 = Block(37)
        self.block3 = Block(37)

    def forward(self, input, grid_pool):
        bran1, bp1 = self.block1(input, grid_pool)
        bran2, bp2 = self.block2(bran1, grid_pool)
        bran3, bp3 = self.block3(bran2, grid_pool)
        return bp2, bp3


class SFCN(nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        cnn = nn.Sequential()
        input_c = [8, 18, 18, 18]
        padding = [3, 3, 3, 3]
        for i in range(4):
            cnn.add_module('merge_sfcn{}'.format(i), nn.Conv2d(input_c[i], 18, 7, padding=padding[i]))
            cnn.add_module('merge_sfcn_relu{}'.format(i), nn.ReLU(True))
            if i == 1 or i == 3:
                cnn.add_module('merge_sfcn_pool{}'.format(i), nn.AvgPool2d(2, 1))
        self.cnn = cnn

    def forward(self, input):
        output = self.cnn(input)
        return output


class Merge(nn.Module):
    def __init__(self):
        super(Merge, self).__init__()
        self.sfcn = SFCN()
        self.branch_up = Branch()
        self.branch_down = Branch()
        self.branch_left = Branch()
        self.branch_right = Branch()

    def forward(self, input, grid_struc):
        grid_pool = GridPooling(grid_struc)
        out_sfcn = self.sfcn(input)
        out_up2, out_up3 = self.branch_up(out_sfcn, grid_pool)
        out_down2, out_down3 = self.branch_down(out_sfcn, grid_pool)
        out_left2, out_left3 = self.branch_left(out_sfcn, grid_pool)
        out_right2, out_right3 = self.branch_right(out_sfcn, grid_pool)
        print(out_right2.size())
        return out_up2, out_up3, out_down2, out_down3, out_left2, out_left3, out_right2, out_right3, grid_pool.get_zone()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    input = np.random.randint(0, 255, size=(1, 8, 100, 100))
    input = input.astype(np.float32)
    input = torch.from_numpy(input)
    input = input.cuda()
    mask = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        if i % 20 == 0:
            mask[i, :] = 1
    for i in range(100):
        if i % 20 == 0:
            mask[:, i] = 1
    merge = Merge()
    merge.cuda()
    merge.apply(weights_init)
    merge(input, mask)