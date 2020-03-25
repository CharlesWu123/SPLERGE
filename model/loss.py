# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/10 下午2:20
@File : loss.py
'''
import torch


def loss(input, label_split, label_merge, split, merge, criterion):
    # criterion 是 nn.BCELoss()
    label_r, label_c = label_split
    label_r, label_c = label_r.cuda(), label_c.cuda()
    label_D, label_R = label_merge
    # 分割模型
    rp3, rp4, rp5, cp3, cp4, cp5 = split(input)     # rp3 b*H  cp3 b*W
    # 合并模型
    # merge_input, grid_struct = get_merge_input(input, rp5, cp5)
    # out_up2, out_up3, out_down2, out_down3, out_left2, out_left3, out_right2, out_right3, split_zone = merge(merge_input, grid_struct)
    # D2, D3, R2, R3 = calc_prob_matrix(out_up2, out_up3, out_down2, out_down3, out_left2, out_left3, out_right2, out_right3, split_zone)
    # 分割损失

    L_split_tot = criterion(rp5, label_r) + 0.25 * criterion(rp4, label_r) + 0.1 * criterion(rp3, label_r) + \
                  criterion(cp5, label_c) + 0.25 * criterion(cp4, label_c) + 0.1 * criterion(cp3, label_c)
    # 合并损失
    # L_merge_tot = criterion(D3, label_D) + 0.25 * criterion(D2, label_D) + \
    #               criterion(R3, label_R) + criterion(R2, label_R)
    # 总损失
    # L_tot = L_split_tot + L_merge_tot
    L_tot = L_split_tot
    return L_tot


def get_merge_input(input, row, col):
    B, C, H, W = input.size()
    rb, rh = row.size()
    cb, ch = col.size()
    assert rb == H and cb == W, "输入图像大小与Split模型输出的行列大小不一致"
    row_ex = row.reshape((1, 1, -1, 1)).expand(1, 1, -1, W)       # 拓展行概率[r] -> [r, r, ..., r]        b*h -> b*c*h*w   b=c=1
    col_ex = col.reshape((1, 1, 1, -1)).expand(1, 1, H, -1)       # 拓展列概率[c] -> [[c], [c], ..., [c]]  b*w -> b*c*h*w
    row_region = torch.zeros((H, W), dtype=torch.float32)
    col_region = torch.zeros((H, W), dtype=torch.float32)


def calc_prob_matrix(out_up2, out_up3, out_down2, out_down3, out_left2, out_left3, out_right2, out_right3, split_zone):
    # b, c, h, w = out_up2.size()
    # b*c*h*w  b=c=1
    out_h, out_w = len(split_zone[0])-1, len(split_zone[1])-1
    # 生成 MxN 的矩阵 u,d,l,r
    def grid_mean(input):
        out = torch.zeros((out_h, out_w), dtype=torch.float32)
        for i in range(out_h):
            row = (split_zone[0][i], split_zone[0][i + 1])
            for j in range(out_w):
                col = (split_zone[1][j], split_zone[1][j + 1])
                grid_mean_v = torch.mean(input[0, 0, row[0]:row[1], col[0]:col[1]])
                out[i, j] = grid_mean_v
        return out
    u2, u3, d2, d3 = grid_mean(out_up2), grid_mean(out_up3), grid_mean(out_down2), grid_mean(out_down3)
    l2, l3, r2, r3 = grid_mean(out_left2), grid_mean(out_left3), grid_mean(out_right2), grid_mean(out_right3)
    # 计算上下合并的概率
    D2 = u2[1:, :] * d2[:-1, :] / 2 + (u2[1:, :] + d2[:-1, :]) / 4
    D3 = u3[1:, :] * d3[:-1, :] / 2 + (u3[1:, :] + d3[:-1, :]) / 4
    # 计算左右合并的概率
    R2 = l2[:, 1:] * r2[:, :-1] / 2 + (l2[:, 1:] + r2[:, :-1]) / 4
    R3 = l3[:, 1:] * r3[:, :-1] / 2 + (l3[:, 1:] + r3[:, :-1]) / 4
    return D2, D3, R2, R3
