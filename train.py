# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/12 下午1:55
@File : train.py.py
'''
import argparse

import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from config import params
from model.merge import Merge
from model.split import Split
from model.loss import loss
from data_generator.generator import TableDataset

writer = SummaryWriter('./scalar')

def init_args():
    args = argparse.ArgumentParser()
    args.add_argument('--images_dir', help='path to dataset', default='/home/charleswu/deeplearning/data/表格数据集/ICDAR2013_SPLERGE_train_data/table_img/images')
    args.add_argument('--json_dir', help='path to dataset', default='/home/charleswu/deeplearning/data/表格数据集/ICDAR2013_SPLERGE_train_data/table_img/json')
    args.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
    return args.parse_args()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


def train(split, merge, train_loader, criterion, optimizer, iteration):
    for p in split.parameters():
        p.requires_grad = True
    for p in merge.parameters():
        p.requires_grad = True
    split.train()
    data_len = len(train_loader)
    for i_batch, (image, label) in enumerate(train_loader):
        if args.cuda:
            image = image.cuda()
        cost = loss(image, label, (None, None), split, merge, criterion)
        split.zero_grad()
        cost.backward()
        optimizer.step()
        writer.add_scalar('train', cost, iteration * data_len + i_batch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration * data_len + i_batch)

        if (i_batch + 1) % params.displayInterval == 0:
            print("[{}/{}][{}/{}] Loss: {}".format(iteration, params.niter, i_batch, data_len, cost))

def val():
    pass


def main(split, merge, train_loader, criterion, optimizer):
    Iteration = 0
    while Iteration < params.niter:
        train(split, merge, train_loader, criterion, optimizer, Iteration)
        adjust_learning_rate(optimizer, Iteration)
        if Iteration % params.saveModel == 0:
            torch.save(split.state_dict(), '{}/split_{}.pth'.format(params.experiment, Iteration))
        Iteration += 1


def adjust_learning_rate(optimizer, epoch):
    """设置学习率衰减 """
    lr = params.lr * (0.75 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    args = init_args()
    image_dir = args.images_dir
    json_dir = args.json_dir
    dataset = TableDataset(image_dir, json_dir)
    train_loader = DataLoader(dataset, batch_size=params.batchSize, shuffle=True, num_workers=params.workers)

    criterion = torch.nn.BCELoss()
    split = Split()
    merge = Merge()
    if args.cuda:
        split = split.cuda()
        merge = merge.cuda()
        criterion = criterion.cuda()
    split.apply(weights_init)

    if params.trained_model:
        print("loading pretrained model from {}".format(params.trained_model))
        split.load_state_dict(torch.load(params.trained_model))

    optimizer = optim.Adam(split.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
    main(split, merge, train_loader, criterion, optimizer)