# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   dataloader.py
@Time    :   2022/11/23 17:28:10
@Author  :   ZeyuXiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   None
"""

import os
from torch.utils.data.dataset import Dataset
from utils.degrade import *
import numpy as np
import h5py
from torch.utils.data import DataLoader
from einops import rearrange
import torch
import random

class TrainSetLoader(): 
    def __init__(self, args):
        super(TrainSetLoader, self).__init__()
        self.args = args

        self.file_path = args.trainset_dir
        self.psize = args.patch_size
        hf = h5py.File(self.file_path,'r')
        self.img_HR = hf.get('LF_RGB')  # [N,ah,aw,h,w]
        self.img_size = hf.get('LF_size')  # [N,2]
    def __getitem__(self, index):
        

        lf = self.img_HR[index]
        H, W = self.img_size[index]
        x = random.randrange(0, H - self.psize, 8)
        y = random.randrange(0, W - self.psize, 8)
        hr = lf[:, :, :,x:x + self.psize, y:y + self.psize]  # [ah,aw,ph,pw]

        hr = torch.from_numpy(hr.astype(np.float32) / 255.0)

        return hr

    def __len__(self):
        return self.img_HR.shape[0]


def MultiTestSetDataLoader(args):
    data_list = os.listdir(args.testset_dir2)
    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name)
        length_of_tests += len(test_Dataset)
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name):
        super(TestSetDataLoader, self).__init__()
        self.args = args
        self.file_list = []
        self.testset_dir = args.testset_dir2 + data_name
        tmp_list = os.listdir(self.testset_dir)
        self.gen_LR = MultiDegrade(
            scale=self.args.upfactor,
            sig=args.sig,
            noise=args.noise,
        )
        self.file_list.extend(tmp_list)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = self.testset_dir + '/' + self.file_list[index]
        with h5py.File(file_name, 'r') as hf:
            lf = np.array(hf.get('lf'))
        lf = np.transpose(lf, (4, 3, 0, 2, 1))
        lf = torch.from_numpy(lf).to(self.args.device)
        U, V, H, W, _ = lf.shape
        lfimg = rearrange(lf, 'u v c h w -> (u v) c h w')
        lfimg_lr, kernels, sigma, noise_level = self.gen_LR(lfimg.unsqueeze(0))
        data = rearrange(lfimg_lr.squeeze(), '(u v) c h w -> u v c h w', u=U, v=V)
        a0 = (9 - self.args.angRes) // 2
        data = data[a0 : a0 + self.args.angRes, a0 : a0 + self.args.angRes, :, :, :]
        label = lf[a0: a0 + self.args.angRes, a0: a0 + self.args.angRes, :, :, :]
        return data, label, sigma, noise_level.squeeze(0)

    def __len__(self):
        return self.item_num



class TestSetDataLoadertt(Dataset):
    def __init__(self, args):
        super(TestSetDataLoadertt, self).__init__()
        self.args = args
        self.psize = args.patch_size
        self.file_list = []
        self.testset_dir = args.testset_dir
        tmp_list = os.listdir(self.testset_dir)

        self.file_list.extend(tmp_list)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = self.testset_dir + '/' + self.file_list[index]
        with h5py.File(file_name, 'r') as hf:
            lf = np.array(hf.get('lf'))
        # print(lf.shape)
        lf = np.transpose(lf, (4, 3, 0, 2, 1))
        # print(lf.shape)
        _,_,_, H, W = lf.shape
        lf = torch.from_numpy(lf).to(self.args.device)

        a0 = (9 - self.args.angRes) // 2

        x = random.randrange(0, H - self.psize, 8)
        y = random.randrange(0, W - self.psize, 8)
        label = lf[a0: a0 + self.args.angRes, a0: a0 + self.args.angRes, :, x:x + self.psize, y:y + self.psize]


        return label

    def __len__(self):
        return self.item_num
