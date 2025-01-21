# @Time = 2021.12.23
# @Author = Zhen
# 基于zhen的distangle的思路做，把SAV的分支用LKA实现。
"""
Spatial SR for light fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.init as init
from einops import rearrange
import torch.nn.functional as F
import math


################################################
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self, n_feat=64, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(False), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = res + x
        return res


class RB(nn.Module):
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.ReLU()
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x


class KernelNet(nn.Module):
    def __init__(self, ksize=21, channel=64, angR=5):
        super(KernelNet, self).__init__()
        self.ksize = ksize
        self.channel = channel
        self.in_conv = nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=3, padding=1, bias=True)

        self.Side2Center1 = Side2CenterFusion(dim=channel,angRes=angR)
        self.Center2Side1 = Center2SideFusion(dim=channel,angRes=angR)

        self.body1 = nn.Sequential(
            # RCAB(n_feat=channel),
            RCAB(n_feat=channel))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.ksize))
        self.body2 = nn.Sequential(
            # RCAB(n_feat=channel),
            RCAB(n_feat=channel),
            nn.Conv2d(channel, 16, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True))
        self.fc_net = nn.Sequential(
            nn.Linear(ksize * ksize, 800, bias=True),
            nn.Linear(800, ksize * ksize, bias=True),
            nn.Softmax()
            )
        
        self.bodyn2 = nn.Sequential(
            # RCAB(n_feat=channel),
            RCAB(n_feat=channel),
            nn.Conv2d(channel, 16, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True))

    def forward(self, lf):
        b, u, v, c, h, w = lf.shape
        x = rearrange(lf, 'b u v c h w -> (b u v) c h w')
        
        buffer = self.in_conv(x)
        buffer = rearrange(buffer, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        center = buffer[:,2,2,:,:,:]
        center = self.Side2Center1(center, buffer.view(b, -1, self.channel, h, w))
        buffer = self.Center2Side1(center, buffer.view(b, -1, self.channel, h, w))
        # center = self.Side2Center2(center, buffer.view(b, -1, self.channel, h, w))
        # buffer = self.Center2Side2(center, buffer.view(b, -1, self.channel, h, w))
        ker_fea = buffer.view(b, u, v, self.channel, h, w)
        ker_fea = rearrange(ker_fea, 'b u v c h w -> (b u v) c h w', b=b, u=u, v=v)


        b, c, h, w = ker_fea.size()
        out = self.body1(ker_fea)

        noise = self.bodyn2(out)

        out = self.global_pool(out)
        out = self.body2(out)
        out = out.view(b, self.ksize * self.ksize)
        out = self.fc_net(out)
        est_kernel = out.view(b, 1, self.ksize, self.ksize)
        # torch.Size([25, 1, 21, 21]) torch.Size([25, 1, 48, 48])
        return est_kernel, noise


class Center2SideFusion(nn.Module):
    def __init__(self, dim, angRes):
        super(Center2SideFusion, self).__init__()
        self.angRes = angRes
        self.fusion = nn.Sequential(
                        nn.Conv2d(2*dim, dim, kernel_size=1, padding=0),
                        RCAB(n_feat=dim),
                        RCAB(n_feat=dim),
                        nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        self.conv_sharing = nn.Conv2d(self.angRes*self.angRes*dim, self.angRes*self.angRes*dim, kernel_size=1, stride=1, padding=0)

    def forward(self, center, side):
        b, n, c, h, w = side.shape
        side_kernel_feas = []
        for i in range(n):
            current_side = side[:, i, :, :, :].contiguous()
            kernel_fea = self.fusion(torch.cat((center, current_side),dim=1))
            side_kernel_feas.append(kernel_fea)
        side_kernel_feas = torch.cat(side_kernel_feas, dim=1)
        kernel_fea = self.conv_sharing(side_kernel_feas)
        kernel_fea = kernel_fea.unsqueeze(1).contiguous().view(b, -1, c, h, w)
        return kernel_fea

class Side2CenterFusion(nn.Module):
    def __init__(self, dim, angRes):
        super(Side2CenterFusion, self).__init__()
        self.angRes = angRes
        self.fusion = self.fusion = nn.Sequential(
                        nn.Conv2d(2*dim, dim, kernel_size=1, padding=0),
                        RCAB(n_feat=dim),
                        RCAB(n_feat=dim),
                        nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        self.conv_f1 = nn.Conv2d(angRes*angRes*dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, center, side):
        b, n, c, h, w = side.shape
        side = side.view(b,-1,h,w)
        side_fea = self.conv_f1(side)
        kernel_fea = self.fusion(torch.cat((side_fea, center),dim=1))
        return kernel_fea

class Net(nn.Module):
    def __init__(self, channel, factor, angRes,ksize, fsize):
        super(Net, self).__init__()
        self.channels = channel
        self.angRes = angRes
        self.ksize = ksize
        self.factor = factor
        self.fsize = fsize
        
        self.initial_conv = nn.Conv2d(3, self.channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.kernelPrediction = KernelNet(ksize=self.ksize, channel=self.channels, angR=self.angRes)
   
    def forward(self, lf):
        b, u, v, c, h, w = lf.shape
        x = rearrange(lf, 'b u v c h w -> (b u v) c h w')

        kernels, noise = self.kernelPrediction(lf)
        
        return kernels, noise
        
    def conv_func(self, input, kernel, padding='same'):
        b, c, h, w = input.size()
        assert b == 1, "only support b=1!"
        _, _, ksize, ksize = kernel.size()
        if padding == 'same':
            pad = ksize // 2
        elif padding == 'valid':
            pad = 0
        else:
            raise Exception("not support padding flag!")

        conv_result_list = []
        for i in range(c):
            conv_result_list.append(F.conv2d(input[:, i:i + 1, :, :], kernel, bias=None, stride=1, padding=pad))
        conv_result = torch.cat(conv_result_list, dim=1)
        return conv_result

    def blur_down(self, x, kernel, scale):
        b, c, h, w = x.size()
        _, kc, ksize, _ = kernel.size()
        psize = ksize // 2
        assert kc == 1, "only support kc=1!"

        # blur
        x = F.pad(x, (psize, psize, psize, psize), mode='replicate')
        blur_list = []
        for i in range(b):
            blur_list.append(self.conv_func(x[i:i + 1, :, :, :], kernel[i:i + 1, :, :, :]))
        blur = torch.cat(blur_list, dim=0)
        blur = blur[:, :, psize:-psize, psize:-psize]

        # down
        blurdown = blur[:, :, ::scale, ::scale]

        return blurdown


if __name__ == "__main__":
    net = KernelNet().cuda()
    pass