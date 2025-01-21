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
import model.deconv_fft as deconv_fft
from model.KernelNet import KernelNet

# import deconv_fft as deconv_fft
# from KernelNet3 import KernelNet


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

# Residual Channel Attention Block (RCAB)
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


class Multi_Context(nn.Module):
    def __init__(self, inchannels):
        super(Multi_Context, self).__init__()
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels * 3, out_channels=inchannels, kernel_size=3, padding=1))

    def forward(self, x):
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x = torch.cat([x1,x2,x3], dim=1)
        x = self.conv2(x)
        return x

class Adaptive_Weight(nn.Module):
    def __init__(self, inchannels):
        super(Adaptive_Weight, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.inchannels = inchannels
        self.fc1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(inchannels//4, 1, kernel_size=1, bias=False)
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.avg(x)
        weight = self.relu1(self.fc1(x_avg))
        weight = self.relu2(self.fc2(weight))
        weight = self.sigmoid(weight)
        out = x * weight
        return out

class SelectiveFusion(nn.Module):
    def __init__(self, inchannels):
        super(SelectiveFusion, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
                                   )

        self.sig = nn.Sigmoid()
        self.mc1 = Multi_Context(inchannels)
        self.mc2 = Multi_Context(inchannels)
        self.ada_w1 = Adaptive_Weight(inchannels)
        self.ada_w2 = Adaptive_Weight(inchannels)

    def forward(self, degra, lf_fea):
        if len(lf_fea.shape)==6:
            b, c, u, v, h, w = lf_fea.shape
            lf_fea = rearrange(lf_fea, 'b c u v h w -> (b u v) c h w', b=b, u=u, v=v)
        mc1 = self.mc1(degra)
        pr1 = lf_fea * self.sig(mc1)
        pr2 = self.conv1(lf_fea)
        pr2 = lf_fea * self.sig(pr2)
        out1 = pr1 + pr2 + lf_fea

        mc2 = self.mc2(lf_fea)
        as1 = degra * self.sig(mc2)
        as2 = self.conv2(degra)
        as2 = degra * self.sig(as2)
        out2 = as1 + as2 + degra

        out1 = self.ada_w1(out1)
        out2 = self.ada_w2(out2)
        out = out1 + out2
        return out

class SAS_conv(nn.Module):
    def __init__(self, fn=64, act='relu'):
        super(SAS_conv, self).__init__()

        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        self.spaconv = nn.Sequential(
                        nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1),
                        CALayer(channel=fn, reduction=16),
                        nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1),
                        )

        self.angconv = nn.Sequential(
                        nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1),
                        CALayer(channel=fn, reduction=16),
                        nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1),
                        )

    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(N * U * V, c, h, w)

        out = self.act(self.spaconv(x))  # [N*U*V,c,h,w]
        out = out.view(N, U * V, c, h * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * h * w, c, U, V)  # [N*h*w,c,U,V]

        out = self.act(self.angconv(out))  # [N*h*w,c,U,V]
        out = out.view(N, h * w, c, U * V)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, U, V, c, h, w)  # [N,U,V,c,h,w]
        out = out.permute(0, 3, 1, 2, 4, 5).contiguous()  # [N,c,U,V,h,w]
        return out

class SAC_conv(nn.Module):
    def __init__(self, fn=64, act='relu', symmetry=True, max_k_size=3):
        super(SAC_conv, self).__init__()

        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        if symmetry:
            k_size_ang = max_k_size
            k_size_spa = max_k_size
        else:
            k_size_ang = max_k_size - 2
            k_size_spa = max_k_size

        self.verconv = nn.Sequential(
                        nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1),
                        # RCAB(n_feat=fn)
                        CALayer(channel=fn, reduction=16),
                        nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1),
                        )

        self.horconv = nn.Sequential(
                        nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1),
                        # RCAB(n_feat=fn)
                        CALayer(channel=fn, reduction=16),
                        nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1),
                        )


    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        # N = N // (self.an * self.an)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N * V * w, c, U, h)

        out = self.act(self.verconv(x))  # [N*V*w,c,U,h]
        out = out.view(N, V * w, c, U * h)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * U * h, c, V, w)  # [N*U*h,c,V,w]

        out = self.act(self.horconv(out))  # [N*U*h,c,V,w]
        out = out.view(N, U * h, c, V * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, V, w, c, U, h)  # [N,V,w,c,U,h]
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()  # [N,c,U,V,h,w]
        return out


class SAV_type1(nn.Module):
    def __init__(self, fn):
        super(SAV_type1, self).__init__()
        self.fusion = SelectiveFusion(fn)
        self.SAS_conv = SAS_conv(fn=fn)
        self.SAC_conv = SAC_conv(fn=fn)
        self.angRes = 5

    def forward(self, x):
        lf_input, degra = x
        feature = self.fusion(degra, lf_input)
        b = int(degra.shape[0]/(self.angRes*self.angRes))
        feature = rearrange(feature, '(b u v) c h w -> b u v c h w', b=b, u=self.angRes, v=self.angRes)
        feature = rearrange(feature, 'b u v c h w -> b c u v h w', b=b, u=self.angRes, v=self.angRes)
        # lf_fea = rearrange(lf_fea, '(b u v) c h w -> b u v c h w', b=b, u=self.angRes, v=self.angRes)
        # N, c, U, V, h, w = x.shape

        feat = self.SAS_conv(feature)
        res = self.SAC_conv(feat)
        return [res, degra]


class SAV_type2(nn.Module):
    def __init__(self, fn):
        super(SAV_type2, self).__init__()
        self.fusion = SelectiveFusion(fn)
        self.SAS_conv = SAS_conv(fn=fn)
        self.SAC_conv = SAC_conv(fn=fn)
        self.angRes = 5

    def forward(self, x):
        lf_input, degra = x
        feature = self.fusion(degra, lf_input)
        b = int(degra.shape[0]/(self.angRes*self.angRes))
        feature = rearrange(feature, '(b u v) c h w -> b u v c h w', b=b, u=self.angRes, v=self.angRes)
        feature = rearrange(feature, 'b u v c h w -> b c u v h w', b=b, u=self.angRes, v=self.angRes)
        # lf_fea = rearrange(lf_fea, '(b u v) c h w -> b u v c h w', b=b, u=self.angRes, v=self.angRes)
        # N, c, U, V, h, w = x.shape

        res = self.SAS_conv(feature)+self.SAC_conv(feature)+feature
        return [res, degra]

class SAV_type3(nn.Module):
    def __init__(self, fn):
        super(SAV_type3, self).__init__()
        self.fusion = SelectiveFusion(fn)
        self.SAS_conv = SAS_conv(fn=fn)
        self.SAC_conv = SAC_conv(fn=fn)
        self.angRes = 5

    def forward(self, x):
        lf_input, degra = x
        feature = self.fusion(degra, lf_input)
        b = int(degra.shape[0]/(self.angRes*self.angRes))
        feature = rearrange(feature, '(b u v) c h w -> b u v c h w', b=b, u=self.angRes, v=self.angRes)
        feature = rearrange(feature, 'b u v c h w -> b c u v h w', b=b, u=self.angRes, v=self.angRes)
        # lf_fea = rearrange(lf_fea, '(b u v) c h w -> b u v c h w', b=b, u=self.angRes, v=self.angRes)
        # N, c, U, V, h, w = x.shape

        res = self.SAC_conv(self.SAS_conv(feature))+feature
        return [res, degra]

class SAV_type4(nn.Module):
    def __init__(self, fn):
        super(SAV_type4, self).__init__()
        self.fusion = SelectiveFusion(fn)
        self.SAS_conv = SAS_conv(fn=fn)
        self.SAC_conv = SAC_conv(fn=fn)
        self.angRes = 5

    def forward(self, x):
        lf_input, degra = x
        feature = self.fusion(degra, lf_input)
        b = int(degra.shape[0]/(self.angRes*self.angRes))
        feature = rearrange(feature, '(b u v) c h w -> b u v c h w', b=b, u=self.angRes, v=self.angRes)
        feature = rearrange(feature, 'b u v c h w -> b c u v h w', b=b, u=self.angRes, v=self.angRes)
        # lf_fea = rearrange(lf_fea, '(b u v) c h w -> b u v c h w', b=b, u=self.angRes, v=self.angRes)
        # N, c, U, V, h, w = x.shape

        res = self.SAS_conv(feature)+feature
        res = self.SAC_conv(res)+res

        return [res, degra]


class ReconNet(nn.Module):
    def __init__(self, channel=64, factor=4, angRes=5,ksize=21,layer=6,kernel_pretrain=None):
        super(ReconNet, self).__init__()
        self.channels = channel
        self.angRes = angRes
        self.ksize = ksize
        self.factor = factor
        self.layer = layer
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3*3, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        self.kernelPrediction = KernelNet(ksize=self.ksize, channel=self.channels, angR=self.angRes)


        if kernel_pretrain != None:
            print('Loading KernelNet pretrain model from {}'.format(kernel_pretrain))
            model = torch.load(kernel_pretrain)
            self.kernelPrediction.load_state_dict(model['state_dict'], strict=True)
            print('success')
            

        self.degradation_conv = nn.Sequential(
            nn.Conv2d(self.ksize*self.ksize+3, self.channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1))
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels*2, self.channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1))

        alt_blocks = [SAV_type2(fn=self.channels) for _ in range(self.layer)]
        self.backbone = nn.Sequential(*alt_blocks)

        self.CondNet = nn.Sequential(
            nn.Conv2d(3*self.factor*self.factor, self.channels, 5, 1, 4, dilation=2), nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, 3, 1, 2, dilation=2), nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, 1), nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, 1)
        )
        
        self.up_sample = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * factor ** 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.PixelShuffle(factor),
            nn.Conv2d(self.channels, 3, kernel_size=1, stride=1, padding=0, bias=False))

    def spatial2depth(self, spatial, scale):
        depth_list = []
        for i in range(scale):
            for j in range(scale):
                depth_list.append(spatial[:, :, i::scale, j::scale])
        depth = torch.cat(depth_list, dim=1)
        return depth

    def forward(self, lf):
        b, u, v, c, h, w = lf.shape

        kernels, noise = self.kernelPrediction(lf)
        lf_batch = rearrange(lf, 'b u v c h w -> (b u v) c h w', b=b, u=self.angRes, v=self.angRes)
        noise_free = lf_batch-noise

        deconv = deconv_fft.deconv_batch(lf_batch, kernels, self.factor)
        deconv_S2D = self.CondNet(self.spatial2depth(deconv, self.factor))

        bk, ck, hk, wk = kernels.shape
        ker_code_exp = kernels.view((bk, ck, hk, wk, 1, 1)).\
            expand((bk, ck, hk, wk, h, w)).\
            view((bk, ck, hk*wk, h, w)).squeeze()  # kernel_map stretch
        # print(deconv.shape, deconv_S2D.shape,kernels.shape,ker_code_exp.shape)
        deg_rep = self.degradation_conv(torch.cat((ker_code_exp,noise),dim=1))


        lf_fea = self.initial_conv(torch.cat((lf_batch,noise_free,noise_free),dim=1))
        # print(deconv_S2D.shape,lf_fea.shape)
        feainput = self.conv(torch.cat((deconv_S2D,lf_fea),dim=1))

        x = rearrange(lf, 'b u v c h w -> (b u v) c h w')
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        res = self.backbone([feainput,deg_rep])
        buffer = rearrange(res[0], 'b c u v h w -> (b u v) c h w', b=b, u=u, v=v)
        out = self.up_sample(buffer)+x_upscale
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        return out

if __name__ == "__main__":
    # angRes = 5
    # factor = 4
    # net = Net(factor, angRes)
 
    # from thop import profile
    # input_lf = torch.randn(4, angRes, angRes, 3, 32, 32)
    # blur = torch.randn(4, 1, angRes, angRes)
    # noise = torch.randn(4, 1, angRes, angRes)
    # total = sum([param.nelement() for param in net.parameters()])
    # flops, params = profile(net, inputs=((input_lf, blur, noise), ))

    # print('   Number of parameters: %.2fM' % (params / 1e6))
    # print('   Number of FLOPs: %.2fG' % (flops / 1e9))
    net = ReconNet().cuda()
    a = torch.rand(1,64,32,32).cuda()
    b = torch.rand(1,25,64,32,32).cuda()
    k = torch.rand(1, 25, 441).cuda()
    lf = torch.rand(1, 5,5,3,32,32).cuda()
    out = net(lf)
    # print(out.shape)
