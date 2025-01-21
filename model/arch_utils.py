import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True)
        )

def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation),
        nn.ReLU(inplace=True)
    )

def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    
class LargeKernel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.spatial_gating_unit = LargeKernel(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class basicBlock1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = normalize(dim)
        self.norm2 = normalize(dim)
        self.mlp = Mlp(dim, dim)
        self.attn = Attention(dim)

    def forward(self, x):
        x1 = self.attn(self.norm1(x)) + x
        x2 = self.mlp(self.norm2(x)) + x1
        return x2


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        out_features = in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.LeakyReLU(0.1, inplace=True)
        # self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            # nn.BatchNorm2d(out_features, eps=1e-5),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.conv2(x)
        return x




class Unet_extractor_HR(nn.Module):
    def __init__(self, in_channels):
        super(Unet_extractor_HR, self).__init__()

        # encoder
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = ResidualBlockNoBN(num_feat=32)
        self.conv1_3 = ResidualBlockNoBN(num_feat=32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = ResidualBlockNoBN(num_feat=64)
        self.conv2_3 = ResidualBlockNoBN(num_feat=64)

        self.conv3_1 = ResidualBlockNoBN(num_feat=64)
        self.conv3_2 = ResidualBlockNoBN(num_feat=64)
        self.conv3_3 = ResidualBlockNoBN(num_feat=64)
        self.conv3_4 = ResidualBlockNoBN(num_feat=64)

        # decoder
        self.upconv3_i = nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1)
        self.upconv3_2 = ResidualBlockNoBN(num_feat=64)
        self.upconv3_1 = ResidualBlockNoBN(num_feat=64)

        # self.upconv2_u = upconv(64, 32)
        self.upconv2_u = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.upconv2_i = nn.Conv2d(64, 16, kernel_size=3,stride=1, padding=1)
        self.upconv2_2 = ResidualBlockNoBN(num_feat=16)
        self.upconv2_1 = ResidualBlockNoBN(num_feat=16)

    def forward(self, x):

        # encoder
        conv1 = self.relu(self.conv1_3(self.conv1_2(self.conv1_1(x))))
        conv2 = self.relu(self.conv2_3(self.conv2_2(self.conv2_1(conv1))))
        conv2 = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(conv2))))

        # decoder
        upconv2 = self.relu(self.upconv3_1(self.upconv3_2(self.upconv3_i(conv2))))
        upconv2 = self.upconv2_u(upconv2)
        cat2 = self.upconv2_i(torch.cat((upconv2,conv1),dim=1))
        upconv1 = self.relu(self.upconv2_1(self.upconv2_2(cat2)))
        return upconv1


class Unet_extractor_LR(nn.Module):
    def __init__(self, in_channels):
        super(Unet_extractor_LR, self).__init__()
        # encoder
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = ResidualBlockNoBN(num_feat=32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = ResidualBlockNoBN(num_feat=64)

        self.conv3_1 = ResidualBlockNoBN(num_feat=64)
        self.conv3_2 = ResidualBlockNoBN(num_feat=64)

        # decoder
        self.upconv3_i = nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1)
        self.upconv3_1 = ResidualBlockNoBN(num_feat=64)

        # self.upconv2_u = upconv(64, 32)
        self.upconv2_u = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.upconv2_i = nn.Conv2d(64, 16, kernel_size=3,stride=1, padding=1)
        self.upconv2_1 = ResidualBlockNoBN(num_feat=16)

    def forward(self, x):

        # encoder
        conv1 = self.relu(self.conv1_2(self.conv1_1(x)))
        conv2 = self.relu(self.conv2_2(self.conv2_1(conv1)))
        conv2 = self.conv3_2(self.conv3_1(conv2))

        # decoder
        upconv2 = self.relu(self.upconv3_1(self.upconv3_i(conv2)))
        upconv2 = self.upconv2_u(upconv2)
        cat2 = self.upconv2_i(torch.cat((upconv2,conv1),dim=1))
        upconv1 = self.relu(self.upconv2_1(cat2))
        return upconv1
