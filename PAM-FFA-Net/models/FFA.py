import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os


# bias=True, 偏置项b
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        # conv+relu+conv
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        res = self.body(x[0]).mul(self.res_scale)
        res += x[0]  # feature maps
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        # print("scale",scale.mean(),"shift",shift.mean())
        # exit()
        res = res * (scale + 1) + shift
        return (res, x[1])


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),  # (c, c/8)
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),  # (c/8, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        res = x * y
        return res


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 一个自适应平均池化层，用于将输入数据的空间维度缩减到 1x1
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),  # (64, 8, 1)
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),  # (8, 64, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        res = x * y
        return res


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)  # (b,1,1,1)

        # 补充
        self.SFT_scale_conv0 = nn.Conv2d(64, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(64, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):  # 逻辑正确
        res = self.act1(self.conv1(x[0]))
        res = res + x[0]
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x[0]
        # inplace=True表示直接在原地修改输入数据，而不需要额外的空间来存储输出
        # 在每个B块后边嵌入参数自适应
        # scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))  # （2，64，240， 240）
        # shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        # res = res * (scale + 1) + shift
        return res, x[1]


# 自适应动态B块
class Block_dt(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block_dt, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)  # (b,1,1,1)

        # 补充
        self.SFT_scale_conv0 = nn.Conv2d(64, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(64, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

        # 初始化为0
        # nn.init.zeros_(self.SFT_shift_conv0.weight)
        # nn.init.zeros_(self.SFT_shift_conv1.weight)
        # nn.init.zeros_(self.SFT_scale_conv0.weight)
        # nn.init.zeros_(self.SFT_scale_conv1.weight)

    def forward(self, x):
        res = self.act1(self.conv1(x[0]))
        res = res + x[0]
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x[0]  # B块的F
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        res = res * (scale + 1) + shift  # F = F + F x r + beta 修改
        return res, x[1]


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        self.conv = conv(dim, dim, kernel_size)
        modules = []
        for i in range(blocks):
            if i % 2 == 0:  # 对于偶数索引，我们添加一个Block (10)
                modules.append(Block(conv, dim, kernel_size))
            else:  # 对于奇数索引，我们添加一个Block_dt (9)
                modules.append(Block_dt(conv, dim, kernel_size))
        # modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        self.gp = nn.Sequential(*modules)  # 使用 torch.nn.Sequential 类将列表中的所有层组合在一起，并将结果存储在变量 self.gp 中。

        # 补充
        self.SFT_scale_conv0 = nn.Conv2d(64, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(64, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

        # 初始化为0
        # nn.init.zeros_(self.SFT_shift_conv0.weight)
        # nn.init.zeros_(self.SFT_shift_conv1.weight)
        # nn.init.zeros_(self.SFT_scale_conv0.weight)
        # nn.init.zeros_(self.SFT_scale_conv1.weight)

    def forward(self, x):  # 逻辑正确
        res = self.gp(x)  # return res, x[1]
        res = self.conv(res[0])
        res += x[0]
        # 在每个G块后边嵌入参数自适应
        # scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))  # （2，64，240， 240）
        # shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        # res = res * (scale + 1) + shift
        return res


class FFA(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3  # default=3
        pre_process = [conv(3, self.dim, kernel_size)]  # 3->64,3x3
        assert self.gps == 3  # G group only = 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),  # (192,4,1x1)
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),  # (4,192,1x1)
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, xx):  # 逻辑正确
        x0 = self.pre(xx[0])  # haze
        x1 = self.pre(xx[1])  # prior
        x = (x0, x1)
        res1 = self.g1(x)[0]  # return res
        res2 = self.g2((res1, x1))[0]
        res3 = self.g3((res2, x1))[0]
        # scale = self.g3(x)[1]
        # shift = self.g3(x)[2]
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + xx[0]


if __name__ == "__main__":
    net = FFA(gps=3, blocks=19)
    print(net)
