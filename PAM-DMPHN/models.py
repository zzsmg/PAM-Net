import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


class Encoder(nn.Module):
    def __init__(self, conv=default_conv):
        super(Encoder, self).__init__()
        self.dim = 64
        kernel_size = 1

        pre_process_32 = [conv(3, 16, kernel_size)]
        pre_process_64 = [conv(3, 32, kernel_size)]
        pre_process_128 = [conv(3, 64, kernel_size)]
        self.pre_32 = nn.Sequential(*pre_process_32)
        self.pre_64 = nn.Sequential(*pre_process_64)
        self.pre_128 = nn.Sequential(*pre_process_128)

        self.SFT_scale_conv0_1 = nn.Conv2d(16, 16, 1)
        self.SFT_scale_conv1_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.SFT_shift_conv0_1 = nn.Conv2d(16, 16, 1)
        self.SFT_shift_conv1_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.SFT_scale_conv0_2 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.SFT_shift_conv0_2 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.SFT_scale_conv0_3 = nn.Conv2d(64, 64, 1)
        self.SFT_scale_conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=4, padding=1)
        self.SFT_shift_conv0_3 = nn.Conv2d(64, 64, 1)
        self.SFT_shift_conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=4, padding=1)

        self.SFT_shift_conv0_1.apply(weight_init)
        self.SFT_shift_conv0_2.apply(weight_init)
        self.SFT_shift_conv0_3.apply(weight_init)
        self.SFT_shift_conv1_1.apply(weight_init)
        self.SFT_shift_conv1_2.apply(weight_init)
        self.SFT_shift_conv1_3.apply(weight_init)
        self.SFT_scale_conv0_1.apply(weight_init)
        self.SFT_scale_conv0_2.apply(weight_init)
        self.SFT_scale_conv0_3.apply(weight_init)
        self.SFT_scale_conv1_1.apply(weight_init)
        self.SFT_scale_conv1_2.apply(weight_init)
        self.SFT_scale_conv1_3.apply(weight_init)


        # Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )

        # Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )

        # Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

    def forward(self, xx):
        x = xx[0]  # (8,3,60,80)
        x_32 = self.pre_32(xx[1])  # prior(8,32,60,80)
        x_64 = self.pre_64(xx[1])
        x_128 = self.pre_128(xx[1])
        # Conv1
        scale_1 = self.SFT_scale_conv1_1(F.leaky_relu(self.SFT_scale_conv0_1(x_32), 0.1, inplace=True))  # (8,32,60,80)
        shift_1 = self.SFT_shift_conv1_1(F.leaky_relu(self.SFT_shift_conv0_1(x_32), 0.1, inplace=True))  # (8,32,60,80)
        x = self.layer1(x)  # (8,32,60,80)

        x = x * (scale_1 + 1) + shift_1

        x = self.layer2(x) + x  # (8,32,60,80)

        x = x * (scale_1 + 1) + shift_1

        x = self.layer3(x) + x  # (8,32,60,80)

        x = x * (scale_1 + 1) + shift_1

        # Conv2
        scale_2 = self.SFT_scale_conv1_2(F.leaky_relu(self.SFT_scale_conv0_2(x_64), 0.1, inplace=True))  # (8,64,30,40)
        shift_2 = self.SFT_shift_conv1_2(F.leaky_relu(self.SFT_shift_conv0_2(x_64), 0.1, inplace=True))  # (8,64,30,40)
        x = self.layer5(x)  # (8,64,30,40)
        # print("scale", scale.mean(), "shift", shift.mean())

        x = x * (scale_2 + 1) + shift_2

        x = self.layer6(x) + x  # (8,64,30,40)

        x = x * (scale_2 + 1) + shift_2

        x = self.layer7(x) + x  # (8,64,30,40)

        x = x * (scale_2 + 1) + shift_2

        # Conv3
        scale_3 = self.SFT_scale_conv1_3(
            F.leaky_relu(self.SFT_scale_conv0_3(x_128), 0.1, inplace=True))  # (8,128,15,20)
        shift_3 = self.SFT_shift_conv1_3(
            F.leaky_relu(self.SFT_shift_conv0_3(x_128), 0.1, inplace=True))  # (8,128,15,20)
        x = self.layer9(x)  # (8,128,15,20)

        x = x * (scale_3 + 1) + shift_3

        x = self.layer10(x) + x  # (8,128,15,20)

        x = x * (scale_3 + 1) + shift_3

        x = self.layer11(x) + x  # (8,128,15,20)

        x = x * (scale_3 + 1) + shift_3
        return x


class Decoder(nn.Module):
    def __init__(self, conv=default_conv):
        super(Decoder, self).__init__()
        self.dim = 64
        kernel_size = 1

        pre_process_32 = [conv(3, 16, kernel_size)]
        pre_process_64 = [conv(3, 32, kernel_size)]
        pre_process_128 = [conv(3, 64, kernel_size)]
        self.pre_32 = nn.Sequential(*pre_process_32)
        self.pre_64 = nn.Sequential(*pre_process_64)
        self.pre_128 = nn.Sequential(*pre_process_128)

        self.SFT_scale_conv0_1 = nn.Conv2d(16, 16, 1)
        self.SFT_scale_conv1_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.SFT_shift_conv0_1 = nn.Conv2d(16, 16, 1)
        self.SFT_shift_conv1_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.SFT_scale_conv0_2 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.SFT_shift_conv0_2 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.SFT_scale_conv0_3 = nn.Conv2d(64, 64, 1)
        self.SFT_scale_conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=4, padding=1)
        self.SFT_shift_conv0_3 = nn.Conv2d(64, 64, 1)
        self.SFT_shift_conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=4, padding=1)

        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        '''nn.ConvTranspose2d是一种特殊的卷积操作，也被称为转置卷积或反卷积，它可以增大特征图的尺寸'''
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

        # Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.SFT_shift_conv0_1.apply(weight_init)
        self.SFT_shift_conv0_2.apply(weight_init)
        self.SFT_shift_conv0_3.apply(weight_init)
        self.SFT_shift_conv1_1.apply(weight_init)
        self.SFT_shift_conv1_2.apply(weight_init)
        self.SFT_shift_conv1_3.apply(weight_init)
        self.SFT_scale_conv0_1.apply(weight_init)
        self.SFT_scale_conv0_2.apply(weight_init)
        self.SFT_scale_conv0_3.apply(weight_init)
        self.SFT_scale_conv1_1.apply(weight_init)
        self.SFT_scale_conv1_2.apply(weight_init)
        self.SFT_scale_conv1_3.apply(weight_init)


    def forward(self, xx):
        x = xx[0]  # (8,128,15,40)
        x_32 = self.pre_32(xx[1])  # prior(8,32,60,160)
        x_64 = self.pre_64(xx[1])  # prior(8,64,60,160)
        x_128 = self.pre_128(xx[1])  # prior(8,128,60,160)

        # Deconv3
        scale_3 = self.SFT_scale_conv1_3(
            F.leaky_relu(self.SFT_scale_conv0_3(x_128), 0.1, inplace=True))  # (8,128,15,40)
        shift_3 = self.SFT_shift_conv1_3(F.leaky_relu(self.SFT_shift_conv0_3(x_128), 0.1, inplace=True))
        x = self.layer13(x) + x  # (8,128,15,40)

        x = x * (scale_3 + 1) + shift_3

        x = self.layer14(x) + x  # (8,128,15,40)

        x = x * (scale_3 + 1) + shift_3

        x = self.layer16(x)  # (8,64,30,80)

        scale_2 = self.SFT_scale_conv1_2(F.leaky_relu(self.SFT_scale_conv0_2(x_64), 0.1, inplace=True))  # (8,64,30,80)
        shift_2 = self.SFT_shift_conv1_2(F.leaky_relu(self.SFT_shift_conv0_2(x_64), 0.1, inplace=True))  # (8,64,30,80)

        x = x * (scale_2 + 1) + shift_2

        # Deconv2
        x = self.layer17(x) + x  # (8,64,30,80)

        x = x * (scale_2 + 1) + shift_2

        x = self.layer18(x) + x  # (8,64,30,80)

        x = x * (scale_2 + 1) + shift_2

        x = self.layer20(x)  # (8,32,60,160)

        scale_1 = self.SFT_scale_conv1_1(F.leaky_relu(self.SFT_scale_conv0_1(x_32), 0.1, inplace=True))  # (8,32,60,160)
        shift_1 = self.SFT_shift_conv1_1(F.leaky_relu(self.SFT_shift_conv0_1(x_32), 0.1, inplace=True))  # (8,32,60,160)

        x = x * (scale_1 + 1) + shift_1

        # Deconv1

        x = self.layer21(x) + x  # (8,32,60,160)

        x = x * (scale_1 + 1) + shift_1

        x = self.layer22(x) + x  # (8,32,60,160)

        x = x * (scale_1 + 1) + shift_1

        x = self.layer24(x)  # (8,3,60,160)

        return x
