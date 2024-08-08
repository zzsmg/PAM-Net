import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import time
from PIL import Image
from datasets import NH_HazeDataset

# Hyper Parameters
METHOD = "PAM-DMPHN"

save_dir = 'results/PAM-DMPHN'

GPU = 0

os.makedirs('./test/' + save_dir, exist_ok=True)


def save_dehazed_images(images, iteration):
    filename = './test/' + save_dir + "/" + str(iteration)
    torchvision.utils.save_image(images, filename)


def main():
    print("init data folders")

    encoder_lv1 = models.Encoder().cuda(GPU)
    encoder_lv2 = models.Encoder().cuda(GPU)
    encoder_lv3 = models.Encoder().cuda(GPU)

    decoder_lv1 = models.Decoder().cuda(GPU)
    decoder_lv2 = models.Decoder().cuda(GPU)
    decoder_lv3 = models.Decoder().cuda(GPU)

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl"))
        encoder_lv1.load_state_dict(ckp['model'])

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl"))
        encoder_lv2.load_state_dict(ckp['model'])

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl"))
        encoder_lv3.load_state_dict(ckp['model'])

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl"))
        decoder_lv1.load_state_dict(ckp['model'])

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl"))
        decoder_lv2.load_state_dict(ckp['model'])

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl"))
        decoder_lv3.load_state_dict(ckp['model'])

    test_time = 0.0

    test_dataset = NH_HazeDataset(
        hazed_image_files='/data/Pytorch_Porjects/PAM-Net/PAM-DMPHN/NH-HAZE/4500/test_haze.txt',
        dehazed_image_files='/data/Pytorch_Porjects/PAM-Net/PAM-DMPHN/NH-HAZE/4500/test_clear.txt',
        root_dir='/data/Pytorch_Porjects/PAM-Net/PAM-DMPHN/NH-HAZE/4500/',
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for iteration, images in enumerate(test_dataloader):
        with torch.no_grad():
            start = time.time()
            images_lv1 = Variable(images['hazed_image'] - 0.5).cuda(GPU)
            image_name = images['clear_name'][0]
            print(image_name)
            H = images_lv1.size(2)
            W = images_lv1.size(3)
            images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
            images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]
            images_lv3_1 = images_lv2_1[:, :, :, 0:int(W / 2)]
            images_lv3_2 = images_lv2_1[:, :, :, int(W / 2):W]
            images_lv3_3 = images_lv2_2[:, :, :, 0:int(W / 2)]
            images_lv3_4 = images_lv2_2[:, :, :, int(W / 2):W]

            prior_images_lv1 = Variable(images['prior_image'] - 0.5).cuda(GPU)
            prior_images_lv2_1 = prior_images_lv1[:, :, 0:int(H / 2), :]
            prior_images_lv2_2 = prior_images_lv1[:, :, int(H / 2):H, :]
            prior_images_lv3_1 = prior_images_lv2_1[:, :, :, 0:int(W / 2)]
            prior_images_lv3_2 = prior_images_lv2_1[:, :, :, int(W / 2):W]
            prior_images_lv3_3 = prior_images_lv2_2[:, :, :, 0:int(W / 2)]
            prior_images_lv3_4 = prior_images_lv2_2[:, :, :, int(W / 2):W]

            feature_lv3_1 = encoder_lv3((images_lv3_1, prior_images_lv3_1))
            feature_lv3_2 = encoder_lv3((images_lv3_2, prior_images_lv3_2))
            feature_lv3_3 = encoder_lv3((images_lv3_3, prior_images_lv3_3))
            feature_lv3_4 = encoder_lv3((images_lv3_4, prior_images_lv3_4))

            feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
            feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)

            feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)

            residual_lv3_top = decoder_lv3((feature_lv3_top, prior_images_lv2_1))
            residual_lv3_bot = decoder_lv3((feature_lv3_bot, prior_images_lv2_2))

            feature_lv2_1 = encoder_lv2((images_lv2_1 + residual_lv3_top, prior_images_lv2_1))
            feature_lv2_2 = encoder_lv2((images_lv2_2 + residual_lv3_bot, prior_images_lv2_2))

            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3

            residual_lv2 = decoder_lv2((feature_lv2, prior_images_lv1))

            feature_lv1 = encoder_lv1((images_lv1 + residual_lv2, prior_images_lv1)) + feature_lv2

            dehazed_image = decoder_lv1((feature_lv1, prior_images_lv1))

            stop = time.time()
            test_time += stop - start

            save_dehazed_images(dehazed_image.data + 0.5, image_name)


if __name__ == '__main__':
    main()
