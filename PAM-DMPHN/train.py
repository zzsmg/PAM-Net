import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import psnr, ssim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datasets import NH_HazeDataset
import time
from loss import CustomLoss_function
import matplotlib.pyplot as plt

'''python DMPHN_train.py --continue_train or retrain '''
parser = argparse.ArgumentParser(description="PAM-DMPHN")
parser.add_argument("-e", "--epochs", type=int, default=2400)
parser.add_argument("-se", "--start_epoch", type=int, default=0)
parser.add_argument("-b", "--batchsize", type=int, default=8)
parser.add_argument("-s", "--imagesize", type=int, default=60)
parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters
METHOD = "PAM-DMPHN"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize


def save_dehazed_images(images, iteration, epoch):
    filename = './checkpoints/' + METHOD + "/epoch" + str(epoch) + "/" + "Iter_" + str(iteration) + "_dehazed.png"
    torchvision.utils.save_image(images, filename)


def lr_schedule_cosdecay(t, T, init_lr=LEARNING_RATE):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


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


def main():
    global start_epoch
    start_epoch = 0
    losses = []
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    losses_val = []

    encoder_lv1 = models.Encoder()
    encoder_lv2 = models.Encoder()
    encoder_lv3 = models.Encoder()

    decoder_lv1 = models.Decoder()
    decoder_lv2 = models.Decoder()
    decoder_lv3 = models.Decoder()

    # encoder_lv1.apply(weight_init).cuda(GPU)
    # encoder_lv2.apply(weight_init).cuda(GPU)
    # encoder_lv3.apply(weight_init).cuda(GPU)
    #
    # decoder_lv1.apply(weight_init).cuda(GPU)
    # decoder_lv2.apply(weight_init).cuda(GPU)
    # decoder_lv3.apply(weight_init).cuda(GPU)

    encoder_lv1.cuda(GPU)
    encoder_lv2.cuda(GPU)
    encoder_lv3.cuda(GPU)

    decoder_lv1.cuda(GPU)
    decoder_lv2.cuda(GPU)
    decoder_lv3.cuda(GPU)

    encoder_lv1_optim = torch.optim.Adam(encoder_lv1.parameters(), lr=LEARNING_RATE)
    # encoder_lv1_scheduler = StepLR(encoder_lv1_optim, step_size=10, gamma=0.1)
    encoder_lv2_optim = torch.optim.Adam(encoder_lv2.parameters(), lr=LEARNING_RATE)
    # encoder_lv2_scheduler = StepLR(encoder_lv2_optim, step_size=10, gamma=0.1)
    encoder_lv3_optim = torch.optim.Adam(encoder_lv3.parameters(), lr=LEARNING_RATE)
    # encoder_lv3_scheduler = StepLR(encoder_lv3_optim, step_size=10, gamma=0.1)

    decoder_lv1_optim = torch.optim.Adam(decoder_lv1.parameters(), lr=LEARNING_RATE)
    # decoder_lv1_scheduler = StepLR(decoder_lv1_optim, step_size=10, gamma=0.1)
    decoder_lv2_optim = torch.optim.Adam(decoder_lv2.parameters(), lr=LEARNING_RATE)
    # decoder_lv2_scheduler = StepLR(decoder_lv2_optim, step_size=10, gamma=0.1)
    decoder_lv3_optim = torch.optim.Adam(decoder_lv3.parameters(), lr=LEARNING_RATE)
    # decoder_lv3_scheduler = StepLR(decoder_lv3_optim, step_size=10, gamma=0.1)

    # start_epoch = 0
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl"))
        my_model_dict = encoder_lv1.state_dict()
        pretrained_dict = {k: v for k, v in ckp.items() if k in my_model_dict}
        my_model_dict.update(pretrained_dict)
        encoder_lv1.load_state_dict(my_model_dict)
        print("load encoder_lv1 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl"))
        my_model_dict = encoder_lv2.state_dict()
        pretrained_dict = {k: v for k, v in ckp.items() if k in my_model_dict}
        my_model_dict.update(pretrained_dict)
        encoder_lv2.load_state_dict(my_model_dict)
        print("load encoder_lv2 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl"))
        my_model_dict = encoder_lv3.state_dict()
        pretrained_dict = {k: v for k, v in ckp.items() if k in my_model_dict}
        my_model_dict.update(pretrained_dict)
        encoder_lv3.load_state_dict(my_model_dict)
        print("load encoder_lv3 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl"))
        my_model_dict = decoder_lv1.state_dict()
        pretrained_dict = {k: v for k, v in ckp.items() if k in my_model_dict}
        my_model_dict.update(pretrained_dict)
        decoder_lv1.load_state_dict(my_model_dict)
        print("load decoder_lv1 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
        ckp = torch.load(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl"))
        my_model_dict = decoder_lv2.state_dict()
        pretrained_dict = {k: v for k, v in ckp.items() if k in my_model_dict}
        my_model_dict.update(pretrained_dict)
        decoder_lv2.load_state_dict(my_model_dict)
        print("load decoder_lv2 success")

    if os.path.exists(str('./checkpoints/' + METHOD + '/decoder_lv3.pkl')):
        ckp = torch.load('./checkpoints/' + METHOD + '/decoder_lv3.pkl')
        my_model_dict = decoder_lv3.state_dict()
        pretrained_dict = {k: v for k, v in ckp.items() if k in my_model_dict}
        my_model_dict.update(pretrained_dict)
        decoder_lv3.load_state_dict(my_model_dict)
        print("load decoder_lv3 success")

    if not os.path.exists('./checkpoints/' + METHOD):
        os.system('mkdir ./checkpoints/' + METHOD)

    for epoch in range(start_epoch, EPOCHS):
        print(f"Training from {start_epoch}")
        train_dataset = NH_HazeDataset(
            hazed_image_files='/data/Pytorch_Porjects/PAM-Net/PAM-DMPHN/NH-HAZE/4500/train_haze.txt',
            # make changes here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            dehazed_image_files='/data/Pytorch_Porjects/PAM-Net/PAM-DMPHN/NH-HAZE/4500/train_clear.txt',
            root_dir='/data/Pytorch_Porjects/PAM-Net/PAM-DMPHN/NH-HAZE/4500/',
            crop=False,
            rotation=False,
            crop_size=IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        epoch_avg_loss = []
        T = len(train_dataloader) * EPOCHS
        for iteration, images in enumerate(train_dataloader):

            t = epoch * len(train_dataloader) + iteration

            custom_loss_fn = CustomLoss_function().cuda(GPU)  # 0.6L1+0.4L2+0.006per+2e-8TV

            gt = Variable(images['dehazed_image'] - 0.5).cuda(GPU)

            H = gt.size(2)
            W = gt.size(3)

            images_lv1 = Variable(images['hazed_image'] - 0.5).cuda(GPU)
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

            # 第三层特征图
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

            loss_lv1 = custom_loss_fn(dehazed_image, gt)


            loss = loss_lv1

            encoder_lv1.zero_grad()
            encoder_lv2.zero_grad()
            encoder_lv3.zero_grad()

            decoder_lv1.zero_grad()
            decoder_lv2.zero_grad()
            decoder_lv3.zero_grad()

            loss.backward()

            encoder_lv1_optim.step()
            encoder_lv2_optim.step()
            encoder_lv3_optim.step()

            decoder_lv1_optim.step()
            decoder_lv2_optim.step()
            decoder_lv3_optim.step()
            epoch_avg_loss.append(loss.item())
            lr_current = encoder_lv1_optim.param_groups[0]['lr']
            if (iteration + 1) % 10 == 0:
                print(
                    f'Epoch: {epoch}, iteration: {iteration + 1}, lr: {lr_current:.7f}, loss: {loss.item():.5f}, {t} / {T}')
        avg_loss = np.mean(epoch_avg_loss)
        losses.append(avg_loss)
        plt.subplot(3, 1, 1)
        plt.plot(losses)
        plt.title('train Losses')
        plt.xlabel(f'{epoch}')
        fig = plt.gcf()
        fig.set_size_inches(16, 8)
        plt.pause(0.1)

        psnr_5val = []
        ssim_5val = []
        loss_5val = []
        if (epoch) % 5 == 0:
            if os.path.exists('./checkpoints/' + METHOD + '/epoch' + str(epoch)) == False:
                os.system('mkdir ./checkpoints/' + METHOD + '/epoch' + str(epoch))

            print("Testing...")
            test_dataset = NH_HazeDataset(
                hazed_image_files='/data/Pytorch_Porjects/PAM-Net/PAM-DMPHN/NH-HAZE/4500/val_haze.txt',
                dehazed_image_files='/data/Pytorch_Porjects/PAM-Net/PAM-DMPHN/NH-HAZE/4500/val_clear.txt',
                root_dir='/data/Pytorch_Porjects/PAM-Net/PAM-DMPHN/NH-HAZE/4500/',
                transform=transforms.Compose([
                    transforms.ToTensor()
                ]))
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            test_time = 0.0
            for iteration, images in enumerate(test_dataloader):
                with torch.no_grad():
                    start = time.time()
                    custom_loss_fn = CustomLoss_function().cuda(GPU)
                    gt = Variable(images['dehazed_image'] - 0.5).cuda(GPU)
                    images_lv1 = Variable(images['hazed_image'] - 0.5).cuda(GPU)
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

                    loss_val_lv1 = custom_loss_fn(dehazed_image, gt)
                    loss_val = loss_val_lv1.item()
                    loss_5val.append(loss_val)
                    ssim1 = ssim(dehazed_image, gt).item()
                    psnr1 = psnr(dehazed_image, gt)
                    psnr_5val.append(psnr1)
                    ssim_5val.append(ssim1)
                    print('RunTime:%.4f' % (stop - start), '  Average Runtime:%.4f' % (test_time / (iteration + 1)))
                    print(f'name: {iteration}, psnr: {psnr1}, ssim: {ssim1}')
                    save_dehazed_images(dehazed_image.data + 0.5, iteration, epoch)

            ssim_eval = np.mean(ssim_5val)
            psnr_eval = np.mean(psnr_5val)
            loss_eval = np.mean(loss_5val)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            losses_val.append(loss_eval)
            plt.subplot(3, 1, 2)
            plt.plot(losses_val)
            plt.title('val Losses')
            plt.pause(0.1)
            plt.subplot(3, 1, 3)
            plt.plot(psnrs)
            plt.title('psnrs')
            plt.pause(0.1)

            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'epoch': epoch,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'losses_val': losses_val,
                    'model': encoder_lv1.state_dict(),
                    'optimizer': encoder_lv1_optim.state_dict(),
                }, str('./checkpoints/' + METHOD + f"/encoder_lv1.pkl"))
                torch.save({
                    'model': encoder_lv2.state_dict(),
                    'optimizer': encoder_lv2_optim.state_dict()
                }, str('./checkpoints/' + METHOD + "/encoder_lv2.pkl"))
                torch.save({
                    'model': encoder_lv3.state_dict(),
                    'optimizer': encoder_lv3_optim.state_dict()
                }, str('./checkpoints/' + METHOD + "/encoder_lv3.pkl"))

                torch.save({
                    'model': decoder_lv1.state_dict(),
                    'optimizer': decoder_lv1_optim.state_dict()
                }, str('./checkpoints/' + METHOD + "/decoder_lv1.pkl"))
                torch.save({
                    'model': decoder_lv2.state_dict(),
                    'optimizer': decoder_lv2_optim.state_dict()
                }, str('./checkpoints/' + METHOD + "/decoder_lv2.pkl"))
                torch.save({
                    'model': decoder_lv3.state_dict(),
                    'optimizer': decoder_lv3_optim.state_dict()
                }, str('./checkpoints/' + METHOD + "/decoder_lv3.pkl"))
            print(f'epoch: {epoch}', 'loss_eval: %.4f' % loss_eval, f'psnr: {psnr_eval:.3f}', f'ssim: {ssim_eval:.3f}')
            print('model has saved')

        if not os.path.exists('./checkpoints/' + METHOD + '/losses'):
            os.system('mkdir ./checkpoints/' + METHOD + '/losses')
        np.save(f'./checkpoints/{METHOD}/losses/losses.npy', losses)
        np.save(f'./checkpoints/{METHOD}/losses/losses_val.npy', losses_val)
        np.save(f'./checkpoints/{METHOD}/losses/psnr_val.npy', psnrs)
        np.save(f'./checkpoints/{METHOD}/losses/ssim_val.npy', ssims)


if __name__ == '__main__':
    main()
