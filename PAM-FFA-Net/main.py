import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from models import *
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from option import opt, model_name, log_dir
from data_utils import *
from torchvision.models import vgg16

models_ = {
    'Pam-ffa': PAM_FFA(gps=opt.gps, blocks=opt.blocks),
}
loaders_ = {
    'train': train_loader,
    'test': test_loader,
}

start_time = time.time()
T = opt.steps
GPU = opt.device


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, loader_test, optim, criterion):
    losses = []
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    losses_val = []
    total_num = int(opt.steps / opt.eval_step)
    start_step = 0
    best_num = 0

    if opt.resume and os.path.exists(opt.model_dir):
        print('train from step 0')
        ckp = torch.load(opt.model_dir)
        my_model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in ckp.items() if k in my_model_dict}
        my_model_dict.update(pretrained_dict)
        net.load_state_dict(my_model_dict)
        # optim.load_state_dict(ckp['optim'])
    else:
        print('train from step 0 ')
    for step in range(start_step + 1, opt.steps + 1):
        now_num = step // opt.eval_step + 1
        net.train()
        lr = opt.lr

        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y, z = next(iter(loader_train))  # haze,clear,prior
        x = x.to(opt.device)
        y = y.to(opt.device)
        z = z.to(opt.device)
        out = net((x, z))
        # loss = criterion[0](out, y)  # L1
        loss = criterion[1](out, y)  # Total

        if opt.perloss:
            loss2 = criterion[2](out, y)
            loss = loss + 0.04 * loss2

        loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        if len(losses) % opt.eval_step == 0:
            plt.subplot(3, 1, 1)
            plt.plot(losses)
            plt.title('train Losses')
            fig = plt.gcf()
            fig.set_size_inches(16, 8)
            plt.pause(0.1)
        print(
            f'\rtrain loss: {loss.item():.5f}| step: {step}/{opt.steps}| lr: {lr :.7f} '
            f'| total_num: {total_num:03} now_num: {now_num:03} best_num: {best_num:03}| best_psnr{max_psnr:.2f}\n',
            end='', flush=True)

        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval, loss_val = test(net, loader_test, max_psnr, max_ssim, step)

            print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}| val_loss:{loss_val:.5f}')

            over_num = step // opt.eval_step
            save_pk = log_dir + '/' + f'{over_num:03}' + '.pk'
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            losses_val.append(loss_val)
            plt.subplot(3, 1, 2)
            plt.plot(losses_val)
            plt.title('val Losses')
            plt.pause(0.1)
            plt.subplot(3, 1, 3)
            plt.plot(psnrs)
            plt.title('psnrs')
            plt.pause(0.1)

            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                best_num = over_num
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'step': step,
                    'best_num': best_num,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'losses_val': losses_val,
                    'model': net.state_dict(),
                    'optim': optimizer.state_dict()
                }, opt.model_dir)
                print(
                    f'\n model saved at step :{step}|best_num:{best_num:03}|psnr:{max_psnr:.2f}')

            torch.save({
                'step': step,
                'best_num': best_num,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'losses_val': losses_val,
                'model': net.state_dict(),
                'optim': optimizer.state_dict()
            }, save_pk)
    plot_path = f'./numpy_files/{model_name}'
    os.makedirs(plot_path, exist_ok=True)
    np.save(f'{plot_path}/losses.npy', losses)
    np.save(f'{plot_path}/ssims.npy', ssims)
    np.save(f'{plot_path}/psnrs.npy', psnrs)
    np.save(f'{plot_path}/losses_val.npy', losses_val)


def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    val_losses = []
    # s=True
    for i, (inputs1, targets, inputs2) in enumerate(loader_test):
        inputs1 = inputs1.to(opt.device)  # haze
        inputs2 = inputs2.to(opt.device)  # prior
        targets = targets.to(opt.device)  # clear
        pred = net((inputs1, inputs2))
        val_loss = criterion[0](pred, targets)
        val_losses.append(val_loss.item())
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    return np.mean(ssims), np.mean(psnrs), np.mean(val_losses)


if __name__ == "__main__":

    loader_train = loaders_['train']
    loader_test = loaders_['test']

    net = models_['Pam-ffa']
    net = net.to(opt.device)

    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    #  Total loss
    total_loss = Total().cuda(GPU)

    criterion = [nn.L1Loss().to(opt.device), total_loss]

    if opt.perloss:
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to(opt.device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        criterion.append(PerLoss(vgg_model).to(opt.device))

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    train(net, loader_train, loader_test, optimizer, criterion)
