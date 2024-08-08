import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from model import fusion_net, Discriminator
import re
from utils import to_psnr, to_ssim_skimage
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim
from perceptual import LossNetwork
from data_utils import *
from option import args

# --- python train.py --crop --crop_size=384 --- #

# --- Load training data --- #
loaders_ = {
    'train': train_loader,
    'test': test_loader,
}
train_loader = loaders_['train']
test_loader = loaders_['test']

# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch

# --- test --- #
predict_result = args.predict_result

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = fusion_net()
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))  # 获取网络参数量
DNet = Discriminator()

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=0.0001)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[3000, 5000, 8000], gamma=0.5)
D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[5000, 7000, 8000], gamma=0.5)

MyEnsembleNet = MyEnsembleNet.to(device)
DNet = DNet.to(device)

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True)
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model)
loss_network.eval()
msssim_loss = msssim

# --- Strat training --- #
iteration = 0
losses_train = []
psnrs = []
ssims = []
max_psnr = 0
max_ssim = 0
best_epoch = 0

if os.path.exists(args.model_dir):
    ckp = torch.load(args.model_dir)
    my_model_dict = MyEnsembleNet.state_dict()
    pretrained_dict = {k: v for k, v in ckp.items() if k in my_model_dict}
    my_model_dict.update(pretrained_dict)
    MyEnsembleNet.load_state_dict(my_model_dict)
transform = transforms.Compose([transforms.ToTensor()])
gt_path = '/data/Pytorch_Porjects/PAM-Net/PAM-DWT/Combined/Test/clear/'
for epoch in range(best_epoch+1, train_epoch):
    start_time = time.time()
    scheduler_G.step()
    scheduler_D.step()
    MyEnsembleNet.train()
    DNet.train()
    loss_epoch_list = []
    for hazy, clean, prior, _ in train_loader:
        iteration += 1
        hazy = hazy.to(device)
        clean = clean.to(device)
        prior = prior.to(device)
        output = MyEnsembleNet((hazy, prior))
        DNet.zero_grad()
        real_out = DNet(clean).mean()
        fake_out = DNet(output).mean()
        D_loss = 1 - real_out + fake_out
        D_loss.backward(retain_graph=True)
        adversarial_loss = torch.mean(1 - fake_out)
        MyEnsembleNet.zero_grad()
        smooth_loss_l1 = F.smooth_l1_loss(output, clean)
        perceptual_loss = loss_network(output, clean)
        msssim_loss_ = -msssim_loss(output, clean, normalize=True)
        total_loss = smooth_loss_l1 + 0.01 * perceptual_loss + 0.0005 * adversarial_loss + 0.2 * msssim_loss_  # 总损失
        total_loss.backward()
        D_optim.step()
        G_optimizer.step()
        losses_train.append(total_loss.item())
        loss_epoch_list.append(total_loss.item())
        loss_epoch = float(np.mean(loss_epoch_list))
        if len(losses_train) % 5 == 0:
            plt.subplot(3, 1, 1)
            plt.plot(losses_train)
            plt.title('train Losses')
            fig = plt.gcf()
            fig.set_size_inches(16, 8)
            plt.pause(0.1)
            print(f'epoch:{epoch}, loss:{loss_epoch:.4f}')

    if epoch % 5 == 0:
        print('we are testing on epoch: ' + str(epoch))
        MyEnsembleNet.eval()
        with torch.no_grad():
            for batch_idx, (clear, hazy_up_left,hazy_up_middle,hazy_up_right,prior_up_left,prior_up_middle,prior_up_right,hazy_middle_left,hazy_middle_middle,hazy_middle_right,prior_middle_left,prior_middle_middle,prior_middle_right,hazy_down_left,hazy_down_middle,hazy_down_right,prior_down_left,prior_down_middle,prior_down_right, name) in enumerate(test_loader):
                psnr_list = []
                ssim_list = []

                hazy_up_left = hazy_up_left.to(device)
                hazy_up_middle = hazy_up_middle.to(device)
                hazy_up_right = hazy_up_right.to(device)

                prior_up_left = prior_up_left.to(device)
                prior_up_middle = prior_up_middle.to(device)
                prior_up_right = prior_up_right.to(device)

                hazy_middle_left = hazy_middle_left.to(device)
                hazy_middle_middle = hazy_middle_middle.to(device)
                hazy_middle_right = hazy_middle_right.to(device)

                prior_middle_left = prior_middle_left.to(device)
                prior_middle_middle = prior_middle_middle.to(device)
                prior_middle_right = prior_middle_right.to(device)

                hazy_down_left = hazy_down_left.to(device)
                hazy_down_middle = hazy_down_middle.to(device)
                hazy_down_right = hazy_down_right.to(device)

                prior_down_left = prior_down_left.to(device)
                prior_down_middle = prior_down_middle.to(device)
                prior_down_right = prior_down_right.to(device)

                frame_out_up_left = MyEnsembleNet((hazy_up_left, prior_up_left))
                frame_out_middle_left = MyEnsembleNet((hazy_middle_left, prior_middle_left))
                frame_out_down_left = MyEnsembleNet((hazy_down_left, prior_down_left))

                frame_out_up_middle = MyEnsembleNet((hazy_up_middle, prior_up_middle))
                frame_out_middle_middle = MyEnsembleNet((hazy_middle_middle, prior_middle_middle))
                frame_out_down_middle = MyEnsembleNet((hazy_down_middle, prior_down_middle))

                frame_out_up_right = MyEnsembleNet((hazy_up_right, prior_up_right))
                frame_out_middle_right = MyEnsembleNet((hazy_middle_right, prior_middle_right))
                frame_out_down_right = MyEnsembleNet((hazy_down_right, prior_down_right))

                frame_out_up_left = frame_out_up_left.to(device)
                frame_out_middle_left = frame_out_middle_left.to(device)
                frame_out_down_left = frame_out_down_left.to(device)
                frame_out_up_middle = frame_out_up_middle.to(device)
                frame_out_middle_middle = frame_out_middle_middle.to(device)
                frame_out_down_middle = frame_out_down_middle.to(device)
                frame_out_up_right = frame_out_up_right.to(device)
                frame_out_middle_right = frame_out_middle_right.to(device)
                frame_out_down_right = frame_out_down_right.to(device)

                if frame_out_up_left.shape[2] == 1600:
                    frame_out_up_left_middle = (frame_out_up_left[:, :, :, 1800:2432] + frame_out_up_middle[:, :, :,
                                                                                        0:632]) / 2
                    frame_out_up_middle_right = (frame_out_up_middle[:, :, :, 1768:2432] + frame_out_up_right[:, :, :,
                                                                                           0:664]) / 2

                    frame_out_middle_left_middle = (frame_out_middle_left[:, :, :, 1800:2432] + frame_out_middle_middle[
                                                                                                :,
                                                                                                :, :,
                                                                                                0:632]) / 2
                    frame_out_middle_middle_right = (frame_out_middle_middle[:, :, :,
                                                     1768:2432] + frame_out_middle_right[:,
                                                                  :,
                                                                  :,
                                                                  0:664]) / 2

                    frame_out_down_left_middle = (frame_out_down_left[:, :, :, 1800:2432] + frame_out_down_middle[:, :,
                                                                                            :,
                                                                                            0:632]) / 2
                    frame_out_down_middle_right = (frame_out_down_middle[:, :, :, 1768:2432] + frame_out_down_right[:,
                                                                                               :, :,
                                                                                               0:664]) / 2

                    frame_out_left_up_middle = (frame_out_up_left[:, :, 1200:1600, 0:1800] + frame_out_middle_left[:, :,
                                                                                             0:400,
                                                                                             0:1800]) / 2
                    frame_out_left_middle_down = (frame_out_middle_left[:, :, 1200:1600, 0:1800] + frame_out_down_left[
                                                                                                   :, :,
                                                                                                   0:400,
                                                                                                   0:1800]) / 2

                    frame_out_left = (torch.cat(
                        [frame_out_up_left[:, :, 0:1200, 0:1800].permute(0, 2, 3, 1),
                         frame_out_left_up_middle.permute(0, 2, 3, 1),
                         frame_out_middle_left[:, :, 400:1200, 0:1800].permute(0, 2, 3, 1),
                         frame_out_left_middle_down.permute(0, 2, 3, 1),
                         frame_out_down_left[:, :, 400:, 0:1800].permute(0, 2, 3, 1)], 1))

                    frame_out_leftmiddle_up_middle = (frame_out_up_left_middle[:, :, 1200:1600,
                                                      :] + frame_out_middle_left_middle[:,
                                                           :, 0:400, :]) / 2
                    frame_out_leftmiddle_middle_down = (frame_out_middle_left_middle[:, :, 1200:1600,
                                                        :] + frame_out_down_left_middle[:, :, 0:400, :]) / 2

                    frame_out_leftmiddle = (torch.cat([frame_out_up_left_middle[:, :, 0:1200, :].permute(0, 2, 3, 1),
                                                       frame_out_leftmiddle_up_middle.permute(0, 2, 3, 1),
                                                       frame_out_middle_left_middle[:, :, 400:1200, :].permute(0, 2, 3,
                                                                                                               1),
                                                       frame_out_leftmiddle_middle_down.permute(0, 2, 3, 1),
                                                       frame_out_down_left_middle[:, :, 400:, :].permute(0, 2, 3, 1)],
                                                      1))

                    frame_out_middle_up_middle = (frame_out_up_middle[:, :, 1200:1600,
                                                  632:1768] + frame_out_middle_middle[
                                                              :, :,
                                                              0:400, 632:1768]) / 2
                    frame_out_middle_middle_down = (frame_out_middle_middle[:, :, 1200:1600,
                                                    632:1768] + frame_out_down_middle[
                                                                :, :,
                                                                0:400, 632:1768]) / 2

                    frame_out_middle = (torch.cat([frame_out_up_middle[:, :, 0:1200, 632:1768].permute(0, 2, 3, 1),
                                                   frame_out_middle_up_middle.permute(0, 2, 3, 1),
                                                   frame_out_middle_middle[:, :, 400:1200, 632:1768].permute(0, 2, 3,
                                                                                                             1),
                                                   frame_out_middle_middle_down.permute(0, 2, 3, 1),
                                                   frame_out_down_middle[:, :, 400:, 632:1768].permute(0, 2, 3, 1)], 1))

                    frame_out_middleright_up_middle = (frame_out_up_middle_right[:, :, 1200:1600,
                                                       :] + frame_out_middle_middle_right[:, :, 0:400, :]) / 2
                    frame_out_middleright_middle_down = (frame_out_middle_middle_right[:, :, 1200:1600,
                                                         :] + frame_out_down_middle_right[:, :, 0:400, :]) / 2

                    frame_out_middleright = (torch.cat([frame_out_up_middle_right[:, :, 0:1200, :].permute(0, 2, 3, 1),
                                                        frame_out_middleright_up_middle.permute(0, 2, 3, 1),
                                                        frame_out_middle_middle_right[:, :, 400:1200, :].permute(0, 2,
                                                                                                                    3,
                                                                                                                   1),
                                                        frame_out_middleright_middle_down.permute(0, 2, 3, 1),
                                                        frame_out_down_middle_right[:, :, 400:, :].permute(0, 2, 3, 1)],
                                                       1))

                    frame_out_right_up_middle = (frame_out_up_right[:, :, 1200:1600, 664:] + frame_out_middle_right[:,
                                                                                             :,
                                                                                             0:400,
                                                                                             664:]) / 2
                    frame_out_right_middle_down = (frame_out_middle_right[:, :, 1200:1600, 664:] + frame_out_down_right[
                                                                                                   :,
                                                                                                   :,
                                                                                                   0:400,
                                                                                                   664:]) / 2

                    frame_out_right = (torch.cat(
                        [frame_out_up_right[:, :, 0:1200, 664:].permute(0, 2, 3, 1),
                         frame_out_right_up_middle.permute(0, 2, 3, 1),
                         frame_out_middle_right[:, :, 400:1200, 664:].permute(0, 2, 3, 1),
                         frame_out_right_middle_down.permute(0, 2, 3, 1),
                         frame_out_down_right[:, :, 400:, 664:].permute(0, 2, 3, 1)], 1))

                if frame_out_up_left.shape[2] == 2432:
                    frame_out_up_left_middle = (frame_out_up_left[:, :, :, 1200:1600] + frame_out_up_middle[:, :, :,
                                                                                        0:400]) / 2
                    frame_out_up_middle_right = (frame_out_up_middle[:, :, :, 1200:1600] + frame_out_up_right[:, :, :,
                                                                                           0:400]) / 2

                    frame_out_middle_left_middle = (frame_out_middle_left[:, :, :, 1200:1600] + frame_out_middle_middle[
                                                                                                :,
                                                                                                :, :,
                                                                                                0:400]) / 2
                    frame_out_middle_middle_right = (frame_out_middle_middle[:, :, :,
                                                     1200:1600] + frame_out_middle_right[:,
                                                                  :,
                                                                  :,
                                                                  0:400]) / 2

                    frame_out_down_left_middle = (frame_out_down_left[:, :, :, 1200:1600] + frame_out_down_middle[:, :,
                                                                                            0:400]) / 2
                    frame_out_down_middle_right = (frame_out_down_middle[:, :, :, 1200:1600] + frame_out_down_right[:,
                                                                                               :, :,
                                                                                               0:400]) / 2

                    frame_out_left_up_middle = (frame_out_up_left[:, :, 1800:2432, 0:1200] + frame_out_middle_left[:, :,
                                                                                             0:632,
                                                                                             0:1200]) / 2
                    frame_out_left_middle_down = (frame_out_middle_left[:, :, 1768:2432, 0:1200] + frame_out_down_left[
                                                                                                   :, :,
                                                                                                   0:664,
                                                                                                   0:1200]) / 2

                    frame_out_left = (torch.cat(
                        [frame_out_up_left[:, :, 0:1800, 0:1200].permute(0, 2, 3, 1),
                         frame_out_left_up_middle.permute(0, 2, 3, 1),
                         frame_out_middle_left[:, :, 632:1768, 0:1200].permute(0, 2, 3, 1),
                         frame_out_left_middle_down.permute(0, 2, 3, 1),
                         frame_out_down_left[:, :, 664:, 0:1200].permute(0, 2, 3, 1)], 1))

                    frame_out_leftmiddle_up_middle = (frame_out_up_left_middle[:, :, 1800:2432,
                                                      :] + frame_out_middle_left_middle[:,
                                                           :, 0:632, :]) / 2
                    frame_out_leftmiddle_middle_down = (frame_out_middle_left_middle[:, :, 1768:2432,
                                                        :] + frame_out_down_left_middle[:, :, 0:664, :]) / 2

                    frame_out_leftmiddle = (torch.cat([frame_out_up_left_middle[:, :, 0:1800, :].permute(0, 2, 3, 1),
                                                       frame_out_leftmiddle_up_middle.permute(0, 2, 3, 1),
                                                       frame_out_middle_left_middle[:, :, 632:1768, :].permute(0, 2, 3,
                                                                                                               1),
                                                       frame_out_leftmiddle_middle_down.permute(0, 2, 3, 1),
                                                       frame_out_down_left_middle[:, :, 664:, :].permute(0, 2, 3, 1)],
                                                      1))

                    frame_out_middle_up_middle = (frame_out_up_middle[:, :, 1800:2432,
                                                  400:1200] + frame_out_middle_middle[
                                                              :, :,
                                                              0:632, 400:1200]) / 2
                    frame_out_middle_middle_down = (frame_out_middle_middle[:, :, 1768:2432,
                                                    400:1200] + frame_out_down_middle[
                                                                :, :,
                                                                0:664, 400:1200]) / 2

                    frame_out_middle = (torch.cat([frame_out_up_middle[:, :, 0:1800, 400:1200].permute(0, 2, 3, 1),
                                                   frame_out_middle_up_middle.permute(0, 2, 3, 1),
                                                   frame_out_middle_middle[:, :, 632:1768, 400:1200].permute(0, 2, 3,
                                                                                                             1),
                                                   frame_out_middle_middle_down.permute(0, 2, 3, 1),
                                                   frame_out_down_middle[:, :, 664:, 400:1200].permute(0, 2, 3, 1)], 1))

                    frame_out_middleright_up_middle = (frame_out_up_middle_right[:, :, 1800:2432,
                                                       :] + frame_out_middle_middle_right[:, :, 0:632, :]) / 2
                    frame_out_middleright_middle_down = (frame_out_middle_middle_right[:, :, 1768:2432,
                                                         :] + frame_out_down_middle_right[:, :, 0:664, :]) / 2

                    frame_out_middleright = (torch.cat([frame_out_up_middle_right[:, :, 0:1800, :].permute(0, 2, 3, 1),
                                                        frame_out_middleright_up_middle.permute(0, 2, 3, 1),
                                                        frame_out_middle_middle_right[:, :, 632:1768, :].permute(0, 2,
                                                                                                                 3,
                                                                                                                 1),
                                                        frame_out_middleright_middle_down.permute(0, 2, 3, 1),
                                                        frame_out_down_middle_right[:, :, 664:, :].permute(0, 2, 3, 1)],
                                                       1))

                    frame_out_right_up_middle = (frame_out_up_right[:, :, 1800:2432, 400:] + frame_out_middle_right[:,
                                                                                             :,
                                                                                             0:632,
                                                                                             400:]) / 2
                    frame_out_right_middle_down = (frame_out_middle_right[:, :, 1768:2432, 400:] + frame_out_down_right[
                                                                                                   :,
                                                                                                   :,
                                                                                                   0:664,
                                                                                                   400:]) / 2

                    frame_out_right = (torch.cat(
                        [frame_out_up_right[:, :, 0:1800, 400:].permute(0, 2, 3, 1),
                         frame_out_right_up_middle.permute(0, 2, 3, 1),
                         frame_out_middle_right[:, :, 632:1768, 400:].permute(0, 2, 3, 1),
                         frame_out_right_middle_down.permute(0, 2, 3, 1),
                         frame_out_down_right[:, :, 664:, 400:].permute(0, 2, 3, 1)], 1))

                frame_out = torch.cat(
                    [frame_out_left, frame_out_leftmiddle, frame_out_middle, frame_out_middleright, frame_out_right],
                    2).permute(0,
                               3,
                               1,
                               2)

                frame_out = frame_out.to(device)

                fourth_channel = torch.ones([frame_out.shape[0], 1, frame_out.shape[2], frame_out.shape[3]],
                                            device='cuda:0')
                frame_out_rgba = torch.cat([frame_out, fourth_channel], 1)

                name = re.findall("\d+", str(name))
                output_path = predict_result + f'epoch_{epoch}/'
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                imwrite(frame_out, output_path + str(name[0]) + '.png', range=(0, 1))
                frame_out = Image.open(output_path + name[0] + '.png').convert('RGB')
                frame_out = transform(frame_out)
                clean = Image.open(gt_path + name[0] + '.png').convert('RGB')
                clean = transform(clean)
                psnr_test_list = to_psnr(frame_out, clean)
                psnr_test = sum(psnr_test_list) / len(psnr_test_list)
                psnr_list.append(psnr_test)
                ssim_test_list = to_ssim_skimage(frame_out, clean)
                ssim_test = sum(ssim_test_list) / len(ssim_test_list)
                ssim_list.append(ssim_test)
            avr_psnr = float(np.mean(psnr_list))
            psnrs.append(avr_psnr)
            avr_ssim = float(np.mean(ssim_list))
            ssims.append(avr_ssim)
            plt.subplot(3, 1, 2)
            plt.plot(psnrs)
            plt.title('psnrs')
            plt.pause(0.1)
            plt.subplot(3, 1, 3)
            plt.plot(ssims)
            plt.title('ssims')
            plt.pause(0.1)
            # writer.add_scalars('testing', {'testing psnr': avr_psnr}, epoch)
            torch.save({'model': MyEnsembleNet.state_dict(),
                        'losses_train': losses_train,
                        'psnrs': psnrs,
                        'ssims': ssims,
                        'max_psnr': max_psnr,
                        'max_ssim': max_ssim,
                        'best_epoch': best_epoch,
                        'G_optim': G_optimizer.state_dict(),
                        'D_optim': D_optim.state_dict()
                        },
                       os.path.join(args.model_save_dir, 'epoch' + str(epoch) + '.pkl'))
            plot_path = args.plot_path
            os.makedirs(plot_path, exist_ok=True)
            np.save(f'{plot_path}/losses_train.npy', losses_train)
            np.save(f'{plot_path}/psnrs.npy', psnrs)
            if avr_psnr > max_psnr:
                best_epoch = epoch
                max_psnr = avr_psnr
                torch.save({'model': MyEnsembleNet.state_dict(),
                            'losses_train': losses_train,
                            'psnrs': psnrs,
                            'ssims': ssims,
                            'max_psnr': max_psnr,
                            'max_ssim': max_ssim,
                            'best_epoch': best_epoch,
                            'G_optim': G_optimizer.state_dict(),
                            'D_optim': D_optim.state_dict()
                            },
                           os.path.join(args.best_model, 'best' + '.pkl'))
            print(f'test_psnr:{avr_psnr}, test_ssim:{avr_ssim},best_epoch:{best_epoch}')
