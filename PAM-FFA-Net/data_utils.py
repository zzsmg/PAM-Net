import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import sys
import numpy as np
sys.path.append('.')
sys.path.append('..')

import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
import os

BS = opt.bs  # batch size = 16
crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size


def tensorShow(tensors, titles=None):
    '''
    t:BCWH
    '''
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()


class MC_NH_HAZE(data.Dataset):
    def __init__(self, path, train, size=crop_size, format='.png'):
        super(MC_NH_HAZE, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'haze'))
        self.haze_imgs = [os.path.join(path, 'haze', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')
        self.prior_dir = os.path.join(path, 'prior')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        name = img.split('/')[-1].split('_haze')[0]
        clear_name = name + self.format
        prior_name = name + '_prior' + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        prior = Image.open(os.path.join(self.prior_dir, prior_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        prior = tfs.CenterCrop(haze.size[::-1])(prior)
        haze = tfs.CenterCrop(haze.size[::-1])(haze)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
            prior = FF.crop(prior, i, j, h, w)
        clear, haze, prior = self.augData(clear.convert("RGB"), haze.convert("RGB"), prior.convert("RGB"))
        return haze, clear, prior

    def augData(self, c, h, p):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)

            if rand_hor:
                h = tfs.RandomHorizontalFlip(rand_hor)(h)
                c = tfs.RandomHorizontalFlip(rand_hor)(c)
                p = tfs.RandomHorizontalFlip(rand_hor)(p)

            if rand_rot:
                c = FF.rotate(c, 90 * rand_rot)
                h = FF.rotate(h, 90 * rand_rot)
                p = FF.rotate(p, 90 * rand_rot)

        h = tfs.ToTensor()(h)
        h = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(h)
        c = tfs.ToTensor()(c)
        p = tfs.ToTensor()(p)
        p = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(p)
        return c, h, p

    def __len__(self):
        return len(self.haze_imgs)


path = '/data/Pytorch_Porjects/PAM-Net/PAM-FFA-Net/MC-NH-HAZE'  # path to your 'data' folder

train_loader = DataLoader(dataset=MC_NH_HAZE(path + '/train', train=True, size=240),
                              batch_size=BS,
                              shuffle=True)
test_loader = DataLoader(dataset=MC_NH_HAZE(path + '/val', train=False, size='whole img'),
                             batch_size=1, shuffle=False)


if __name__ == "__main__":
    pass
