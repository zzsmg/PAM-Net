import torch.utils.data as data
from torchvision.transforms import functional as FF
import sys
from torch.utils.data import Dataset
from torchvision import transforms
sys.path.append('.')
sys.path.append('..')
import random
from PIL import Image
from torch.utils.data import DataLoader
import os
from option import args

BS = args.train_batch_size  # batch size = 16
crop_size = 'whole_img'
if args.crop:
    crop_size = args.crop_size


class Combined_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, format='.png'):
        super(Combined_Dataset, self).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.size = size
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'haze'))
        self.haze_imgs = [os.path.join(path, 'haze', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')
        self.prior_dir = os.path.join(path, 'prior')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index]).convert("RGB")
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        name = img.split('/')[-1].split('_haze')[0]
        clear_name = name + self.format
        prior_name = name + '_prior' + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name)).convert("RGB")
        prior = Image.open(os.path.join(self.prior_dir, prior_name)).convert("RGB")
        clear = transforms.CenterCrop(haze.size[::-1])(clear)
        prior = transforms.CenterCrop(haze.size[::-1])(prior)
        haze = transforms.CenterCrop(haze.size[::-1])(haze)
        if not isinstance(self.size, str):
            i, j, h, w = transforms.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
            prior = FF.crop(prior, i, j, h, w)
        clear = self.transform(clear)
        haze = self.transform(haze)
        prior = self.transform(prior)
        clear, haze, prior = self.augData(clear, haze, prior)
        return haze, clear, prior, str(name)

    def augData(self, c, h, p):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)

            if rand_hor:
                h = transforms.RandomHorizontalFlip(rand_hor)(h)
                c = transforms.RandomHorizontalFlip(rand_hor)(c)
                p = transforms.RandomHorizontalFlip(rand_hor)(p)

            if rand_rot:
                c = FF.rotate(c, 90 * rand_rot)
                h = FF.rotate(h, 90 * rand_rot)
                p = FF.rotate(p, 90 * rand_rot)

        return c, h, p

    def __len__(self):
        return len(self.haze_imgs)


class Dataset_test(data.Dataset):
    def __init__(self, path, train, size=crop_size, format='.png'):
        super(Dataset_test, self).__init__()
        self.size = size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'haze'))
        self.haze_imgs = [os.path.join(path, 'haze', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')
        self.prior_dir = os.path.join(path, 'prior')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index]).convert("RGB")
        img = self.haze_imgs[index]
        name = img.split('/')[-1].split('_haze')[0]
        clear_name = name + self.format
        prior_name = name + '_prior' + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name)).convert("RGB")
        prior = Image.open(os.path.join(self.prior_dir, prior_name)).convert("RGB")
        haze = self.transform(haze)
        prior = self.transform(prior)
        clear = self.transform(clear)
        hazy_up_left = haze[:, 0:1600, 0:2432]
        hazy_up_middle = haze[:, 0:1600, 1800:4232]
        hazy_up_right = haze[:, 0:1600, 3568:6000]

        prior_up_left = prior[:, 0:1600, 0:2432]
        prior_up_middle = prior[:, 0:1600, 1800:4232]
        prior_up_right = prior[:, 0:1600, 3568:6000]

        hazy_middle_left = haze[:, 1200:2800, 0:2432]
        hazy_middle_middle = haze[:, 1200:2800, 1800:4232]
        hazy_middle_right = haze[:, 1200:2800, 3568:6000]

        prior_middle_left = prior[:, 1200:2800, 0:2432]
        prior_middle_middle = prior[:, 1200:2800, 1800:4232]
        prior_middle_right = prior[:, 1200:2800, 3568:6000]

        hazy_down_left = haze[:, 2400:4000, 0:2432]
        hazy_down_middle = haze[:, 2400:4000, 1800:4232]
        hazy_down_right = haze[:, 2400:4000, 3568:6000]

        prior_down_left = prior[:, 2400:4000, 0:2432]
        prior_down_middle = prior[:, 2400:4000, 1800:4232]
        prior_down_right = prior[:, 2400:4000, 3568:6000]

        name = clear_name

        return clear, hazy_up_left, hazy_up_middle, hazy_up_right, prior_up_left, prior_up_middle, prior_up_right, hazy_middle_left, hazy_middle_middle, hazy_middle_right, prior_middle_left, prior_middle_middle, prior_middle_right, hazy_down_left, hazy_down_middle, hazy_down_right, prior_down_left, prior_down_middle, prior_down_right, str(
            name)

    def __len__(self):
        return len(self.haze_imgs)


path = '/data/Pytorch_Porjects/PAM-Net/PAM-DWT/Combined'

train_loader = DataLoader(dataset=Combined_Dataset(path + '/Train', train=True, size=crop_size),
                              batch_size=BS, shuffle=True)
test_loader = DataLoader(dataset=Dataset_test(path + '/Test', train=False, size='whole img'),
                             batch_size=1, shuffle=False)

if __name__ == "__main__":
    pass
