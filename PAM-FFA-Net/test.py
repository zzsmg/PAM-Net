import argparse
import numpy as np
from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import time

abs_path = os.getcwd() + '/'

parser = argparse.ArgumentParser()
parser.add_argument('--gps', type=int, default=3, help='residual_groups')
parser.add_argument('--blocks', type=int, default=19, help='residual_blocks')
opt = parser.parse_args()

gps = opt.gps
blocks = opt.blocks
img_dir = abs_path + 'test/hazy/'
img_dir_prior = abs_path + 'test/prior/'

# Corresponding output path
output_dir = abs_path + 'test/results/L_Total/PAM-FFA-1'   # save path

os.makedirs(img_dir, exist_ok=True)
os.makedirs(img_dir_prior, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
print("results:", output_dir)

# !!!
model_dir = abs_path + f'trained_models/Total_Pam_ffa_1.pk'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir, map_location=device)
net = PAM_FFA(gps=gps, blocks=blocks)
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'], strict=False)
net.eval()
runtimes = []
for im in os.listdir(img_dir):
    name = im.split('/')[-1].split('_haze')[0] + '_prior.png'
    for file in os.listdir(img_dir_prior):
        if file == name:
            T1 = time.time()  # runtime
            haze = Image.open(img_dir + im)
            haze = haze.convert("RGB")
            haze1 = tfs.Compose([
                tfs.ToTensor(),
                tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
            ])(haze)[None, ::]

            prior = Image.open(img_dir_prior + file)
            prior = prior.convert("RGB")
            prior1 = tfs.Compose([
                tfs.ToTensor(),
                tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
            ])(prior)[None, ::]

            haze_no = tfs.ToTensor()(haze)[None, ::]
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                pred = net((haze1, prior1))[0]
                # scale = net((haze1, prior1))[1]
                # shift = net((haze1, prior1))[2]
            torch.cuda.synchronize()
            end = time.time()
            runtime = end - start
            print(f'{im}')
            runtimes.append(runtime)
            ts = torch.squeeze(pred.clamp(0, 1).cpu())
            vutils.save_image(ts, output_dir + '/' + im.split('/')[-1].split('_haze')[0] + '.png')



