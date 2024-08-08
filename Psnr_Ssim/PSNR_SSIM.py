import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from math import log10
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim


def to_psnr(frame_out, gt):
    mse = F.mse_loss(frame_out, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    rmse_list = np.sqrt(mse_list)
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 1).data.cpu().numpy().squeeze() for ind in
                      range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [ssim(dehaze_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True) for ind in
                 range(len(dehaze_list))]

    return ssim_list


def save_results_to_file(filename, results, avg_psnr, avg_ssim):
    with open(filename, 'w') as f:
        for result in results:
            f.write(f"{result[0]}: PSNR = {result[1]}, SSIM = {result[2]}\n")
        f.write(f'平均PSNR：{avg_psnr}\n')
        f.write(f'平均SSIM：{avg_ssim}\n')


gt_path = '/data/Pytorch_Porjects/DWT-FFC/datasets/combined_dataset/Test_offi/clear/'
out_pth = '/data/Pytorch_Porjects/PAM-Net/PAM-DWT/test/PAM-DWT/result/'
gt_files = os.listdir(gt_path)
out_files = os.listdir(out_pth)
transform = transforms.Compose([transforms.ToTensor()])
psnr_sum = 0
ssim_sum = 0
results = []
for gt in gt_files:
    gt_img = Image.open(os.path.join(gt_path, gt)).convert('RGB')
    gt_img = transform(gt_img)
    if gt in out_files:
        out_img = Image.open(os.path.join(out_pth, gt)).convert('RGB')
        out_img = transform(out_img)
        psnr = to_psnr(out_img, gt_img)
        ssim_list = to_ssim_skimage(out_img, gt_img)
        avr_psnr = sum(psnr) / len(psnr)
        avr_ssim = sum(ssim_list) / len(ssim_list)
        psnr_sum += avr_psnr
        ssim_sum += avr_ssim
        results.append((gt, avr_psnr, avr_ssim))
        print(f'{gt}: Psnr: {avr_psnr}, Ssim: {avr_ssim}')
avg_psnr = psnr_sum / len(gt_files)
avg_ssim = ssim_sum / len(gt_files)
save_results_to_file(out_pth + 'results.txt', results, avg_psnr, avg_ssim)
print(f'avgPsnr：{psnr_sum / len(gt_files)}')
print(f'avgSsim：{ssim_sum / len(gt_files)}')
