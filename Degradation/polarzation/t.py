import cv2
import sys
import math
import numpy as np
from sympy import Symbol, solve
import pandas
import time
from PIL import Image
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def process_pixel(y):
    global img_gx, img_jc
    row_1 = []
    row_2 = []
    row_3 = []
    for x in range(12, w - 12):
        patch_gx = img_gx[y - 12:y + 13, x - 12:x + 13]
        patch_jc = img_jc[y - 12:y + 13, x - 12:x + 13]
        patch_I = cv2.add(patch_gx, patch_jc)
        patch_drtI = cv2.subtract(patch_jc, patch_gx)
        patch_I1 = cv2.subtract(patch_I, A)
        patch_I2 = cv2.multiply(patch_I, pa)
        patch_drtI1 = cv2.subtract(patch_drtI, patch_I2)
        eps = 1e-5
        patch_drtI1 = cv2.add(patch_drtI1, eps)
        H1 = cv2.divide(patch_I1, patch_drtI1)
        patch_drtI2 = cv2.subtract(patch_drtI, A * pa)
        H2 = cv2.divide(patch_drtI2, patch_drtI1)
        patch_I = cv2.add(patch_I, eps)
        TDOP_I = patch_drtI / patch_I
        TDOP_mid = TDOP_I[12, 12]

        max_TDOP = 0
        height, width = TDOP_I.shape
        for i3 in range(height):
            for j in range(width):
                if TDOP_I[i3, j] > max_TDOP:
                    max_TDOP = TDOP_I[i3, j]

        height_I, width_I = patch_I.shape
        max_pixel = 0
        pixel_mid = patch_I[12, 12]
        pixel_mid_drt = patch_drtI[12, 12]
        for i2 in range(height_I):
            for j in range(width_I):
                if patch_I[i2, j] > max_pixel:
                    max_pixel = patch_I[i2, j]

        Em_x = (max_TDOP - TDOP_mid) * (max_pixel - pixel_mid)
        W_x = 0
        W_xy = []
        for i1 in range(height_I):
            row = []
            for j in range(width_I):
                Em_y = (max_TDOP - TDOP_I[i1, j]) * (max_pixel - patch_I[i1, j])
                w_xy = np.float64(math.exp(-((Em_x - Em_y) ** 2) / (0.25 * 0.25)))
                W_x = W_x + w_xy
                row.append(w_xy)
            W_xy.append(row)
        W_xy = np.float64(W_xy)
        wg_I = 0
        wg_drtI = 0
        wg_H1 = 0
        wg_H2 = 0
        rows = len(W_xy)
        cols = len(W_xy[0])
        for row in range(rows):
            for col in range(cols):
                wg_I = wg_I + W_xy[row][col] * patch_I[row, col]
                wg_drtI = wg_drtI + W_xy[row][col] * patch_drtI[row, col]
                wg_H1 = wg_H1 + W_xy[row][col] * H1[row, col]
                wg_H2 = wg_H2 + W_xy[row][col] * H2[row, col]
        Ex_I = (1 / W_x) * wg_I
        Ex_drtI = (1 / W_x) * wg_drtI
        Ex_H1 = (1 / W_x) * wg_H1
        Ex_H2 = (1 / W_x) * wg_H2
        Cx_I_H1_1 = 0
        Cx_I_H2_1 = 0
        Cx_drtI_H1_1 = 0
        Cx_drtI_H2_1 = 0
        for row in range(rows):
            for col in range(cols):
                Cx_I_H1_1 = Cx_I_H1_1 + ((patch_I[row, col] - Ex_I) * (H1[row, col] - Ex_H1)) * W_xy[row][col]
                Cx_I_H2_1 = Cx_I_H2_1 + ((patch_I[row, col] - Ex_I) * (H2[row, col] - Ex_H2)) * W_xy[row][col]
                Cx_drtI_H1_1 = Cx_drtI_H1_1 + (
                        ((patch_drtI[row, col] - Ex_drtI) * (H1[row, col] - Ex_H1)) * W_xy[row][col])
                Cx_drtI_H2_1 = Cx_drtI_H2_1 + (
                        ((patch_drtI[row, col] - Ex_drtI) * (H2[row, col] - Ex_H2)) * W_xy[row][col])
        Cx_I_H1 = (1 / W_x) * Cx_I_H1_1
        Cx_drtI_H1 = (1 / W_x) * Cx_drtI_H1_1
        Cx_drtI_H2 = (1 / W_x) * Cx_drtI_H2_1
        a = Cx_I_H1
        b = - Cx_drtI_H1 - np.cov(patch_I.ravel(), H2.ravel())[0, 1]
        c = Cx_drtI_H2

        drt = b ** 2 - 4 * a * c
        if drt > 0 and a != 0:
            pd = -(b / a) - pa
        elif drt < 0:
            f_pa = a * (pa ** 2) + b * pa + c
            pd_1 = Symbol('pd_1')
            pd_2 = Symbol('pd_2')
            result1 = pa + math.sqrt(f_pa / a)
            result2 = pa - math.sqrt(f_pa / a)
            f1 = a * ((result1 / pd_1) ** 2) + b * (result1 / pd_1) + c
            f2 = a * ((result2 / pd_2) ** 2) + b * (result2 / pd_2) + c
            df11 = f1.diff(pd_1)
            df22 = f2.diff(pd_2)
            pd_1_min = solve(df11, pd_1)[0]
            pd_2_min = solve(df22, pd_2)[0]
            if pd_1_min > pd_2_min:
                pd = pd_2_min
            else:
                pd = pd_1_min

        else:
            neighbors = patch_I[11:14, 11:14]
            patch_I[12, 12] = (np.sum(neighbors) - patch_I[12, 12]) / 8
            pd = -0.01
            print(f"Wrong coordinates：({x}{y})")

        pd = np.float64(pd)
        t = np.float64((1 - ((pd * pixel_mid - pixel_mid_drt) / ((pd - pa) * A))))
        t_1 = t
        if t_1 > 1:
            t_1 = 1
            t_1 = np.float64(t_1)
        elif t_1 < 0:
            t_1 = 0
            t_1 = np.float64(t_1)
        row_1.append(t)
        row_2.append(t_1)

    return row_1, row_2


def worker(start_y, end_y):
    img_depth_t = []
    img_depth_t1 = []
    for y in range(start_y, end_y):
        img_depth_t.append(process_pixel(y)[0])
        img_depth_t1.append(process_pixel(y)[1])
    return img_depth_t, img_depth_t1


def crop_image(image_path, output_path):
    with Image.open(image_path) as img:
        width, height = img.size
        left = 12
        top = 12
        right = width - 12
        bottom = height - 12
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)


def downsampling(image_path, output_path):
    with Image.open(image_path) as img:
        w1 = img.width // 2
        h1 = img.height // 2
        bicubic_img = img.resize((w1, h1), Image.BICUBIC)
        bicubic_img.save(output_path)


if __name__ == '__main__':
    stat_time = time.time()

    # input img, rename!!
    P1 = "img/haze_0.png"
    P2 = "img/haze_90.png"
    P3 = "img/clear_0.png"
    P4 = "img/clear_90.png"
    img_gx = cv2.imread(P1, cv2.IMREAD_GRAYSCALE)
    img_jc = cv2.imread(P2, cv2.IMREAD_GRAYSCALE)
    clear_gx = cv2.imread(P3, cv2.IMREAD_GRAYSCALE)
    clear_jc = cv2.imread(P4, cv2.IMREAD_GRAYSCALE)
    img_gx = np.float64(img_gx)
    img_jc = np.float64(img_jc)  # float64
    img_I = cv2.add(img_gx, img_jc)
    img_I = np.array(img_I)
    clear_I = np.array(cv2.add(np.float64(clear_gx), np.float64(clear_jc)))

    height_gx, width_gx = img_gx.shape
    height_jc, width_jc = img_jc.shape
    if height_gx != height_jc and width_gx != width_jc:
        sys.exit("Image sizes are inconsistent")

    # estimate A
    y1, y2, x1, x2 = 154, 168, 325, 341
    patch1_gx = np.float64(img_gx[y1:y2, x1:x2])
    patch1_jc = np.float64(img_jc[y1:y2, x1:x2])
    mean_patch1_gx = cv2.mean(patch1_gx)[0]
    mean_patch1_jc = cv2.mean(patch1_jc)[0]
    A = np.float64(mean_patch1_gx + mean_patch1_jc)
    pa = np.float64(- abs((mean_patch1_jc - mean_patch1_gx) / (mean_patch1_gx + mean_patch1_jc)))  # Pa,float64
    print(f"A\u221E = {A:.5f}\npa = {pa:.4f}")

    h, w = img_gx.shape
    num_process = 30
    length = (h - 24) // num_process
    if length < 13:
        raise ValueError("Reduce the number of processes")
    starts = [12 + i * length for i in range(num_process)]
    ends = [start + length for start in starts]
    ends[-1] = h - 12

    with Pool(processes=num_process) as pool:
        results = pool.starmap(worker, [(start, end) for start, end in zip(starts, ends)])
        img_depth_1 = [item for sublist in results for item in sublist[0]]
        img_depth_2 = [item for sublist in results for item in sublist[1]]

    img_depth_1 = np.array(img_depth_1)
    img_depth_2 = np.array(img_depth_2)

    root = 'result/001/'
    path1 = root + 'value/'
    path2 = root + 'pixel/'
    path3 = root + 'img_depth/'
    path4 = root + 'img_I/'

    if not os.path.exists(os.path.dirname(root)):
        os.makedirs(os.path.dirname(root))
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))
    if not os.path.exists(os.path.dirname(path2)):
        os.makedirs(os.path.dirname(path2))
    if not os.path.exists(os.path.dirname(path3)):
        os.makedirs(os.path.dirname(path3))
    if not os.path.exists(os.path.dirname(path4)):
        os.makedirs(os.path.dirname(path4))

    value_t = pandas.DataFrame(img_depth_1, dtype=float)
    value_t = value_t.round(2)
    value_t.to_excel(path1 + 'value_t.xlsx', index=False)

    depth_imwrite = np.array(np.abs(img_depth_1) * 255).astype(np.uint8)
    # depth_imshow = np.array(np.abs(img_depth_2) * 255).astype(np.uint8)
    cv2.imwrite(path3 + 'depth_imwrite.png', depth_imwrite)
    # cv2.imwrite(path3 + 'depth_imshow.png', depth_imshow)
    cv2.imwrite(path4 + 'haze_I.png', img_I)
    cv2.imwrite(path4 + 'clear_I.png', clear_I)

    # crop I -12 -12
    crop_image(path4 + 'haze_I.png', path4 + 'haze_I.png', )
    crop_image(path4 + 'clear_I.png', path4 + 'clear_I.png', )

    # downsampling(path3 + 'depth_imshow.png', path5 + 'LRx2_depth_imshow.png')

    pixel_depth_imwrite = cv2.imread(path3 + 'depth_imwrite.png', cv2.IMREAD_GRAYSCALE)
    # pixel_depth_imshow = cv2.imread(path3 + 'depth_imshow.png', cv2.IMREAD_GRAYSCALE)
    pixel_haze_I = cv2.imread(path4 + 'haze_I.png', cv2.IMREAD_GRAYSCALE)
    pixel_clear_I = cv2.imread(path4 + 'clear_I.png', cv2.IMREAD_GRAYSCALE)
    df1 = pandas.DataFrame(pixel_depth_imwrite)
    df1.to_excel(path2 + 'pixel_depth_imwrite.xlsx', index=False)
    # df2 = pandas.DataFrame(pixel_depth_imshow)
    # df2.to_excel(path2 + 'pixel_depth_imshow.xlsx', index=False)
    df3 = pandas.DataFrame(pixel_haze_I)
    df3.to_excel(path2 + 'pixel_haze_I.xlsx', index=False)
    df3 = pandas.DataFrame(pixel_clear_I)
    df3.to_excel(path2 + 'pixel_clear_I.xlsx', index=False)

    end_time = time.time()
    runtime = end_time - stat_time
    print(f"{round(runtime, 2)}")

