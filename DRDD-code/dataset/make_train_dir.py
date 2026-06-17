# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# import shutil

# def add_gaussian_noise(img, sigma):
#     noise = np.random.normal(0, sigma / 255.0, img.shape)
#     noisy_img = img + noise
#     noisy_img = np.clip(noisy_img, 0, 1)
#     return noisy_img

# def process_folder(src_folder, sigma, input_dir, gt_dir):
#     folder_name = os.path.basename(src_folder.rstrip('/'))
#     img_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
#     for img_name in tqdm(img_files, desc=f'Processing {folder_name}'):
#         src_path = os.path.join(src_folder, img_name)
#         # 去除原有扩展名，统一用png后缀
#         base_name = os.path.splitext(img_name)[0]
#         out_name = f"{folder_name}_{base_name}.png"
#         # 读取图片
#         img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
#         if img is None:
#             print(f"Warning: {src_path} cannot be read, skipping.")
#             continue
#         # 归一化到[0,1]
#         img_float = img.astype(np.float32) / 255.0
#         # 加噪
#         noisy_img = add_gaussian_noise(img_float, sigma)
#         # 转回uint8
#         noisy_img_uint8 = (noisy_img * 255.0).round().astype(np.uint8)

#         # 保存gt
#         gt_save_path = os.path.join(gt_dir, out_name)
#         shutil.copy(src_path, gt_save_path)
#         # 保存input
#         input_save_path = os.path.join(input_dir, out_name)
#         cv2.imwrite(input_save_path, noisy_img_uint8)

# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--sigma', type=int, default=25, help='Gaussian noise sigma')
#     args = parser.parse_args()

#     folders = [
#         '/root/data1/linziyue/RDDM/data/restoration/denoise/BSD400',
#         '/root/data1/linziyue/RDDM/data/restoration/denoise/DIV2K',
#         '/root/data1/linziyue/RDDM/data/restoration/denoise/Flickr2K',
#         '/root/data1/linziyue/RDDM/data/restoration/denoise/WaterlooED'
#     ]
#     input_dir = '/root/data1/linziyue/RDDM/data/restoration/denoise/train/input25'
#     gt_dir = '/root/data1/linziyue/RDDM/data/restoration/denoise/train/gt'

#     os.makedirs(input_dir, exist_ok=True)
#     os.makedirs(gt_dir, exist_ok=True)

#     for folder in folders:
#         process_folder(folder, args.sigma, input_dir, gt_dir)

# if __name__ == '__main__':
#     main()



import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma / 255.0, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img

def process_folder(src_folder, sigmas, input_dir, gt_dir, input_list, gt_list):
    folder_name = os.path.basename(src_folder.rstrip('/'))
    img_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    for img_name in tqdm(img_files, desc=f'Processing {folder_name}'):
        src_path = os.path.join(src_folder, img_name)
        base_name = os.path.splitext(img_name)[0]

        # 读取图片
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: {src_path} cannot be read, skipping.")
            continue
        img_float = img.astype(np.float32) / 255.0

        # 保存gt（只保存一次）
        gt_out_name = f"{base_name}_{folder_name}.png"
        gt_save_path = os.path.join(gt_dir, gt_out_name)
        shutil.copy(src_path, gt_save_path)

        for sigma in sigmas:
            # 加噪
            noisy_img = add_gaussian_noise(img_float, sigma)
            noisy_img_uint8 = (noisy_img * 255.0).round().astype(np.uint8)
            input_out_name = f"{base_name}_{folder_name}_{sigma}.png"
            input_save_path = os.path.join(input_dir, input_out_name)
            cv2.imwrite(input_save_path, noisy_img_uint8)

            # 记录到list
            input_list.append(input_save_path)
            gt_list.append(gt_save_path)

def main():
    # 只处理BSD400和WaterlooED
    folders = [
        '/root/data1/linziyue/RDDM/data/restoration/denoise/BSD400',
        '/root/data1/linziyue/RDDM/data/restoration/denoise/WaterlooED',
        '/root/data1/linziyue/RDDM/data/restoration/denoise/remote_sensing',
        '/root/data1/linziyue/RDDM/data/restoration/denoise/medical'
    ]
    sigmas = [15, 25, 50]
    root_dir = '/root/data1/linziyue/RDDM/data/restoration/denoise/multi_domain_noise'
    input_dir = os.path.join(root_dir, 'input')
    gt_dir = os.path.join(root_dir, 'gt')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # 存储flist路径
    input_list, gt_list = [], []

    for folder in folders:
        process_folder(folder, sigmas, input_dir, gt_dir, input_list, gt_list)

    # 写入flist
    input_flist = os.path.join(root_dir, 'input.flist')
    gt_flist = os.path.join(root_dir, 'gt.flist')
    with open(input_flist, 'w') as f_in, open(gt_flist, 'w') as f_gt:
        for i, g in zip(input_list, gt_list):
            f_in.write(i + '\n')
            f_gt.write(g + '\n')

    print(f"Done. Generated {len(input_list)} pairs.")
    print(f"input.flist: {input_flist}")
    print(f"gt.flist: {gt_flist}")

if __name__ == '__main__':
    main()


import os
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma / 255.0, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img

def process_folder(src_folder, sigmas, input_dir, gt_dir, input_list, gt_list, ratio=0.25):
    folder_name = os.path.basename(src_folder.rstrip('/'))
    img_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # ====== 新增：随机采样 ======
    num_to_sample = int(len(img_files) * ratio)
    if num_to_sample < 1:
        num_to_sample = 1  # 至少1张
    sampled_files = random.sample(img_files, num_to_sample)
    # ==========================

    for img_name in tqdm(sampled_files, desc=f'Processing {folder_name}'):
        src_path = os.path.join(src_folder, img_name)
        base_name = os.path.splitext(img_name)[0]

        # 读取图片
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: {src_path} cannot be read, skipping.")
            continue
        img_float = img.astype(np.float32) / 255.0

        # 保存gt（只保存一次）
        gt_out_name = f"{base_name}_{folder_name}.png"
        gt_save_path = os.path.join(gt_dir, gt_out_name)
        shutil.copy(src_path, gt_save_path)

        for sigma in sigmas:
            # 加噪
            noisy_img = add_gaussian_noise(img_float, sigma)
            noisy_img_uint8 = (noisy_img * 255.0).round().astype(np.uint8)
            input_out_name = f"{base_name}_{folder_name}_{sigma}.png"
            input_save_path = os.path.join(input_dir, input_out_name)
            cv2.imwrite(input_save_path, noisy_img_uint8)

            # 记录到list
            input_list.append(input_save_path)
            gt_list.append(gt_save_path)

def main():
    folders = [
        '/root/data1/linziyue/RDDM/data/restoration/denoise/BSD400',
        '/root/data1/linziyue/RDDM/data/restoration/denoise/WaterlooED',
        '/root/data1/linziyue/RDDM/data/restoration/denoise/remote_sensing',
        '/root/data1/linziyue/RDDM/data/restoration/denoise/medical'
    ]
    sigmas = [15, 25, 50]
    root_dir = '/root/data1/linziyue/RDDM/data/restoration/denoise/multi_domain_noise'
    input_dir = os.path.join(root_dir, 'input')
    gt_dir = os.path.join(root_dir, 'gt')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # 存储flist路径
    input_list, gt_list = [], []

    # ====== 新增：随机种子（可选）======
    random.seed(42)
    # ==================================

    for folder in folders:
        process_folder(folder, sigmas, input_dir, gt_dir, input_list, gt_list, ratio=1)  # <--- ratio=0.25

    # 写入flist
    input_flist = os.path.join(root_dir, 'input.flist')
    gt_flist = os.path.join(root_dir, 'gt.flist')
    with open(input_flist, 'w') as f_in, open(gt_flist, 'w') as f_gt:
        for i, g in zip(input_list, gt_list):
            f_in.write(i + '\n')
            f_gt.write(g + '\n')

    print(f"Done. Generated {len(input_list)} pairs.")
    print(f"input.flist: {input_flist}")
    print(f"gt.flist: {gt_flist}")

if __name__ == '__main__':
    main()