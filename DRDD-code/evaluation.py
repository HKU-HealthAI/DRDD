import os
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import argparse
import lpips  
from metrics.niqe import calculate_niqe

def calculate_psnr(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    """Calculate PSNR between two images."""
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
    
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    img1 = reorder_image(img1, input_order)
    img2 = reorder_image(img2, input_order)

    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    max_value = 1. if img1.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))


from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim_skimage
import numpy as np
import torch

def calculate_ssim(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    """Calculate SSIM between two images using skimage implementation."""
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}'
    
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    img1 = reorder_image(img1, input_order)
    img2 = reorder_image(img2, input_order)

    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        return ssim_skimage(img1[..., 0], img2[..., 0], data_range=img1.max() - img1.min())

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    max_value = 1 if img1.max() <= 1 else 255
    
    if img1.ndim == 3 and img1.shape[2] > 1:
        return ssim_skimage(img1, img2, data_range=max_value, channel_axis=-1)
    else:
        return ssim_skimage(img1, img2, data_range=max_value)



def calculate_lpips(img1, img2, lpips_model, device='cuda'):
    """Calculate LPIPS between two images."""
    if img1.max() <= 1.0:
        img1 = (img1 * 255.0).astype(np.uint8)
    if img2.max() <= 1.0:
        img2 = (img2 * 255.0).astype(np.uint8)
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    img1_tensor = torch.from_numpy(img1_rgb).permute(2, 0, 1).float() / 255.0 * 2 - 1
    img2_tensor = torch.from_numpy(img2_rgb).permute(2, 0, 1).float() / 255.0 * 2 - 1
    
    img1_tensor = img1_tensor.unsqueeze(0).to(device)
    img2_tensor = img2_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        lpips_score = lpips_model(img1_tensor, img2_tensor)
    
    return lpips_score.item()

def _ssim_cly(img1, img2):
    """Calculate SSIM for one channel images."""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def _ssim_3d(img1, img2, max_value):
    """Calculate SSIM for 3D images (color images)."""
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if torch.cuda.is_available():
        kernel = _generate_3d_gaussian_kernel().cuda()
        img1 = torch.tensor(img1).float().cuda()
        img2 = torch.tensor(img2).float().cuda()
    else:
        kernel = _generate_3d_gaussian_kernel().cpu()
        img1 = torch.tensor(img1).float().cpu()
        img2 = torch.tensor(img2).float().cpu()

    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1*img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def _3d_gaussian_calculator(img, conv3d):
    """Apply 3D Gaussian filter."""
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out

def _generate_3d_gaussian_kernel():
    """Generate 3D Gaussian kernel."""
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d

def crop_img(img, base=16):
    if base <= 1:
        return img
    
    h, w = img.shape[:2]
    new_h = h - h % base
    new_w = w - w % base
    
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    
    if len(img.shape) == 3:
        return img[top:top+new_h, left:left+new_w, :]
    else:
        return img[top:top+new_h, left:left+new_w]

def align_image_sizes(img1, img2, crop_base=16):
    """Align two images to have the same size."""
    if crop_base > 1:
        img1 = crop_img(img1, crop_base)
        img2 = crop_img(img2, crop_base)
    
    if img1.shape[:2] != img2.shape[:2]:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        
        h1, w1 = img1.shape[:2]
        top1 = (h1 - min_h) // 2
        left1 = (w1 - min_w) // 2
        if len(img1.shape) == 3:
            img1 = img1[top1:top1+min_h, left1:left1+min_w, :]
        else:
            img1 = img1[top1:top1+min_h, left1:left1+min_w]
        
        h2, w2 = img2.shape[:2]
        top2 = (h2 - min_h) // 2
        left2 = (w2 - min_w) // 2
        if len(img2.shape) == 3:
            img2 = img2[top2:top2+min_h, left2:left2+min_w, :]
        else:
            img2 = img2[top2:top2+min_h, left2:left2+min_w]
    
    return img1, img2

def find_image_pairs(gt_dir, pred_dir):
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
    gt_images = []
    pred_images = []

    for ext in image_extensions:
        gt_images.extend(glob.glob(os.path.join(gt_dir, f'*.{ext}')))
        gt_images.extend(glob.glob(os.path.join(gt_dir, f'*.{ext.upper()}')))
    gt_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_images}

    for ext in image_extensions:
        pred_images.extend(glob.glob(os.path.join(pred_dir, f'*.{ext}')))
        pred_images.extend(glob.glob(os.path.join(pred_dir, f'*.{ext.upper()}')))

    pairs = []
    for pred_path in pred_images:
        pred_name = os.path.splitext(os.path.basename(pred_path))[0]
        if pred_name.endswith('_second_last'):
            pred_name = pred_name[:-len('_second_last')]
        key = pred_name.split('_')[0] 
        if key in gt_dict:
            gt_path = gt_dict[key]
            pairs.append((gt_path, pred_path))
        else:
            print(f"Warning: No GT image found for pred {pred_name}")

    return pairs

def reorder_image(img, input_order='HWC'):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    if input_order == 'CHW' and len(img.shape) == 3:
        img = img.transpose(1, 2, 0)
    return img

def to_y_channel(img):
    if img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img

def bgr2ycbcr(img, y_only=False):
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, '
                        f'but got {img_type}')
    return img

def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type."""
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def process_images(image_pairs, crop_border=0, test_y_channel=False, crop_base=16, 
                  lpips_model=None, device='cuda', calculate_niqe_flag=False):
    """Process images with multiple metrics."""
    results = []
    
    for gt_path, pred_path in tqdm(image_pairs, desc="Processing images"):
        try:
            gt_img = cv2.imread(gt_path)
            pred_img = cv2.imread(pred_path)
            
            if gt_img is None or pred_img is None:
                continue
            
            original_gt_shape = gt_img.shape
            original_pred_shape = pred_img.shape
            gt_img, pred_img = align_image_sizes(gt_img, pred_img, crop_base)
            
            psnr = calculate_psnr(gt_img, pred_img, crop_border, test_y_channel=test_y_channel)
            ssim_val = calculate_ssim(gt_img, pred_img, crop_border, test_y_channel=test_y_channel)
            
            lpips_score = None
            if lpips_model is not None:
                lpips_score = calculate_lpips(gt_img, pred_img, lpips_model, device)
            
            niqe_score = None
            if calculate_niqe_flag:
                niqe_score = calculate_niqe(pred_img, crop_border)
            
            result = {
                'image_name': os.path.basename(gt_path),
                'psnr': psnr,
                'ssim': ssim_val,
                'gt_shape': f"{original_gt_shape[1]}x{original_gt_shape[0]}",
                'pred_shape': f"{original_pred_shape[1]}x{original_pred_shape[0]}",
                'processed_shape': f"{gt_img.shape[1]}x{gt_img.shape[0]}"
            }
            
            if lpips_score is not None:
                result['lpips'] = lpips_score
            if niqe_score is not None:
                result['niqe'] = niqe_score
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {gt_path}: {str(e)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Image Quality Assessment (PSNR, SSIM, LPIPS & NIQE)')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground truth images directory')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to predicted images directory')
    parser.add_argument('--output', type=str, default='quality_metrics.csv', help='Output CSV file path')
    parser.add_argument('--crop_border', type=int, default=0, help='Number of pixels to crop from each border')
    parser.add_argument('--test_y_channel', action='store_true', help='Test on Y channel of YCbCr')
    parser.add_argument('--crop_base', type=int, default=16, help='Crop base to ensure image size divisible by this number')
    parser.add_argument('--calculate_lpips', action='store_true', default=True, help='Calculate LPIPS metric')
    parser.add_argument('--calculate_niqe', action='store_true', default=False, help='Calculate NIQE metric')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for computation')
    
    args = parser.parse_args()

    if not os.path.isdir(args.gt_dir) or not os.path.isdir(args.pred_dir):
        print("Error: Directories do not exist")
        return

    print("\nImage Quality Assessment Tool")
    print(f"GT Directory: {args.gt_dir}")
    print(f"Predicted Directory: {args.pred_dir}")
    print(f"Crop Border: {args.crop_border} pixels")
    print(f"Test Y Channel: {args.test_y_channel}")
    print(f"Crop Base: {args.crop_base}")
    print(f"Calculate LPIPS: {args.calculate_lpips}")
    print(f"Calculate NIQE: {args.calculate_niqe}")
    print(f"Device: {args.device}")
    print()

    lpips_model = None
    if args.calculate_lpips:
        try:
            lpips_model = lpips.LPIPS(net='alex').to(args.device)
            lpips_model.eval()
        except Exception as e:
            print(f"Error initializing LPIPS: {e}")
            args.calculate_lpips = False

    image_pairs = find_image_pairs(args.gt_dir, args.pred_dir)
    if not image_pairs:
        print("No matching image pairs found")
        return

    print(f"Found {len(image_pairs)} image pairs for evaluation")

    results = process_images(
        image_pairs,
        crop_border=args.crop_border,
        test_y_channel=args.test_y_channel,
        crop_base=args.crop_base,
        lpips_model=lpips_model,
        device=args.device,
        calculate_niqe_flag=args.calculate_niqe
    )

    if not results:
        print("No valid results")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Results saved to: {args.output}")

    metrics = ['psnr', 'ssim']
    if args.calculate_lpips and 'lpips' in df.columns:
        metrics.append('lpips')
    if args.calculate_niqe and 'niqe' in df.columns:
        metrics.append('niqe')

    print("\nSummary Statistics:")
    for metric in metrics:
        avg_value = float(df[metric].mean()) if metric == 'niqe' else df[metric].mean()
        print(f"Average {metric.upper()}: {avg_value:.4f}")
    
    print(f"Number of images evaluated: {len(df)}")

if __name__ == "__main__":
    main()
