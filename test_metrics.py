import cv2
import os
import glob
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

# 1. Set the paths to your folders
gt_folder = 'evaluation_dataset/ground_truth' # Folder with original color images
gen_folder = 'evaluation_dataset/generated'   # Folder with your model's outputs

# 2. Get lists of all images
gt_images = sorted(glob.glob(os.path.join(gt_folder, '*.jpg'))) # adjust extension if using .png
gen_images = sorted(glob.glob(os.path.join(gen_folder, '*.jpg')))

total_psnr = 0
total_ssim = 0
num_images = len(gt_images)

print(f"Testing {num_images} images...")

# 3. Loop through and calculate metrics for all images
for gt_path, gen_path in zip(gt_images, gen_images):
    img_gt = cv2.imread(gt_path)
    img_gen = cv2.imread(gen_path)
    
    # Ensure they match in size
    if img_gt.shape != img_gen.shape:
        print(f"Skipping {gt_path} due to shape mismatch.")
        num_images -= 1
        continue
        
    total_psnr += calculate_psnr(img_gt, img_gen, crop_border=0, input_order='HWC')
    total_ssim += calculate_ssim(img_gt, img_gen, crop_border=0, input_order='HWC')

# 4. Calculate and print the averages
if num_images > 0:
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
else:
    print("No valid images found to compare.")