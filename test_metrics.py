import cv2
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

# 1. Load your Ground Truth (original color) image and Generated (colorized) image.
# cv2.imread loads images as numpy arrays with values in the range [0, 255]
img_gt = cv2.imread('IMG_5865 1.jpg')
img_generated = cv2.imread('colorized.jpg')


# Note: Ensure both images have the exact same shape (width, height, and channels)
if img_gt.shape != img_generated.shape:
    print("Error: Image shapes do not match!")
else:
    # 2. Define the crop_border. 
    # Use 0 if you want to evaluate the entire image without ignoring the edges.
    crop_border = 0

    # 3. Calculate PSNR
    # The functions expect images with range [0, 255] and default input order 'HWC'
    psnr_score = calculate_psnr(img_gt, img_generated, crop_border=crop_border, input_order='HWC')
    print(f"PSNR Score: {psnr_score:.2f}")

    # 4. Calculate SSIM
    ssim_score = calculate_ssim(img_gt, img_generated, crop_border=crop_border, input_order='HWC')
    print(f"SSIM Score: {ssim_score:.4f}")