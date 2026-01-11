import os
import io
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

from models.unet import TinyUNet
from utils.color import lab_to_rgb_tensor

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global model cache
model_cache = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path: str):
    """Load model with caching."""
    # Normalize checkpoint path to handle relative paths consistently
    if not checkpoint_path:
        checkpoint_path = None
    elif not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(os.path.normpath(checkpoint_path))
    else:
        checkpoint_path = os.path.normpath(checkpoint_path)
    
    # Use normalized path as cache key
    cache_key = checkpoint_path if checkpoint_path else "default"
    
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    model = TinyUNet(in_channels=1, out_channels=2, base_c=32)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
    model.to(device)
    model.eval()
    model_cache[cache_key] = model
    return model


def pil_to_L(pil_img: Image.Image, image_size: int) -> torch.Tensor:
    """Convert PIL image to L channel tensor."""
    img = pil_img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    img_np = np.array(img)
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0:1] / 255.0
    L_t = torch.from_numpy(L.transpose(2, 0, 1)).unsqueeze(0).float()
    return L_t


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'device': device
    })


@app.route('/api/colorize', methods=['POST'])
def colorize():
    """Colorize an uploaded image."""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get parameters
        checkpoint_path = request.form.get('checkpoint', 'checkpoints/model_epoch_20.pth')
        try:
            image_size = int(request.form.get('image_size', 256))
            if image_size <= 0 or image_size > 2048:
                return jsonify({'error': 'image_size must be between 1 and 2048'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid image_size parameter'}), 400
        
        # Load and process image
        # Read file content into bytes buffer
        file_content = file.read()
        if len(file_content) == 0:
            return jsonify({'error': 'Empty file provided'}), 400
        
        try:
            img = Image.open(io.BytesIO(file_content)).convert("RGB")
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Store original size for upscaling
        original_size = img.size  # (width, height)
        
        # Load model
        model = load_model(checkpoint_path)
        
        # Colorize at model input size
        L = pil_to_L(img, image_size).to(device)
        with torch.no_grad():
            pred_ab = model(L)
        
        # Upscale AB channels to original image size using bilinear interpolation
        if original_size != (image_size, image_size):
            # Upscale pred_ab using bilinear interpolation
            pred_ab_upscaled = F.interpolate(
                pred_ab, 
                size=(original_size[1], original_size[0]),  # (height, width)
                mode='bilinear', 
                align_corners=False
            )
            
            # Get original L channel at full resolution
            img_np = np.array(img)
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            lab_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            L_orig = lab_orig[:, :, 0:1] / 255.0
            L_orig_t = torch.from_numpy(L_orig.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            
            # Combine original L with upscaled AB
            rgb = lab_to_rgb_tensor(L_orig_t, pred_ab_upscaled).clamp(0, 1)
        else:
            # No upscaling needed
            rgb = lab_to_rgb_tensor(L, pred_ab).clamp(0, 1)
        
        # Convert to PIL image
        rgb_np = (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        # Ensure RGB mode explicitly
        out_img = Image.fromarray(rgb_np, mode='RGB')
        
        # Save to bytes with no compression to preserve quality
        buf = io.BytesIO()
        out_img.save(buf, format="PNG", optimize=False, compress_level=0)
        buf.seek(0)
        
        return send_file(buf, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


