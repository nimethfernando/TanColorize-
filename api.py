import os
import io
import torch
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
    if checkpoint_path in model_cache:
        return model_cache[checkpoint_path]
    
    model = TinyUNet(in_channels=1, out_channels=2, base_c=32)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    model_cache[checkpoint_path] = model
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
        image_size = int(request.form.get('image_size', 256))
        
        # Load and process image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Load model
        model = load_model(checkpoint_path)
        
        # Colorize
        L = pil_to_L(img, image_size).to(device)
        with torch.no_grad():
            pred_ab = model(L)
            rgb = lab_to_rgb_tensor(L, pred_ab).clamp(0, 1)
        
        # Convert to PIL image
        rgb_np = (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        out_img = Image.fromarray(rgb_np)
        
        # Save to bytes
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        buf.seek(0)
        
        return send_file(buf, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


