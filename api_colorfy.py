import os
import io
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

# Import Colorfy model (assuming basicsr is installed)
try:
    from basicsr.archs.colorfy_arch import Colorfy
except ImportError:
    Colorfy = None
    print("Warning: basicsr not found. Please install it: pip install basicsr")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global model cache
model_cache = {}
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_colorfy_model(checkpoint_path: str, input_size: int = 512):
    """Load Colorfy model with caching."""
    # Normalize checkpoint path
    if not checkpoint_path:
        checkpoint_path = None
    elif not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(os.path.normpath(checkpoint_path))
    else:
        checkpoint_path = os.path.normpath(checkpoint_path)
    
    # Use normalized path and input_size as cache key
    cache_key = f"{checkpoint_path}_{input_size}" if checkpoint_path else f"default_{input_size}"
    
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    if Colorfy is None:
        raise ImportError("basicsr package not installed. Please install it: pip install basicsr")
    
    # Model configuration
    config = {
        "encoder_name": "convnext-l",
        "decoder_name": "MultiScaleColorDecoder",
        "input_size": [input_size, input_size],
        "num_output_channels": 2,
        "last_norm": "Spectral",
        "do_normalize": False,
        "num_queries": 100,
        "num_scales": 3,
        "dec_layers": 9,
    }
    
    model = Colorfy(**config)
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            
            # Extract params if needed
            if "params" in state_dict:
                state_dict = state_dict["params"]
            
            # Filter compatible weights
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
            
            print(f"Loaded {len(filtered_dict)} of {len(state_dict)} layers")
            model.load_state_dict(filtered_dict, strict=False)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
    
    model.to(device)
    model.eval()
    model_cache[cache_key] = model
    return model


def colorize_image_colorfy(model, img_pil, input_size):
    """Colorize an image using Colorfy model."""
    # Convert PIL to OpenCV format (BGR)
    img_np = np.array(img_pil)
    if len(img_np.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Save original dimensions and extract L channel
    height, width = img.shape[:2]
    img_float = img.astype(np.float32) / 255.0
    orig_l = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)[:, :, :1]
    
    # Resize and prepare for model
    img_resized = cv2.resize(img_float, (input_size, input_size))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    
    # Process with model
    tensor = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_ab = model(tensor).cpu()
    
    # Resize output and combine with original L channel
    output_ab = F.interpolate(
        output_ab, 
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )[0].float().numpy().transpose(1, 2, 0)
    
    # Combine L with predicted AB
    output_lab = np.concatenate((orig_l, output_ab), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    
    # Convert to uint8 and then to RGB for PIL
    output_bgr_uint8 = (output_bgr * 255.0).round().astype(np.uint8)
    output_rgb = cv2.cvtColor(output_bgr_uint8, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(output_rgb, mode='RGB')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'device': device,
        'model_type': 'Colorfy'
    })


@app.route('/api/colorize', methods=['POST'])
def colorize():
    """Colorize an uploaded image using Colorfy model."""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get parameters
        checkpoint_path = request.form.get('checkpoint', '')
        if not checkpoint_path:
            return jsonify({'error': 'checkpoint parameter is required'}), 400
        
        try:
            input_size = int(request.form.get('input_size', 512))
            if input_size <= 0 or input_size > 2048:
                return jsonify({'error': 'input_size must be between 1 and 2048'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid input_size parameter'}), 400
        
        # Load and process image
        file_content = file.read()
        if len(file_content) == 0:
            return jsonify({'error': 'Empty file provided'}), 400
        
        try:
            img = Image.open(io.BytesIO(file_content)).convert("RGB")
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Load model
        model = load_colorfy_model(checkpoint_path, input_size)
        
        # Colorize image
        colored_img = colorize_image_colorfy(model, img, input_size)
        
        # Save to bytes with no compression to preserve quality
        buf = io.BytesIO()
        colored_img.save(buf, format="PNG", optimize=False, compress_level=0)
        buf.seek(0)
        
        return send_file(buf, mimetype='image/png')
    
    except ImportError as e:
        return jsonify({'error': f'Model import error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Using port 5001 to avoid conflict with existing API
