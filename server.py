import os
import sys
import glob
import cv2
import torch
import numpy as np
import tempfile
import subprocess
import uvicorn
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage

# Import your architecture
from basicsr.archs.tancolorize_arch import TanColorize

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEW: Defined the download function ---
def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a model file from Google Cloud Storage."""
    # This will automatically use the GOOGLE_APPLICATION_CREDENTIALS env var
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    print(f"Downloading model {source_blob_name} from bucket {bucket_name}...")
    blob.download_to_filename(destination_file_name)
    print("Download complete.")

class ImageColorizer:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 512

    def load_model(self, model_path, input_size=512):
        self.input_size = input_size
        config = {
            "encoder_name": "convnext-l",
            "decoder_name": "MultiScaleColorDecoder",
            "input_size": (self.input_size, self.input_size),
            "num_output_channels": 2,
            "last_norm": "Spectral",
            "do_normalize": False,
            "num_queries": 100,
            "num_scales": 3,
            "dec_layers": 9,
        }
        
        model = TanColorize(**config)
        state_dict = torch.load(model_path, map_location=self.device)
        if "params" in state_dict:
            state_dict = state_dict["params"]
            
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        model.load_state_dict(filtered_dict, strict=False)
        self.model = model.to(self.device)
        self.model.eval()
        return f"Model loaded: {len(filtered_dict)} layers"

    @torch.no_grad()
    def colorize(self, img_array):
        if self.model is None:
            raise ValueError("Model not loaded")
            
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            
        height, width = img_array.shape[:2]
        img_float = img_array.astype(np.float32) / 255.0
        orig_l = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)[:, :, :1]
        
        img_resized = cv2.resize(img_float, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        tensor = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor).cpu()
        
        output_ab = torch.nn.functional.interpolate(output_ab, size=(height, width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        
        return (output_bgr * 255.0).round().astype(np.uint8)

# Initialize Global Colorizer
colorizer = ImageColorizer()

import requests

import os
import requests
# ... other imports ...

# 1. Define the download function FIRST
def download_model_simple():
    url = "https://storage.googleapis.com/tancorize/np.pth"
    destination = "model.pth"
    
    if not os.path.exists(destination):
        print(f"Downloading model from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Check for errors
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download model: {e}")

# 2. Call the function BEFORE the model tries to load
download_model_simple()

# 3. Now initialize your colorizer
LOCAL_MODEL_PATH = "model.pth"
# ... rest of your code ...

@app.post("/colorize-image")
async def colorize_image(file: UploadFile = File(...)):
    if colorizer.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        output_bgr = colorizer.colorize(img)
        success, encoded_img = cv2.imencode('.png', output_bgr)
        return {"image_data": encoded_img.tobytes().hex()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Use environment variable for port if available (standard for Render/Heroku)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)