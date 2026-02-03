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
import os

# Import your architecture (Ensure you run this script from the project root)
from basicsr.archs.tancolorize_arch import TanColorize

app = FastAPI()

# Enable CORS so React (port 3000) can talk to FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utilities from your original app.py ---

def select_file_using_subprocess(is_folder=False):
    """Opens a native system dialog on the server machine to select files/folders."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    temp_file.close()
    
    script_content = b"""
import tkinter as tk
from tkinter import filedialog
import sys

def select_path(output_file, is_folder):
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    
    if is_folder:
        path = filedialog.askdirectory(title="Select Folder")
    else:
        path = filedialog.askopenfilename(title="Select Model File", filetypes=(("PyTorch files", "*.pth"), ("All files", "*.*")))
    
    root.destroy()
    with open(output_file, 'w') as f:
        f.write(path)

if __name__ == "__main__":
    select_path(sys.argv[1], sys.argv[2].lower() == 'true')
"""
    script_file = tempfile.NamedTemporaryFile(delete=False, suffix='.py')
    script_file.write(script_content)
    script_file.close()
    
    try:
        subprocess.run(["python", script_file.name, temp_file.name, str(is_folder)], check=True)
        with open(temp_file.name, 'r') as f:
            path = f.read().strip()
        os.unlink(temp_file.name)
        os.unlink(script_file.name)
        return path
    except Exception as e:
        print(f"Error: {e}")
        return None

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

# --- API Endpoints ---

class ModelConfig(BaseModel):
    path: str
    input_size: int

class FolderConfig(BaseModel):
    input_folder: str
    output_folder: str

@app.get("/browse")
def browse_path(type: str):
    # Opens dialog on the server machine
    is_folder = (type == "folder")
    path = select_file_using_subprocess(is_folder)
    return {"path": path if path else ""}

@app.post("/load-model")
def load_model_endpoint(config: ModelConfig):
    try:
        msg = colorizer.load_model(config.path, config.input_size)
        return {"status": "success", "message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/colorize-image")
async def colorize_image(file: UploadFile = File(...)):
    if colorizer.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        output_bgr = colorizer.colorize(img)
        
        # Encode back to png to send to React
        success, encoded_img = cv2.imencode('.png', output_bgr)
        return {"image_data": encoded_img.tobytes().hex()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Updated Configuration ---
BUCKET_NAME = "tancorize"  # Extracted from your link
MODEL_BLOB_NAME = "np.pth" # Extracted from your link
LOCAL_MODEL_PATH = "model.pth"

# 1. Download from GCS on startup if it doesn't exist
if not os.path.exists(LOCAL_MODEL_PATH):
    try:
        download_model_from_gcs(BUCKET_NAME, MODEL_BLOB_NAME, LOCAL_MODEL_PATH)
    except Exception as e:
        print(f"Error downloading from GCS: {e}")

# 2. AUTOMATICALLY LOAD THE MODEL
# This ensures colorizer.model is not None when the user uploads an image
try:
    if os.path.exists(LOCAL_MODEL_PATH):
        # We use 512 as that is the default input_size in your ImageColorizer
        load_msg = colorizer.load_model(LOCAL_MODEL_PATH, input_size=512)
        print(f"Successfully auto-loaded: {load_msg}")
    else:
        print("Warning: model.pth not found. Automated colorization will fail until loaded.")
except Exception as e:
    print(f"Failed to initialize model on startup: {e}")

@app.post("/process-folder")
def process_folder(config: FolderConfig):
    if colorizer.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    if not os.path.exists(config.input_folder):
        raise HTTPException(status_code=404, detail="Input folder not found")
        
    os.makedirs(config.output_folder, exist_ok=True)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(config.input_folder, ext)))
    
    count = 0
    for img_path in image_files:
        try:
            img = cv2.imread(img_path)
            output_bgr = colorizer.colorize(img)
            save_path = os.path.join(config.output_folder, f"colorized_{os.path.basename(img_path)}")
            cv2.imwrite(save_path, output_bgr)
            count += 1
        except Exception as e:
            print(f"Skipped {img_path}: {e}")
            
    return {"status": "success", "processed_count": count, "output_folder": config.output_folder}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)