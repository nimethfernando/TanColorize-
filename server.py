import os
import sys
import glob
import cv2
import torch
import numpy as np
import tempfile
import subprocess
import uvicorn
import base64
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage
import boto3
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Import your architecture (Ensure you run this script from the project root)
from basicsr.archs.tancolorize_arch import TanColorize

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "your-s3-bucket-name")

app = FastAPI()

# Enable CORS so React (port 3000) can talk to FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize S3 Client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

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

# --- Auto-load for Vertex AI ---
VERTEX_MODEL_PATH = os.getenv("VERTEX_MODEL_PATH")
if VERTEX_MODEL_PATH and os.path.exists(VERTEX_MODEL_PATH):
    print(f"Vertex AI Mode: Auto-loading model from {VERTEX_MODEL_PATH}")
    colorizer.load_model(VERTEX_MODEL_PATH, input_size=512)

# --- API Endpoints ---

class ModelConfig(BaseModel):
    path: str
    input_size: int

class FolderConfig(BaseModel):
    input_folder: str
    output_folder: str

@app.get("/browse")
def browse_path(type: str):
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
async def colorize_image(file: UploadFile = File(...), user_id: str = Form(None)):
    if colorizer.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        output_bgr = colorizer.colorize(img)
        
        # Encode back to png
        success, encoded_img = cv2.imencode('.png', output_bgr)
        image_bytes = encoded_img.tobytes()
        
        s3_url = None
        # If user is signed in, upload BOTH to S3
        if user_id:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_id = uuid.uuid4().hex[:8]
                
                # 1. Upload Original Image
                orig_s3_key = f"history/{user_id}/{timestamp}_{unique_id}_original.png"
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=orig_s3_key,
                    Body=contents,
                    ContentType=file.content_type or 'image/png'
                )

                # 2. Upload Colorized Image
                col_s3_key = f"history/{user_id}/{timestamp}_{unique_id}_colorized.png"
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=col_s3_key,
                    Body=image_bytes,
                    ContentType='image/png'
                )
                
                # Construct the public S3 URL
                s3_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{col_s3_key}"
                print(f"Successfully uploaded to S3: {s3_url}")
            except Exception as s3_err:
                print(f"Error uploading to S3: {s3_err}")
        
        return {
            "image_data": image_bytes.hex(),
            "s3_url": s3_url 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """Fetches and groups user colorization history from S3 with Pre-signed URLs."""
    try:
        prefix = f"history/{user_id}/"
        
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=prefix
        )
        
        grouped_files = {}
        if 'Contents' in response:
            for obj in response['Contents']:
                
                # 1. EXTRACT FILENAME FIRST
                filename = obj['Key'].split('/')[-1]

                # 2. GENERATE PRE-SIGNED URL WITH CONTENT-DISPOSITION
                file_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': S3_BUCKET_NAME,
                        'Key': obj['Key'],
                        'ResponseContentDisposition': f'attachment; filename="{filename}"' 
                    },
                    ExpiresIn=3600 # This link works for exactly 1 hour
                )
                
                # Group files by their timestamp and unique ID
                if "_original" in filename or "_colorized" in filename:
                    base_id = filename.replace("_original.png", "").replace("_colorized.png", "")
                    
                    if base_id not in grouped_files:
                        grouped_files[base_id] = {
                            "timestamp": base_id,
                            "original": None,
                            "colorized": None,
                            "last_modified": obj['LastModified']
                        }
                        
                    if "_original" in filename:
                        grouped_files[base_id]["original"] = file_url
                    elif "_colorized" in filename:
                        grouped_files[base_id]["colorized"] = file_url

        # Convert the dictionary to a list and sort by newest first
        files = list(grouped_files.values())
        files.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return {"history": files}
    except Exception as e:
        print(f"Error fetching history from S3: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Vertex AI Endpoints ---

@app.get("/health")
def health_check():
    """Vertex AI health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    """Vertex AI prediction endpoint."""
    if colorizer.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded.")

    try:
        body = await request.json()
        instances = body.get("instances", [])
        
        predictions = []
        for instance in instances:
            # Vertex AI typically sends images as base64 encoded strings
            image_b64 = instance.get("image_bytes")
            if not image_b64:
                continue
                
            image_data = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Run inference
            output_bgr = colorizer.colorize(img)
            
            # Encode result back to base64
            success, encoded_img = cv2.imencode('.png', output_bgr)
            result_b64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
            predictions.append({"colorized_image": result_b64})
            
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Use AIP_HTTP_PORT for Vertex AI, default to 8000 for local development
    port = int(os.environ.get("AIP_HTTP_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)