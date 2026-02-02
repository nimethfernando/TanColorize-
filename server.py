import os
import cv2
import torch
import numpy as np
import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

class ImageColorizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_size = 512
        # Try loading default model on startup, but don't crash if missing
        self.load_model("model.pth")

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}. Waiting for upload...")
            return False

        print(f"Loading model from {model_path}...")
        
        # Configuration must match your training
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
        
        try:
            self.model = TanColorize(**config)
            state_dict = torch.load(model_path, map_location=self.device)
            if "params" in state_dict:
                state_dict = state_dict["params"]
                
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    @torch.no_grad()
    def colorize(self, img_array):
        if self.model is None:
            raise ValueError("No model loaded! Please upload a model first.")
            
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

colorizer = ImageColorizer()

@app.get("/")
def root():
    return {"message": "TanColorize Backend Running", "model_loaded": colorizer.model is not None}

# --- NEW: Upload Model Endpoint ---
@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to disk
        file_location = "uploaded_model.pth"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        # Load this new model
        success = colorizer.load_model(file_location)
        
        if success:
            return {"message": f"Successfully loaded model: {file.filename}"}
        else:
            raise HTTPException(status_code=500, detail="File uploaded but failed to load. Is it a valid .pth file?")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/colorize-image")
async def colorize_image(file: UploadFile = File(...)):
    if colorizer.model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please upload a .pth file first.")
    
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)