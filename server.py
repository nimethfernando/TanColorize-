import os
import cv2
import torch
import numpy as np
import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import the architecture from your uploaded files
from basicsr.archs.tancolorize_arch import TanColorize

app = FastAPI()

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
        self.input_size = 128  # Matches training configuration

    def load_model(self, model_path: str):
        # Configuration must match your tancolorize_arch.py requirements
        config = {
            "encoder_name": "convnext-t",
            "encoder_from_pretrain": False,
            "num_queries": 32,
            "num_scales": 3,
            "dec_layers": 2,
            "num_output_channels": 2,
            "decoder_name": "MultiScaleColorDecoder",
            "last_norm": "Spectral",
            "do_normalize": False
        }

        model = TanColorize(**config)
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Pull params if it's a BasicSR/TanColorize checkpoint
        if "params" in state_dict:
            state_dict = state_dict["params"]
        elif "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]

        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        h, w = img_bgr.shape[:2]
        img = img_bgr.astype(np.float32) / 255.0
        
        # Pre-process image to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, :1]
        
        # Resize for model input
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        l_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)[:, :, :1]

        # Prepare input tensor (using RGB-fake as per inference logic)
        lab_fake = np.concatenate([l_resized, np.zeros_like(l_resized), np.zeros_like(l_resized)], axis=-1)
        rgb_fake = cv2.cvtColor(lab_fake, cv2.COLOR_LAB2RGB)
        tensor = torch.from_numpy(rgb_fake.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        # Model Inference
        ab_pred = self.model(tensor)
        
        # Upscale AB channels back to original size
        ab_pred = torch.nn.functional.interpolate(ab_pred, size=(h, w), mode="bilinear", align_corners=False)
        ab_pred = ab_pred[0].cpu().numpy().transpose(1, 2, 0)

        # Merge with original L and convert back
        lab_out = np.concatenate([l_channel, ab_pred], axis=-1)
        bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        return (bgr_out * 255).clip(0, 255).astype(np.uint8)

colorizer = ImageColorizer()
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".pth"):
        raise HTTPException(400, "Only .pth files allowed")
    
    path = os.path.join(MODEL_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        colorizer.load_model(path)
        return {"status": "success", "message": f"Loaded {file.filename}"}
    except Exception as e:
        raise HTTPException(500, f"Load Error: {str(e)}")

@app.post("/colorize-image")
async def colorize_image(file: UploadFile = File(...)):
    if not colorizer.model:
        raise HTTPException(400, "Model not loaded")
    
    buf = await file.read()
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    
    try:
        res = colorizer.colorize(img)
        _, enc = cv2.imencode(".png", res)
        return {"image_data": enc.tobytes().hex()}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)