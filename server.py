import os
import cv2
import torch
import numpy as np
import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# TanColorize architecture
from basicsr.archs.tancolorize_arch import TanColorize

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Image Colorizer
# -----------------------------------------------------------------------------
class ImageColorizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_size = 128  # matches training gt_size

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found")

        config = {
            "encoder_name": "convnext-t",              # MUST match training
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
        if "params" in state_dict:
            state_dict = state_dict["params"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print("⚠ Missing keys:", missing[:10])
        if len(unexpected) > 0:
            print("⚠ Unexpected keys:", unexpected[:10])

        model.to(self.device)
        model.eval()

        self.model = model
        print("✅ Model loaded successfully")

    @torch.no_grad()
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if img_bgr is None:
            raise ValueError("Invalid image")

        h, w = img_bgr.shape[:2]

        img = img_bgr.astype(np.float32) / 255.0
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        l = lab[:, :, :1]  # original L
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        l_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)[:, :, :1]

        # Fake ab for model input (as used in TanColorize inference)
        lab_fake = np.concatenate(
            [l_resized, np.zeros_like(l_resized), np.zeros_like(l_resized)],
            axis=-1
        )
        rgb_fake = cv2.cvtColor(lab_fake, cv2.COLOR_LAB2RGB)

        tensor = torch.from_numpy(rgb_fake.transpose(2, 0, 1)) \
                      .unsqueeze(0).float().to(self.device)

        ab = self.model(tensor)
        ab = torch.nn.functional.interpolate(
            ab, size=(h, w), mode="bilinear", align_corners=False
        )[0].cpu().numpy().transpose(1, 2, 0)

        lab_out = np.concatenate([l, ab], axis=-1)
        bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

        return (bgr_out * 255).clip(0, 255).astype(np.uint8)

# -----------------------------------------------------------------------------
# Global instance (kept in memory)
# -----------------------------------------------------------------------------
colorizer = ImageColorizer()

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "running",
        "model_loaded": colorizer.model is not None,
        "device": str(colorizer.device)
    }

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".pth"):
        raise HTTPException(400, "Only .pth files allowed")

    model_path = os.path.join(MODEL_DIR, file.filename)

    with open(model_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        colorizer.load_model(model_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {str(e)}")

    return {
        "status": "success",
        "message": "Model uploaded and loaded successfully"
    }

@app.post("/colorize-image")
async def colorize_image(file: UploadFile = File(...)):
    if colorizer.model is None:
        raise HTTPException(400, "No model loaded")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "Invalid image file")

    try:
        output = colorizer.colorize(img)
        success, encoded = cv2.imencode(".png", output)
        if not success:
            raise RuntimeError("Image encoding failed")

        return {"image_data": encoded.tobytes().hex()}
    except Exception as e:
        raise HTTPException(500, str(e))

# -----------------------------------------------------------------------------
# Run (Render uses PORT env var)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
