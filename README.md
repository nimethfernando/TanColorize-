
# Simple Colorizer

A minimal grayscale-to-color training and inference pipeline.

## Install

```
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r simple_colorizer/requirements.txt
```

## Prepare data
- Put your RGB training images under a folder, e.g. `D:/data/train_rgb/`
- (Optional) Validation images under `D:/data/val_rgb/`

## Train
```
python simple_colorizer/train.py --train_dir D:/data/train_rgb --val_dir D:/data/val_rgb --save_dir simple_colorizer/checkpoints --image_size 256 --batch_size 8 --epochs 20
```

This trains a tiny U-Net to predict AB from L in LAB space.

## Inference Apps

### Streamlit App
```
python -m streamlit run web_app\app.py --server.port 8501 --server.headless true
```

- In the sidebar, set your checkpoint path (e.g., `checkpoints/model_epoch_20.pth`).
- Upload an image and view the colorized result.

### React Frontend with Flask API

1. **Start the Flask API backend:**
```bash
python api.py
```
The API will run on `http://localhost:5000`

2. **Start the React frontend:**
```bash
cd frontend
npm install
npm start
```
The React app will open at `http://localhost:3000`

- Upload an image using the web interface
- Adjust image size and checkpoint path as needed
- Click "Colorize Image" to process
- Download the colorized result

