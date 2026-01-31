# Training Guide for TanColorize Model

## Setup

1. **Prerequisites**:
   - Ensure the virtual environment is set up and activated (or reference the python executable directly).
   - Ensure NVIDIA GPU is available (RTX 3050 confirmed).

2. **Verify dataset paths** in `options/train/train_tancolorize.yml`:
   - Training dataset: `C:\Users\nimet\Documents\IIT\L6\FYP\Dataset\SkinTan`
   - Validation dataset: `C:\Users\nimet\Documents\IIT\L6\FYP\Dataset\Datase_Validation`

## Training Command

Use the following command to start training. It sets the `PYTHONPATH` and uses the virtual environment's Python to ensure all dependencies (including CUDA/GPU support) are loaded correctly.

```powershell
$env:PYTHONPATH="C:\Users\nimet\Documents\IIT\L6\FYP\IPD\TanColorize-"
.\venv\Scripts\python.exe basicsr/train.py -opt options/train/train_tancolorize.yml
```

## Training Configuration

The training configuration is in `options/train/train_tancolorize.yml`:

- **Model**: TanColorize architecture
- **Training iterations**: 10,000 (configurable)
- **Batch size**: 4 per GPU
- **Learning rate**: 1e-5 for both generator and discriminator
- **Losses**: L1, Perceptual, GAN, and Colorfulness losses

## Output

Training outputs will be saved to:
- `experiments/TanColorize_v1/models/` - Model checkpoints
- `experiments/TanColorize_v1/training_states/` - Training state (for resuming)
- `experiments/TanColorize_v1/logs/` - Training logs
- `tb_logger/TanColorize_v1/` - TensorBoard logs (if enabled)

## Resume Training

To resume from a checkpoint:
```powershell
$env:PYTHONPATH="C:\Users\nimet\Documents\IIT\L6\FYP\IPD\TanColorize-"
.\venv\Scripts\python.exe basicsr/train.py -opt options/train/train_tancolorize.yml --auto_resume
```
