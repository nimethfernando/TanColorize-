# TanColorize 

TanColorize is an AI-powered image colorization system designed to transform grayscale images into vibrant, realistic versions. A primary focus of this project is addressing algorithmic bias by ensuring accurate representation of tan and darker skin tones using the curated **SkinTan** dataset.

## About the Project

The project utilizes advanced neural network techniques, specifically the custom **TanColorize** architecture (integrating a **ConvNeXt-L** encoder and a **Transformer-based MultiScaleColorDecoder**), to achieve high-fidelity colorization.

**Primary Goals:**
* **Skin Tone Accuracy**: Enhance the representation of tan and dark skin tones in grayscale-to-color transformations.
* **Artifact Reduction**: Minimize color bleeding and artificial generation during the process.
* **Robustness**: Develop a generalizable model capable of handling diverse human-centric images.

---

## Getting Started

To run this application locally, you will need to set up the FastAPI backend and the React frontend.

### Prerequisites
* **Python 3.9+**
* **Node.js & npm**
* **NVIDIA GPU** (RTX 3050 or higher confirmed for training)
* **CUDA Toolkit** (compatible with your PyTorch installation)

### 1. Installation & Setup

```bash
# Clone the repository
git clone 
cd TanColorize

# Create and activate a virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart opencv-python torch torchvision basicsr google-cloud-storage

# Setup the frontend
cd frontend
npm install

2. Running the Application
You will need two separate terminal windows to run the backend and frontend simultaneously.

# Ensure your virtual environment is activated
cd <project-root>
uvicorn server:app --reload --host 0.0.0.0 --port 8000

The backend API will be available at http://localhost:8000

Start the React Frontend:
cd frontend
npm start
The web interface will automatically open at http://localhost:3000

🧠 Training Guide
Training is managed via the basicsr/train.py script using configuration files that define the network architecture and loss functions.

Dataset Configuration
Ensure your local dataset paths are correctly set in options/train/train_tancolorize.yml:

Training Dataset: Path to your curated human-centric images (e.g., Dataset/SkinTan).

Validation Dataset: Path to your validation set (e.g., Dataset/Dataset_Validation).
Execution
Run the following command to begin training. Make sure to set your PYTHONPATH to the root directory of the project so the custom architecture registers correctly.

Windows (PowerShell):

$env:PYTHONPATH="C:\Users\nimet\Documents\IIT\L6\FYP\IPD\TanColorize-"
.\venv\Scripts\python.exe basicsr/train.py -opt options/train/train_tancolorize.yml


Frontend URL
https://frontend-chi-bice-33.vercel.app/