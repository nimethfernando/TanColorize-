## 🔧 About the Project

This project explores advanced neural network techniques to transform grayscale images into vibrant, colorized versions. We use supervised training with a curated dataset of human-centric images, focusing particularly on tan skin tone representation (TanVis dataset).

The primary goals of this project are:
- Enhance skin tone accuracy in grayscale-to-color transformations  
- Minimize color bleeding and artifact generation  
- Develop a robust and generalizable colorization model

---

## 🚀 Getting Started

To run this application locally, you will need to set up both the Python backend and the React frontend.

### Prerequisites
* **Python 3.9+**
* **Node.js & npm** (Download from [nodejs.org](https://nodejs.org/))

### 1. Clone the Repository

```bash
git clone 

# Create and activate a virtual environment (optional but recommended)
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies (including FastAPI and Uvicorn)
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart opencv-python torch torchvision basicsr

# Setup front end
cd frontend
npm install

# Start the Backend Server
uvicorn server:app --reload

# Start the React Frontend
cd frontend
npm start