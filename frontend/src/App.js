import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [originalFile, setOriginalFile] = useState(null);
  const [colorizedImage, setColorizedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageSize, setImageSize] = useState(256);
  const [checkpoint, setCheckpoint] = useState('checkpoints/model_epoch_20.pth');

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setOriginalFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setOriginalImage(reader.result);
        setColorizedImage(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleColorize = async () => {
    if (!originalFile) {
      setError('Please upload an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Create form data with the actual file
      const formData = new FormData();
      formData.append('image', originalFile);
      formData.append('image_size', imageSize.toString());
      formData.append('checkpoint', checkpoint);

      // Call API
      const apiResponse = await axios.post(`${API_URL}/api/colorize`, formData, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Convert blob to data URL
      const reader = new FileReader();
      reader.onloadend = () => {
        setColorizedImage(reader.result);
        setLoading(false);
      };
      reader.readAsDataURL(apiResponse.data);
    } catch (err) {
      let errorMessage = 'Failed to colorize image';
      
      if (err.code === 'ECONNREFUSED' || err.message.includes('Network Error')) {
        errorMessage = 'Cannot connect to API server. Make sure the Flask API is running on http://localhost:5000';
      } else if (err.response) {
        // Try to parse error message from response
        if (err.response.data instanceof Blob) {
          errorMessage = `Server error: ${err.response.status} ${err.response.statusText}`;
        } else {
          errorMessage = err.response.data?.error || `Server error: ${err.response.status}`;
        }
      } else {
        errorMessage = err.message || errorMessage;
      }
      
      setError(errorMessage);
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (colorizedImage) {
      const link = document.createElement('a');
      link.href = colorizedImage;
      link.download = 'colorized.png';
      link.click();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎨 Image Colorizer</h1>
        <p className="subtitle">Transform grayscale images into vibrant color</p>
      </header>

      <main className="App-main">
        <div className="controls">
          <div className="control-group">
            <label htmlFor="image-upload" className="upload-button">
              📁 Upload Image
            </label>
            <input
              id="image-upload"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              style={{ display: 'none' }}
            />
          </div>

          <div className="control-group">
            <label htmlFor="image-size">Image Size: {imageSize}px</label>
            <input
              id="image-size"
              type="range"
              min="128"
              max="512"
              step="64"
              value={imageSize}
              onChange={(e) => setImageSize(parseInt(e.target.value))}
            />
          </div>

          <div className="control-group">
            <label htmlFor="checkpoint">Checkpoint Path:</label>
            <input
              id="checkpoint"
              type="text"
              value={checkpoint}
              onChange={(e) => setCheckpoint(e.target.value)}
              placeholder="checkpoints/model_epoch_20.pth"
            />
          </div>

          <button
            className="colorize-button"
            onClick={handleColorize}
            disabled={!originalImage || loading}
          >
            {loading ? '⏳ Colorizing...' : '✨ Colorize Image'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        <div className="image-container">
          <div className="image-box">
            <h3>Original Image</h3>
            {originalImage ? (
              <img src={originalImage} alt="Original" className="preview-image" />
            ) : (
              <div className="placeholder">No image uploaded</div>
            )}
          </div>

          <div className="image-box">
            <h3>Colorized Image</h3>
            {loading ? (
              <div className="loading-spinner">
                <div className="spinner"></div>
                <p>Processing...</p>
              </div>
            ) : colorizedImage ? (
              <>
                <img src={colorizedImage} alt="Colorized" className="preview-image" />
                <button className="download-button" onClick={handleDownload}>
                  💾 Download
                </button>
              </>
            ) : (
              <div className="placeholder">Colorized result will appear here</div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
