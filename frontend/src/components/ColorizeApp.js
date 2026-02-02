import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import '../App.css';

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function ColorizeApp() {
  const { currentUser, logout } = useAuth();
  
  // State
  const [inputSize, setInputSize] = useState(512);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");

  // Model Upload State
  const [modelFile, setModelFile] = useState(null);
  const [modelUploadStatus, setModelUploadStatus] = useState("");

  // Single Image State
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalPreview, setOriginalPreview] = useState(null);
  const [colorizedImage, setColorizedImage] = useState(null);

  // --- NEW: Handle Model Upload ---
  const handleModelFileChange = (e) => {
    if (e.target.files[0]) {
      setModelFile(e.target.files[0]);
      setModelUploadStatus(""); // Clear status when new file selected
    }
  };

  const handleUploadModel = async () => {
    if (!modelFile) return alert("Please select a .pth file first");
    
    setLoading(true);
    setModelUploadStatus("Uploading model... please wait.");
    
    const formData = new FormData();
    formData.append("file", modelFile);

    try {
      const res = await axios.post(`${API_URL}/upload-model`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setModelUploadStatus("✅ " + res.data.message);
    } catch (err) {
      console.error(err);
      setModelUploadStatus("❌ Error uploading model. Check console.");
    } finally {
      setLoading(false);
    }
  };

  // --- Handle Image Colorization ---
  const processSingleImage = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setStatusMsg("Processing image...");
    
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await axios.post(`${API_URL}/colorize-image`, formData);
      
      // Convert hex string back to image for display
      const hex = res.data.image_data;
      const bytes = new Uint8Array(hex.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
      const blob = new Blob([bytes], { type: "image/png" });
      setColorizedImage(URL.createObjectURL(blob));
      setStatusMsg("");
    } catch (err) {
      console.error(err);
      const errorMsg = err.response?.data?.detail || "Error processing image. Is a model loaded?";
      setStatusMsg("❌ " + errorMsg);
      alert(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  // Helper: Handle file input change
  const onFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setOriginalPreview(URL.createObjectURL(file));
      setColorizedImage(null);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Failed to logout:', error);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="user-info">
          <div className="user-email">
            {currentUser ? currentUser.email : 'Guest user'}
          </div>
          {currentUser ? (
            <button onClick={handleLogout} className="logout-btn">Logout</button>
          ) : (
            <div className="auth-actions">
              <Link to="/signin" className="logout-btn">Sign In</Link>
              <Link to="/signup" className="logout-btn">Sign Up</Link>
            </div>
          )}
        </div>
        
        <h2>Model Settings</h2>
        
        {/* --- NEW UPLOAD SECTION --- */}
        <div className="control-group" style={{ borderBottom: '1px solid #444', paddingBottom: '20px', marginBottom: '20px' }}>
          <label>1. Upload Custom Model (.pth)</label>
          <div className="file-input-group" style={{ flexDirection: 'column', gap: '10px' }}>
            <input 
              type="file" 
              accept=".pth"
              onChange={handleModelFileChange}
              style={{ color: 'white' }}
            />
            <button 
              className="primary-btn" 
              onClick={handleUploadModel}
              disabled={loading || !modelFile}
            >
              {loading && modelUploadStatus.includes("Uploading") ? "Uploading..." : "Upload & Load Model"}
            </button>
          </div>
          <p className="status-text" style={{ marginTop: '10px', fontSize: '0.9em' }}>
            {modelUploadStatus}
          </p>
        </div>

        <div className="control-group">
          <label>Input Size: {inputSize}</label>
          <input 
            type="range" min="256" max="1024" step="64" 
            value={inputSize} 
            onChange={(e) => setInputSize(parseInt(e.target.value))} 
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <h1>🎨 Tancolorize Cloud</h1>
        
        <div className="single-mode">
          <div className="upload-section">
            <input type="file" onChange={onFileChange} accept="image/*" />
            <button 
              className="action-btn"
              onClick={processSingleImage} 
              disabled={!selectedFile || loading}
            >
              {loading && !modelUploadStatus.includes("Uploading") ? "Processing..." : "Colorize Image"}
            </button>
          </div>
          
          <p style={{ textAlign: 'center', color: '#ff6b6b' }}>{statusMsg}</p>

          <div className="image-preview-container">
            <div className="img-box">
              <h4>Original</h4>
              {originalPreview && <img src={originalPreview} alt="Original" />}
            </div>
            <div className="img-box">
              <h4>Colorized</h4>
              {colorizedImage ? (
                <img src={colorizedImage} alt="Result" />
              ) : (
                <div className="placeholder">Result will appear here</div>
              )}
            </div>
          </div>
          
          {colorizedImage && (
            <a href={colorizedImage} download="colorized.png" className="download-btn">
              Download Result
            </a>
          )}
        </div>
      </div>
    </div>
  );
}