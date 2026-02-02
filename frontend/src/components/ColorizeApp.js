import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import '../App.css';

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function ColorizeApp() {
  const { currentUser, logout } = useAuth();
  
  // State
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Ready");

  // Single Image State
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalPreview, setOriginalPreview] = useState(null);
  const [colorizedImage, setColorizedImage] = useState(null);

  // Automatically trigger colorization when a file is selected
  const onFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // 1. Update UI for preview and loading state
    setSelectedFile(file);
    setOriginalPreview(URL.createObjectURL(file));
    setColorizedImage(null);
    setLoading(true);
    setStatusMsg("Colorizing image... please wait.");

    // 2. Automatically send to backend
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_URL}/colorize-image`, formData);
      
      // 3. Process the hex result from server.py
      const hex = res.data.image_data;
      const bytes = new Uint8Array(hex.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
      const blob = new Blob([bytes], { type: "image/png" });
      
      setColorizedImage(URL.createObjectURL(blob));
      setStatusMsg("Success! Your image is ready.");
    } catch (err) {
      console.error(err);
      setStatusMsg("Error: Could not colorize image. Ensure the server is running.");
    } finally {
      setLoading(false);
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
      {/* Sidebar - Simplified for Automated Experience */}
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
        
        <h2>Automatic Colorizer</h2>
        <p className="status-text">Status: <strong>{statusMsg}</strong></p>
        
        <div className="info-box">
          <p>Your model is hosted on Google Cloud Storage and is ready for inference.</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <h1>🎨 Tancolorize</h1>
        
        <div className="single-mode">
          <div className="upload-section">
            <label className="custom-file-upload">
              <input type="file" onChange={onFileChange} accept="image/*" disabled={loading} />
              {loading ? "Processing..." : "Upload & Colorize"}
            </label>
          </div>

          <div className="image-preview-container">
            <div className="img-box">
              <h4>Original</h4>
              {originalPreview ? (
                <img src={originalPreview} alt="Original" />
              ) : (
                <div className="placeholder">Upload an image to start</div>
              )}
            </div>
            <div className="img-box">
              <h4>Colorized</h4>
              {colorizedImage ? (
                <img src={colorizedImage} alt="Result" />
              ) : (
                <div className="placeholder">
                  {loading ? "Model is working..." : "Result will appear here"}
                </div>
              )}
            </div>
          </div>
          
          {colorizedImage && (
            <div className="actions-footer">
               <a href={colorizedImage} download="colorized.png" className="download-btn">
                Download Result
              </a>
              <button onClick={() => {
                setSelectedFile(null);
                setOriginalPreview(null);
                setColorizedImage(null);
                setStatusMsg("Ready");
              }} className="secondary-btn">Clear</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}