import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import '../App.css';

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8080";

export default function ColorizeApp() {
  const { currentUser, logout } = useAuth();

  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalPreview, setOriginalPreview] = useState(null);
  const [colorizedImage, setColorizedImage] = useState(null);
  const [downloadFormat, setDownloadFormat] = useState("png");
  const [statusMsg, setStatusMsg] = useState("");

  // Helper: Process Single Image
  const processSingleImage = async () => {
    if (!selectedFile) return alert("Please select an image first");

    setLoading(true);
    setStatusMsg("Colorizing...");
    setColorizedImage(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    if (currentUser) {
      const userId = currentUser.uid || currentUser.email;
      formData.append("user_id", userId);
    }

    try {
      const res = await axios.post(`${API_URL}/colorize-image`, formData);
      const hex = res.data.image_data;
      const bytes = new Uint8Array(hex.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
      const blob = new Blob([bytes], { type: "image/png" });
      setColorizedImage(URL.createObjectURL(blob));
      setStatusMsg("Done!");
    } catch (err) {
      setStatusMsg("Error processing image: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Helper: Handle file input change
  const onFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setOriginalPreview(URL.createObjectURL(file));
    setColorizedImage(null);
    setStatusMsg("");
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Failed to logout:', error);
    }
  };

  // Helper: Handle image download
  const handleDownload = () => {
    if (!colorizedImage) return;

    if (downloadFormat === 'png') {
      const link = document.createElement('a');
      link.href = colorizedImage;
      link.download = 'colorized.png';
      link.click();
    } else if (downloadFormat === 'jpeg') {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = 'colorized.jpg';
        link.click();
      };
      img.src = colorizedImage;
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
            <>
              <Link
                to="/history"
                className="action-btn"
                style={{ display: 'block', textAlign: 'center', marginBottom: '10px', textDecoration: 'none' }}
              >
                View History
              </Link>
              <button onClick={handleLogout} className="logout-btn">Logout</button>
            </>
          ) : (
            <div className="auth-actions">
              <Link to="/signin" className="logout-btn">Sign In</Link>
              <Link to="/signup" className="logout-btn">Sign Up</Link>
            </div>
          )}
        </div>

        <h2>TanColorize</h2>
        <p style={{ fontSize: '13px', color: '#aaa', marginTop: '8px' }}>
          AI-powered image colorization with accurate skin tone reproduction.
          Upload a grayscale image and click Colorize.
        </p>

        <p className="status-text">{statusMsg}</p>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <h1>🎨 TanColorize</h1>

        <div className="single-mode">
          <div className="upload-section">
            <input type="file" onChange={onFileChange} accept="image/*" />
            <button
              className="action-btn"
              onClick={processSingleImage}
              disabled={!selectedFile || loading}
            >
              {loading ? "Colorizing..." : "Colorize Image"}
            </button>
          </div>

          <div className="image-preview-container">
            <div className="img-box">
              <h4>Original</h4>
              {originalPreview
                ? <img src={originalPreview} alt="Original" />
                : <div className="placeholder">Upload an image</div>
              }
            </div>
            <div className="img-box">
              <h4>Colorized</h4>
              {colorizedImage
                ? <img src={colorizedImage} alt="Result" />
                : <div className="placeholder">Result will appear here</div>
              }
            </div>
          </div>

          {colorizedImage && (
            <div className="download-container" style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: '20px' }}>
              <select
                value={downloadFormat}
                onChange={(e) => setDownloadFormat(e.target.value)}
                style={{ padding: '10px', borderRadius: '5px' }}
              >
                <option value="png">PNG</option>
                <option value="jpeg">JPG</option>
              </select>
              <button onClick={handleDownload} className="download-btn">
                Download Result
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
