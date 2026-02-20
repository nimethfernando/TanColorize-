import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import '../App.css';

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function ColorizeApp() {
  const { currentUser, logout } = useAuth();
  
  // State
  const [modelPath, setModelPath] = useState("");
  const [inputSize, setInputSize] = useState(512);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [mode, setMode] = useState("single"); // 'single' or 'folder'
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");

  // Single Image State
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalPreview, setOriginalPreview] = useState(null);
  const [colorizedImage, setColorizedImage] = useState(null);

  // Folder State
  const [inputFolder, setInputFolder] = useState("");
  const [outputFolder, setOutputFolder] = useState("");

  // Helper: Open File Dialog on Server
  const handleBrowse = async (type, setter) => {
    try {
      const res = await axios.get(`${API_URL}/browse?type=${type}`);
      if (res.data.path) setter(res.data.path);
    } catch (err) {
      console.error(err);
      alert("Error opening file dialog");
    }
  };

  // Helper: Load Model
  const loadModel = async () => {
    if (!modelPath) return alert("Please select a model file first");
    setLoading(true);
    setStatusMsg("Loading model...");
    try {
      const res = await axios.post(`${API_URL}/load-model`, {
        path: modelPath,
        input_size: inputSize
      });
      setModelLoaded(true);
      setStatusMsg(res.data.message);
    } catch (err) {
      setStatusMsg("Error loading model: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Helper: Process Single Image
  const processSingleImage = async () => {
    if (!selectedFile) return;
    if (!modelLoaded) return alert("Load model first!");
    
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    if (currentUser) {
          // Assuming Firebase auth is used, currentUser usually has a 'uid'
          // If it doesn't, you can use currentUser.email as the identifier
          const userId = currentUser.uid || currentUser.email; 
          formData.append("user_id", userId);
        }
        
    try {
      const res = await axios.post(`${API_URL}/colorize-image`, formData);
      // Convert hex string back to image for display
      const hex = res.data.image_data;
      const bytes = new Uint8Array(hex.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
      const blob = new Blob([bytes], { type: "image/png" });
      setColorizedImage(URL.createObjectURL(blob));
    } catch (err) {
      alert("Error processing image");
    } finally {
      setLoading(false);
    }
  };

  // Helper: Process Folder
  const processFolder = async () => {
    if (!inputFolder || !outputFolder) return alert("Select both folders");
    if (!modelLoaded) return alert("Load model first!");

    setLoading(true);
    setStatusMsg("Processing folder... check console/server logs for details.");
    try {
      const res = await axios.post(`${API_URL}/process-folder`, {
        input_folder: inputFolder,
        output_folder: outputFolder
      });
      setStatusMsg(`Success! Processed ${res.data.processed_count} images to ${res.data.output_folder}`);
    } catch (err) {
      setStatusMsg("Error processing folder");
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
        
        <div className="control-group">
          <label>Model Path</label>
          <div className="file-input-group">
            <input type="text" value={modelPath} readOnly placeholder="Select .pth file" />
            <button onClick={() => handleBrowse('file', setModelPath)}>Browse</button>
          </div>
        </div>

        <div className="control-group">
          <label>Input Size: {inputSize}</label>
          <input 
            type="range" min="256" max="1024" step="64" 
            value={inputSize} 
            onChange={(e) => setInputSize(parseInt(e.target.value))} 
          />
        </div>

        <button 
          className="primary-btn" 
          onClick={loadModel} 
          disabled={loading || !modelPath}
        >
          {loading ? "Loading..." : "Load Model"}
        </button>
        
        <p className="status-text">{statusMsg}</p>

        <hr />
        <h3>Mode</h3>
        <div className="radio-group">
          <label>
            <input 
              type="radio" checked={mode === 'single'} 
              onChange={() => setMode('single')} 
            /> Single Image
          </label>
          <label>
            <input 
              type="radio" checked={mode === 'folder'} 
              onChange={() => setMode('folder')} 
            /> Folder Process
          </label>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <h1>🎨 Tancolorize</h1>
        
        {mode === 'single' ? (
          <div className="single-mode">
            <div className="upload-section">
              <input type="file" onChange={onFileChange} accept="image/*" />
              <button 
                className="action-btn"
                onClick={processSingleImage} 
                disabled={!selectedFile || !modelLoaded || loading}
              >
                {loading ? "Processing..." : "Colorize Image"}
              </button>
            </div>

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
        ) : (
          <div className="folder-mode">
            <div className="control-group">
              <label>Input Folder</label>
              <div className="file-input-group">
                <input type="text" value={inputFolder} readOnly />
                <button onClick={() => handleBrowse('folder', setInputFolder)}>Select Input</button>
              </div>
            </div>
            
            <div className="control-group">
              <label>Output Folder</label>
              <div className="file-input-group">
                <input type="text" value={outputFolder} readOnly />
                <button onClick={() => handleBrowse('folder', setOutputFolder)}>Select Output</button>
              </div>
            </div>

            <button 
              className="action-btn"
              onClick={processFolder}
              disabled={!inputFolder || !outputFolder || !modelLoaded || loading}
            >
              {loading ? "Processing..." : "Process All Images"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
