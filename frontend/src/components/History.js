import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import '../App.css';

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function History() {
  const { currentUser } = useAuth();
  const [historyItems, setHistoryItems] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      if (!currentUser) {
        setLoading(false);
        return;
      }
      try {
        const userId = currentUser.uid || currentUser.email;
        const res = await axios.get(`${API_URL}/history/${userId}`);
        
        // Ensure we only set valid paired items
        const validItems = (res.data.history || []).filter(
          item => item.original && item.colorized
        );
        
        setHistoryItems(validItems);
      } catch (err) {
        console.error("Error fetching history:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [currentUser]);

  // Helper function to force download instead of opening in a new tab
  const handleDownload = async (url, filename) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error('Download failed. Ensure your S3 bucket has CORS configured.', error);
      // Fallback: open in new tab
      window.open(url, '_blank');
    }
  };

  if (!currentUser) {
    return (
      <div className="app-container">
        <div className="main-content">
          <h2>Please sign in to view your history.</h2>
          <Link to="/" className="primary-btn">Back to Home</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="main-content" style={{ width: '100%', maxWidth: '1000px', margin: '0 auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h1>🖼️ Your Colorization History</h1>
          <Link to="/" className="action-btn" style={{ textDecoration: 'none' }}>Back to Colorizer</Link>
        </div>
        
        {loading ? (
          <p>Loading your history...</p>
        ) : historyItems.length === 0 ? (
          <p>No colorized images found in your history.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '40px', marginTop: '30px' }}>
            {historyItems.map((item, index) => (
              <div key={index} style={{ 
                display: 'flex', gap: '20px', backgroundColor: '#2d2d2d', padding: '20px', borderRadius: '10px' 
              }}>
                {/* Original Image Container */}
                <div style={{ flex: 1 }}>
                  <h4 style={{ marginBottom: '10px' }}>Grayscale (Original)</h4>
                  <div className="img-box" style={{ padding: '0', border: 'none' }}>
                    <img src={item.original} alt={`Original ${index}`} style={{ width: '100%', borderRadius: '8px' }} />
                  </div>
                  <button 
                    onClick={() => handleDownload(item.original, `original_${item.timestamp}.png`)} 
                    className="action-btn" 
                    style={{ marginTop: '15px', width: '100%' }}>
                    Download Original
                  </button>
                </div>

                {/* Colorized Image Container */}
                <div style={{ flex: 1 }}>
                  <h4 style={{ marginBottom: '10px' }}>Colorized Result</h4>
                  <div className="img-box" style={{ padding: '0', border: 'none' }}>
                    <img src={item.colorized} alt={`Colorized ${index}`} style={{ width: '100%', borderRadius: '8px' }} />
                  </div>
                  <button 
                    onClick={() => handleDownload(item.colorized, `colorized_${item.timestamp}.png`)} 
                    className="action-btn" 
                    style={{ marginTop: '15px', width: '100%', backgroundColor: '#4CAF50' }}>
                    Download Colorized
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}