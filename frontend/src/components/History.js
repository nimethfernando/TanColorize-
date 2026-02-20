import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import '../App.css';

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function History() {
  const { currentUser } = useAuth();
  const [images, setImages] = useState([]);
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
        setImages(res.data.images || []);
      } catch (err) {
        console.error("Error fetching history:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [currentUser]);

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
        ) : images.length === 0 ? (
          <p>No colorized images found in your history.</p>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '20px', marginTop: '20px' }}>
            {images.map((url, index) => (
              <div key={index} className="img-box" style={{ padding: '10px' }}>
                <a href={url} target="_blank" rel="noopener noreferrer">
                  <img 
                    src={url} 
                    alt={`Colorized ${index}`} 
                    style={{ width: '100%', borderRadius: '8px', cursor: 'pointer' }} 
                  />
                </a>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}