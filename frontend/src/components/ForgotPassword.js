import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Link } from 'react-router-dom';
import './Auth.css';

export default function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const { resetPassword } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      setMessage('');
      setError('');
      setLoading(true);
      await resetPassword(email);
      setMessage('Check your inbox for further instructions.');
    } catch (err) {
      setError('Failed to reset password. Please verify your email.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h2>Password Reset</h2>
          <p>Enter your email and we'll send you a link to reset your password.</p>
        </div>
        
        {error && (
          <div className="error-message">
             {error}
          </div>
        )}
        {message && (
          <div className="success-message" style={{ color: 'green', marginBottom: '15px', textAlign: 'center' }}>
             {message}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Email Address</label>
            <div className="input-wrapper">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="you@example.com"
              />
            </div>
          </div>
          <button type="submit" disabled={loading} className="auth-button">
            {loading ? 'Sending...' : 'Reset Password'}
          </button>
        </form>
        
        <div className="auth-link" style={{ marginTop: '15px' }}>
          <Link to="/signin">Back to Sign In</Link>
        </div>
      </div>
    </div>
  );
}