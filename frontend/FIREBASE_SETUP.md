# Firebase Authentication Setup Guide

## Step 1: Create a Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project" or select an existing project
3. Follow the setup wizard to create your project

## Step 2: Enable Email/Password Authentication

1. In your Firebase project, go to **Authentication** in the left sidebar
2. Click **Get Started**
3. Go to the **Sign-in method** tab
4. Click on **Email/Password**
5. Enable **Email/Password** authentication
6. Click **Save**

## Step 3: Get Your Firebase Configuration

1. In Firebase Console, click the gear icon ⚙️ next to "Project Overview"
2. Select **Project settings**
3. Scroll down to **Your apps** section
4. Click the **Web** icon (`</>`) to add a web app
5. Register your app (you can use any app nickname)
6. Copy the Firebase configuration object

## Step 4: Update Firebase Configuration

1. Open `frontend/src/firebase.js`
2. Replace the placeholder values with your actual Firebase config:

```javascript
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_AUTH_DOMAIN",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_STORAGE_BUCKET",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID"
};
```

## Step 5: Test the Application

1. Start your React app: `npm start`
2. Navigate to `http://localhost:3000`
3. You should be redirected to the sign-in page
4. Click "Sign Up" to create a new account
5. After signing up, you'll be redirected to the main app

## Features

- ✅ Email/Password authentication
- ✅ Protected routes (main app requires authentication)
- ✅ Sign In page
- ✅ Sign Up page with password confirmation
- ✅ Logout functionality
- ✅ User email display in sidebar

## Security Notes

- Never commit your Firebase config with real credentials to public repositories
- Consider using environment variables for production
- Firebase handles password hashing and security automatically
