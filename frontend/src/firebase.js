import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

// Your web app's Firebase configuration
// Replace these values with your Firebase project config
const firebaseConfig = {
  apiKey: "AIzaSyBcAvucZxa9VnTHCTCkjiYS5Ry6yC2dOtk",
  authDomain: "tancolorize.firebaseapp.com",
  projectId: "tancolorize",
  storageBucket: "tancolorize.firebasestorage.app",
  messagingSenderId: "506662312452",
  appId: "1:506662312452:web:18a73ba25f9f24b5547351"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app);
export default app;
