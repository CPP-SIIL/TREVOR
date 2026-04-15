// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCFDwGrjrTykSvD3UVUvKixj9HrG4Vn3nM",
  authDomain: "trevor-a0f08.firebaseapp.com",
  projectId: "trevor-a0f08",
  storageBucket: "trevor-a0f08.firebasestorage.app",
  messagingSenderId: "940496114286",
  appId: "1:940496114286:web:350a12301c07594a3c4c7f",
  measurementId: "G-BMPPS6776M"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
