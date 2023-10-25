// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.20.0/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/9.20.0/firebase-analytics.js";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyBczZwcE8VlmEc0eVGxpiCNeVg_Nks9duk",
    authDomain: "face-b9860.firebaseapp.com",
    projectId: "face-b9860",
    storageBucket: "face-b9860.appspot.com",
    messagingSenderId: "492662055035",
    appId: "1:492662055035:web:065587e60540b215d7c2de",
    measurementId: "G-XSYBFGBHH5"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, signInWithPopup, GoogleAuthProvider } from "https://www.gstatic.com/firebasejs/9.20.0/firebase-auth.js";
const auth = getAuth();

// Initialize Firebase
const provider = new GoogleAuthProvider();
document.getElementById("googleLogin").addEventListener("click", () => {
signInWithPopup(auth, provider)
    .then((result) => {
    // This gives you a Google Access Token. You can use it to access the Google API.
    const credential = GoogleAuthProvider.credentialFromResult(result);
    const token = credential.accessToken;
    // The signed-in user info.
    const user = result.user;
    console.log(result);
    window.location.href='/main';
    // ...
    })
    .catch((error) => {
    // Handle Errors here.
    const errorCode = error.code;
    const errorMessage = error.message;
    // The email of the user's account used.
    const email = error.customData.email;
    // The AuthCredential type that was used.
    const credential = GoogleAuthProvider.credentialFromError(error);
    console.log(error);
    // ...
    });
});