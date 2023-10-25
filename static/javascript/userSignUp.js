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

document.getElementById('signUpButton').addEventListener('click', (event) => {
    event.preventDefault()
    const signUpEmail = document.getElementById('signUpEmail').value
    const signUpPassword = document.getElementById('signUpPassword').value

    
    createUserWithEmailAndPassword(auth, signUpEmail, signUpPassword)
    .then((userCredential) => {
        console.log(userCredential)
        // Signed in
        const user = userCredential.user;
        // ...
    })
    // 제대로 로그인이 안되면 밑에 코드가 실행됨
    .catch((error) => {
        console.log('error')
        
        const errorCode = error.code;
        const errorMessage = error.message;
        // ..
    });
})