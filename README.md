# 🌿 Plant Disease Detection (Flask Deployment)

A deep learning web application built with **Flask** and **TensorFlow/Keras** that classifies plant leaf images into three categories:
- **Healthy**
- **Powdery**
- **Rust**

---

## 📌 Features
- Upload a leaf image via Flask web interface.
- Model predicts the disease class with confidence score.
- Simple and lightweight Flask backend (easy to deploy on **Heroku, Render, Railway, or AWS EC2**).

---

## 🗂 Project Structure
plant_disease_detection/
│
├── static/ # CSS, JS, images
├── templates/ # HTML templates (index.html, result.html)
├── model/ # Saved model (.keras / .h5)
├── app.py # Flask application
├── requirements.txt # Python dependencies
└── README.md # Documentation