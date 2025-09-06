import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join("model/plant_disease_model.keras") 
model = load_model(MODEL_PATH)

# Define class names (adjust to your dataset)
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

def getResult(image_path):
    img = load_img(image_path, target_size=(224, 224))  # match model input size
    x = img_to_array(img)
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400

    f = request.files["file"]
    if f.filename == "":
        return "No file selected", 400

    basepath = os.path.dirname(__file__)
    upload_folder = os.path.join(basepath, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, secure_filename(f.filename))
    f.save(file_path)

    predictions = getResult(file_path)
    predicted_label = CLASS_NAMES[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)

    return f"Prediction: {predicted_label} ({confidence}%)"

if __name__ == "__main__":
    app.run(debug=True)
