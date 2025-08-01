from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import urllib.request
import cv2
import pickle

app = Flask(__name__)
CORS(app)

# Chargement du modèle et des encodeurs
model = load_model("model_doc_classifier.keras")

with open("enc_entreprise.pkl", "rb") as f:
    le_entreprise = pickle.load(f)

with open("enc_label.pkl", "rb") as f:
    le_label = pickle.load(f)

IMG_SIZE = 224

def load_image_from_url(url):
    resp = urllib.request.urlopen(url)
    image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_url = data.get("image_url")
    entreprise_name = data.get("entreprise")

    if entreprise_name not in le_entreprise.classes_:
        return jsonify({"error": "Entreprise inconnue"}), 400

    try:
        image = load_image_from_url(image_url)
    except:
        return jsonify({"error": "Erreur de chargement d’image"}), 400

    entreprise_encoded = le_entreprise.transform([entreprise_name])
    entreprise_encoded = np.array(entreprise_encoded)

    prediction = model.predict([image, entreprise_encoded])
    predicted_label = le_label.inverse_transform([np.argmax(prediction)])[0]

    return jsonify({"document_type": predicted_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)