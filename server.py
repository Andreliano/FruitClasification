import base64
import io

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model = load_model("fruit-classification-model.h5")

fruits = ['Fresh Apple', 'Fresh Banana', 'Fresh Grape', 'Fresh Guava',
          'Fresh Jujube', 'Fresh Orange',
          'Fresh Pomegranate', 'Fresh Strawberry', 'Rotten Apple',
          'Rotten Banana', 'Rotten Grape', 'Rotten Guava',
          'Rotten Jujube', 'Rotten Orange', 'Rotten Pomegranate',
          'Rotten Strawberry']


def make_prediction(img):
    img_array = np.array([img])
    pred = model.predict(img_array)
    print("PRED: ", pred)
    fruit_index = np.argmax(pred[0])
    print("FRUIT-INDEX: ", fruit_index)
    classified_as = fruits[fruit_index]
    return classified_as, max(pred[0])


@app.route('/', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return {'status': 'error', 'message': 'No file part'}

    file = request.files['file']

    if file.filename == '':
        return {'status': 'error', 'message': 'No selected file'}

    if file:
        img = Image.open(io.BytesIO(file.read()))
        img_resized = img.resize((224, 224))
        print("IMG-RESIZED: ", img_resized)

        classified_as, probability = make_prediction(img_resized.getdata())
        print("CLASSIFIED-AS: ", classified_as)
        return {'status': 'success', 'classified_as': classified_as, 'probability': str(round(probability, 2))}


if __name__ == '__main__':
    app.run(debug=True, port=5000)
