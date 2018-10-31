from flask import Flask, render_template,request,make_response
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import os
import cv2

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:9000/userinfo"
mongo = PyMongo(app)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/intro')
def intro():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('layout.html')

@app.route('/home', methods=['post'])
def up_photo():
    img = request.files.get('photo') 
    username = request.form.get('name')
    upload_path = os.path.join(basedir, 'photo', secure_filename(img.filename))
    img.save(upload_path)
    img_data = open(upload_path, "rb").read()
    model_file = "/junoth/ec601/AirbusShipDetective-model-VGG16.h5"
    model = load_model(model_file)
    newimg = image.load_img(upload_path,target_size=(224,224))
    x = image.img_to_array(newimg)
    x = np.expand_dims(x, axis=0)
    result = model.predict(x)
    return str(result[0][0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
