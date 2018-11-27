from flask import Flask, flash, render_template, request, make_response, redirect, url_for, send_file, send_from_directory
from flask_login import LoginManager, login_required, login_user, UserMixin, current_user
from wtforms import Form, BooleanField, StringField, PasswordField, validators
from wtforms.validators import DataRequired
from flask_pymongo import PyMongo
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from werkzeug import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from multiprocessing.pool import ThreadPool
import uuid
import flask
import numpy as np
import os
import cv2
import datetime

#Flask initial
app = Flask(__name__)

#Download location
directpath = "/app"
vgg16file = "AirbusShipDetective-model-VGG16.h5"
vgg19file = "AirbusShipDetective-model-VGG19.h5"

#Model location
basedir =  "/app/"
model_file1 = "/app/AirbusShipDetective-model-VGG16.h5"
model_file2 = "/app/AirbusShipDetective-model-VGG19.h5"

#Database initial
app.config['MONGO_DBNAME'] = 'shipdb'
app.config['MONGO_URI'] = 'mongodb://mongodb:27017/shipdb'
mongo = PyMongo(app)

#Threading initial
pool = ThreadPool(processes=3)

#Registration model
class RegistrationForm(Form):
    username = StringField('username', [validators.Length(min=4, max=25)])
    email = StringField('email', [validators.Length(min=6, max=35)])
    password = PasswordField('password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('confirm')

class LoginForm(Form):
    username = StringField('username', [validators.DataRequired()])
    password = PasswordField('password', [validators.DataRequired()])

class User(UserMixin):
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def is_authenticated(self):
        return True
    
    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return self.id

#Login part
app.secret_key = 's3cr36'
login_manager = LoginManager()
login_manager.session_protection = 'strong'
login_manager.login_view = 'login'
login_manager.login_message = u"Please log in to access this page."
login_manager.init_app(app)

@login_manager.user_loader
def load_user(id):
    u = mongo.db.posts.find_one({'Unique ID':id})
    if u is None:
        return None
    return User(u['Username'], id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm(request.form)
    if form.validate() == False:
        return render_template('login.html')
    if request.method == 'POST':
        username = request.form['username']
        passwd = request.form['password']
        if mongo.db.posts.find_one({'Username':username}) is None:
            flask.flash('Username not exist or password is wrong,please try again', 'info')
        elif check_password_hash(mongo.db.posts.find_one({'Username':username})['Password'] ,passwd) == False:
            flask.flash('Username not exist or password is wrong,please try again', 'info')
        else:   
            id = mongo.db.posts.find_one({'Username':username})['Unique ID']
            user = User(form.username, id) 
            login_user(user)
            next = flask.request.args.get('next')
            return flask.redirect(next or flask.url_for('index'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm(request.form)
    if request.method == 'POST' and form.validate():
        if mongo.db.posts.find_one({'Username':form.username.data}) is None:
            info = {"Username":form.username.data, 
            "Password":generate_password_hash(form.password.data),
            "Email":form.email.data,
            "Register Date":datetime.datetime.now(),
            "Unique ID":uuid.uuid4()}
            mongo.db.posts.insert_one(info)
            return redirect(url_for('login'))
        else:
           flask.flash("This username has been used")
    return render_template('register.html')
    
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/Introduction')
@login_required
def intro():
    return render_template('Introduction.html')

@app.route('/Team')
@login_required
def team():
    return render_template('Team.html')

@app.route('/Download')
@login_required
def download():
    return render_template('Download.html')

@app.route('/VGG16')
@login_required
def download_VGG16():
    return send_from_directory(directpath, vgg16file, as_attachment=True) 

@app.route('/VGG19')
@login_required
def download_VGG19():
    return send_from_directory(directpath, vgg19file, as_attachment=True)

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/home', methods=['post'])
@login_required
def up_photo():
    img = request.files.get('photo')
    upload_path = os.path.join(basedir, 'photo', secure_filename(img.filename))
    img.save(upload_path)
    newimg1 = image.load_img(upload_path,target_size=(224,224))
    newimg2 = image.load_img(upload_path,target_size=(256,256))
    results = []
    #result1 = pool.apply_async(modeltest,(model_file1,newimg1))
    #result2 = pool.apply_async(modeltest,(model_file2,newimg2))
    results.append(modeltest(model_file1,newimg1))
    results.append(modeltest(model_file2,newimg2))
    outputs = [{ 'model': u'VGG16', 'output':results[0]}, 
                {'model': u'VGG19', 'output':results[1]}]
    mongo.db.posts.update({"Username": current_user.name}, {"$set": {"VGG16_result":float(results[0][0][0]),"VGG19_result": float(results[1][0][0])}}, upsert=True)
    return render_template('output.html', outputs = outputs)

def modeltest(file,newimg):
    model = load_model(file)
    x = image.img_to_array(newimg)
    x = np.expand_dims(x, axis=0)
    result = model.predict(x)
    K.clear_session()
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug = True)


