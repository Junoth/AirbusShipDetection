Airbus Ship Detection Challenge
=

This is a [competition](https://www.kaggle.com/c/airbus-ship-detection) in kaggle.

![kaggle](https://storage.googleapis.com/kaggle-media/competitions/Airbus/ships.jpg)

The first step in this project is to find whether there is a ship in an image.Secondly,we need to mark the location of ships.
At last,we need to design an interface for users to upload images and dectect ships.

The [dataset](https://www.kaggle.com/c/airbus-ship-detection/data) is given in kaggle website.If you have interests,you can go to the website and download it.Labels of train set images are stored in a CSV file.There are **231723** masks in all train images,which means the ship number is also **231723**.

Data process
-
Firstly,to get the label from the CSV file,we use **pandas** library to process data.
```python
  import pandas as panda
  masks = pd.read_csv('./train_ship_segmentations_v2.csv')
```

If there is a mask or several marks in an image,then we label this image as **'1'**.If there is no ship then we label it 
as **'0'**.

```python
  masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
```

Secondly,we use **sklearn** library to split dataset as train set and valid set.We use the number of ships in a image to split all images to make sure images are evenly separated into different sets.The train set occupies **70%** and valid set occupies **30%**.
```python
  from sklearn.model_selection import train_test_split
```

Thirdly,the tool used in the image pre-process part is **Keras**.To build a better model,we use keras to do some process(crop,flip,rotation) in every image to make sure the space diversity of our dataset.
```python
  from keras.preprocessing.image import ImageDataGenerator
```

Transfer Learning
-

Unet
-

Faster_RCNN
-

Website
-

### Frame
The **[website](http://www.airbusshipdect.online/)** for users has been set up.You can upload on this website and it will 
return the probability of ships.

**Flask** is the main web frame for this website.Flask is a microframework for Python with two libraries:Werkzeug and Jinja 2.

![flask](http://flask.pocoo.org/static/logo/flask.pdf)

To setup flask,you need the command:
```
pip install Flask
```

Then,it can be easily used in the app:
```
from flask import Flask
```

### Web Server
The web server is **Nginx**,which is a light-weight HTTP and reverse proxy server and a generic TCP/UDP proxy server.
