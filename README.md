Airbus Ship Detection Challenge
=

This is a [competition](https://www.kaggle.com/c/airbus-ship-detection) in kaggle.

                ![kaggle](https://storage.googleapis.com/kaggle-media/competitions/Airbus/ships.jpg)

The first step in this project is to find whether there is a ship in an image.Secondly,we need to mark the location of ships.
At last,we need to design an interface for users to upload images and dectect ships.

The [dataset](https://www.kaggle.com/c/airbus-ship-detection/data) is given in kaggle website.If you have interests,you can go to the website and download it.Labels of train set images are stored in a CSV file.There are **231723** masks in all train images,which means the ship number is also **231723**.

(Milestone update: 1.Use VGG19 to predict posibility of ship exsiting; 2. Use unet to predict ship location; 3. Build website interface)<br>
(Short-term Goals: 1.Use fast RCNN to detect ships; 2. Change and improve prarmeters in unet; 3. Improve website)

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

VGG19
-
Based on keras VGG19.<br>
* decode data from kaggle -> get mask of ship location on the pictures -> combine labels and images together to be the dataset.<br>
* do augment to get more complex dataset -> set part of the parameters used for training<br>
* remove the last prediction layer from original VGG19 network -> instead, use a simple layer to predict only posibility

Unet
-
The main part of program is from kaggle: https://www.kaggle.com/hmendonca/u-net-model-with-submission# <br>
The whole program can be runned seperately on kaggle.<br>
* decode & make mask
* seperate dataset & do augment to images
* build & train model
* output prediction

Faster_RCNN
-
Faster RCNN is one of the most efficient and accurate model of object detection. So we try to use Faster RCNN to detect where the ship is in the picture the users given.
here's a little demo of the result:
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/picFaster2000iter2.jpg)
the code we use to accomplish Faster RCNN can be found here:
https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3.5
