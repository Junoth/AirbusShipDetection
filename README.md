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
  masks['path'] = masks['ImageId'].map(lambda x: os.path.join(train_image_dir, x))
```

If there is a mask or several marks in an image,then we label this image as **'1'**.If there is no ship then we label it 
as **'0'**.

```python
  masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
  unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
  unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
  unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
  masks.drop(['ships'], axis=1, inplace=True)
```

Secondly,we use **sklearn** library to split dataset as train set and valid set.We use the number of ships in a image to split all images to make sure images are evenly separated into different sets.The train set occupies **70%** and valid set occupies **30%**.
```python
  from sklearn.model_selection import train_test_split
```

Transfer Learning
-

Unet
-

Faster_RCNN
-

Website
-








