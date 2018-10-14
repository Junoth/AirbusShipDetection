
# Part 1:Read data from csv
import numpy as np 					# linear algebra
import pandas as pd 					# data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
from skimage import io
import matplotlib.pyplot as plt
import os
import pandas as pd 
import keras
from keras import backend as K
from keras.layers import Activation
from keras import applications
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix
from keras.models import load_model, Sequential, Model 
import itertools
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings('ignore')


# decode function
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

train_path = './project_train'
train_image_dir = os.path.join(train_path, 'train')
masks = pd.read_csv('./train_ship_segmentations_v2.csv')
masks['path'] = masks['ImageId'].map(lambda x: os.path.join(train_image_dir, x))
masks.head()

imgs = os.listdir(train_path+'/train')



#mask the img of train
def mask_img(file):
	img = imread(train_path + '/' + file)
	img_masks = masks.loc[masks['ImageId'] == file, 'EncodedPixels'].tolist()
	all_masks = np.zeros((768,768))
	for mask in img_masks:
		if mask==mask:
			all_masks += rle_decode(mask)
	return all_masks

# split into training and validation groups
from sklearn.model_selection import train_test_split
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
#unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
masks.drop(['ships'], axis=1, inplace=True)
train_ids, valid_ids = train_test_split(unique_img_ids, 
                 test_size = 0.3, 
                 stratify = unique_img_ids['ships'])
train_df = pd.merge(masks, train_ids)
train_df = train_df.sample(min(15000, train_df.shape[0]))
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')


'''
img = imread(train_path + '/' + imgs[100])
fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(mask_img(imgs[100]))
axarr[2].imshow(img)
axarr[2].imshow(mask_img(imgs[100]), alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()
'''

#  use imggenerator to process the images
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 45, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')
valid_args = dict(
					fill_mode = 'reflect',
                   data_format = 'channels_last')

core_idg = ImageDataGenerator(**dg_args)
valid_idg = ImageDataGenerator(**valid_args)

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'has_ship_vec', 
                            target_size = (256,256),
                             color_mode = 'rgb',
                            batch_size = 30)

# used a fixed dataset for evaluating the algorithm
valid_x, valid_y = next(flow_from_dataframe(valid_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'has_ship_vec', 
                            target_size = (256,256),
                             color_mode = 'rgb',
                            batch_size = 1000)) # one big batch
print(valid_x.shape, valid_y.shape)


# use imageDataGenerator to process images
# use vgg19 to do transfer learning

vgg19_model = keras.applications.vgg19.VGG19(weights = "imagenet",include_top=False,input_shape = (256, 256, 3))
for layer in vgg19_model.layers:
    layer.trainable = False
vgg19_model.summary()


x = vgg19_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
prediction = Dense(1,activation='sigmoid')(x)
head_model = Model(input = vgg19_model.input, output = prediction)


head_model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['binary_accuracy'])

checkpoint = ModelCheckpoint("vgg19_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


head_model.fit_generator(train_gen, validation_data=(valid_x, valid_y), epochs=5,callbacks = [checkpoint, early])
head_model.save("AirbusShipDetective-model-VGG19.h5")


