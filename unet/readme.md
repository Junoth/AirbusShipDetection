Unet Ship Location Detection
=

the main part of program is from kaggle: https://www.kaggle.com/hmendonca/u-net-model-with-submission# <br>
<br>
to run the whole program, just run the seperate parts on kaggle<br>
![](https://github.com/Junoth/AirbusShipDetection/blob/master/unet/unetstructure.jpg)
Using Iou to be the evaluation function, which usually has a thresold value range from 0.3-0.5.<br>
This function compares the prediction area with the real area. When the value is larger than the set thresold value, then the area is reconginized as a valid result.

Model Parameters
-

![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/1.JPG)

Decode
-

transfer the label from .csv file to mask images<br>
Use rel-mask to decode the data from .csv<br>
The first column means the name of images, the second column express the pixel with ship in the images.<br>
Two number are for a group, the first one means the start pixel of a column, the second one means how many pixel continus represent ship.<br> 
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/2.JPG)

Split dataset
-

Set all the data to train and validation set.<br>
20% of all the images are selected to be validation dataset.<br>

Combination mask with original pictures
-

![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/6.JPG)

Preprocessing Images
-

This is argment step. We do flip, rotation, stretch or other operation to enlarge the dataset, which enhance the ability of program to recoginize different ships.<br> 

Build Model
-
Using unet model to train all the images.<br>
Set eariler stopping condition.<br>
Train the model, let the training processing break down when loss is low enough.<br>
If the loss is not satisfiable, then change the learning rate.<br>
loss=IoU, metrics=['binary_accuracy']<br>
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/10_1.JPG)
Print out loss and accuracy.<br>
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/11loss.JPG)

Predictions
-

prepare full resolution model.<br>
Visualize predictions.<br>
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/13_1.JPG)
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/13_3.JPG)
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/13_8.JPG)
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/13_2.JPG)
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/13_6.JPG)
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/13_7.JPG)
