Unet Ship Location Detection
===
the main part of program is from kaggle: https://www.kaggle.com/hmendonca/u-net-model-with-submission# \<br>
\<br>
*Model Parameters
-----
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/1.JPG)
\<br>
*Decode
-----
transfer the label from .csv file to mask images
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/2.JPG)
\<br>
*Split dataset
-----
Set all the data to train and validation set.\<br>
Besides, count the number of images that have ships.\<br>
*Combination mask with original pictures
-----
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/6.JPG)
\<br>
*Preprocessing Images
-----
This is argment step. We do flip, rotation, stretch or other operation to enlarge the dataset, which enhance the ability of program to recoginize different ships.\<br> 
*Build Model
-----
Build several layers of convolution layers, pooling layers, fully connection layers......\<br>
Get the weights of the network.\<br>
Train the model, let the training processing break down when loss is low enough.\<br>
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/10_1.JPG)
Print out loss and accuracy.\<br>
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/11loss.JPG)
*Predictions
-----
Prepare full resolution model.\<br>
Visualize predictions.\<br>
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/13_1.JPG)
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/13_3.JPG)
![](https://github.com/SandyHao/AirbusShipDetection/blob/patch-2/unet/13_8.JPG)
