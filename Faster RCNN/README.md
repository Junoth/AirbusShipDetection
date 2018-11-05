# Faster RCNN
## Target
#### mark the ship within the picture and tell the users the probability there is a ship within the marked region.

## Introduction of Faster RCNN
#### we compared several models of object detection. YOLO, RCNN, FAST RCNN and so on. their generic performance are as picture below
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/%E5%90%84%E4%BD%8D%E7%BD%AE%E8%AF%86%E5%88%AB%E7%AE%97%E6%B3%95%E6%AF%94%E8%BE%83.jpg)
#### from this picture we can find that the RCNN might be the best model,because it's map is the best(which means they use the smallest mark region to mark the ship). but it might be slower than YOLO, but in our opinion, the accuracy is more important. So we use this model.


## Steps:
#### 1.using our own build model or vgg19 as a embed model of faster RCNN to train the data
#### 2.Set the configuration:
####     enviroment :GPU：G1050ti，win10,python3.6 and python3.5, caffe, pycharm.
####     the process of set the configuration are too much and difficult to describe, so just ignore it here...
#### 3.Interface of the input picture and the program.
####     for the Faster RCNN model, the input have to be the xml file. which have the information of the marked region of training picture. But we don't have it. So we have to use the program(MATLAB and PYTHON) and marked the region mannually to form the xml files. for these two ways, the program is faster but have lower accuracy(because there are too many conbinations of ship in the picture, program always can hardly mark the region very accurate :( .),but the mannually label and mark the region tooks a longer time. 
####     after solve the problem of picture, we build some folders and files for accessing the model,this is important if use the open source code of Faster RCNN(it's a large project!). which are upload to the github.

### to be continued
