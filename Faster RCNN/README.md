# Faster RCNN
## Target
#### mark the ship within the picture and tell the users the probability there is a ship within the marked region.

## Introduction of Faster RCNN
### what is Faster RCNN

#### Faster RCNN was come up on 2015, here's the paper:
#### S.Ren K.He,R.Girshick and J.Sun.Faster RCNN:Towards Real-Time Object Detection with Region Proposal Networks:https://arxiv.org/pdf/1506.01497.pdf
#### for Faster RCNN network, at first, we can simply see this model as RPNs(Region Proposal Network)+Fast RCNN.To generate region proposals, we slide a small network over the convolutional feature map output by the last shared convolutional layer. This smallnetwork  takes as input an n × n spatial window of the input convolutional feature map. 
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/2.PNG)

#### the loss it uses:
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/faster%20rcnn%20loss.PNG)
### Why we choose Faster RCNN
#### we compared several models of object detection. YOLO, RCNN, FAST RCNN and so on. their generic performance are as picture below
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/%E5%90%84%E4%BD%8D%E7%BD%AE%E8%AF%86%E5%88%AB%E7%AE%97%E6%B3%95%E6%AF%94%E8%BE%83.jpg)
#### from this picture we can find that the RCNN might be the best model,because it's map is the best(which means they use the smallest mark region to mark the ship). but it might be slower than YOLO, but in our opinion, the accuracy is more important. So we use this model.


## Steps:
#### 1.using our own build model or vgg19 as a embed model of faster RCNN to train the data
#### 2.Set the configuration:
####     enviroment :
####           CPU：G1050ti
####           system:win10,
####           language: python3.5 
####           tools:coco, pycharm,tensorflow
####     the process of set the configuration are too much and difficult to describe, so just ignore it here...
#### 3.Interface of the input picture and the program.
####     for the Faster RCNN model, the input have to be the xml file. which have the information of the marked region of training picture. But we don't have it. So we have to use the program(MATLAB and PYTHON) and marked the region mannually to form the xml files. for these two ways, the program is faster but have lower accuracy(because there are too many conbinations of ship in the picture, program always can hardly mark the region very accurate :( .),but the mannually label and mark the region tooks a longer time. 
####     after solving the problem of picture, we build some folders and files for accessing the model,this is important if use the open source code of Faster RCNN(it's a large project!). which are upload to the github.
#### after setting the enviroment and file, the folder architecture was as follows:
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/file_arch.PNG)



### Training:
#### 500 iterations' loss:
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/loss-500.PNG)
#### 1000 iterations' loss:
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/loss-1000.PNG)
#### 1000 iterations prediction demo
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/picFaster1000ite.jpg)
#### 1000 iterations prediction demo2
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/picFaster1000ite2.jpg)
### ... to be continued
