# Faster RCNN
## Target
#### mark the ship within the picture and tell the users the probability there is a ship within the marked region.

## Introduction of Faster RCNN
### what is Faster RCNN

#### Faster RCNN was come up on 2015, here's the paper:
#### S.Ren K.He,R.Girshick and J.Sun.Faster RCNN:Towards Real-Time Object Detection with Region Proposal Networks:https://arxiv.org/pdf/1506.01497.pdf
- for Faster RCNN network, at first, we can simply see this model as RPNs(Region Proposal Network)+Fast RCNN.To generate region proposals, we slide a small network over the convolutional feature map output by the last shared convolutional layer. This smallnetwork  takes as input an n × n spatial window of the input convolutional feature map. 
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/2.PNG)

#### the loss it uses:

![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/faster%20rcnn%20loss.PNG)
<div align=center><img width="400" height="100" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/smoothl1_loss.PNG"/></div>
<div align=center><img width="400" height="100" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/smoothl1_shape.PNG"/></div>

### Why we choose Faster RCNN
- we compared several models of object detection. YOLO, RCNN, FAST RCNN and so on. their generic performance are as picture below
![](https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/%E5%90%84%E4%BD%8D%E7%BD%AE%E8%AF%86%E5%88%AB%E7%AE%97%E6%B3%95%E6%AF%94%E8%BE%83.jpg)
- from this picture we can find that the RCNN might be the best model,because it's map is the best(which means they use the smallest mark region to mark the ship). but it might be slower than YOLO, but in our opinion, the accuracy is more important. So we use this model.


## Steps:
#### 1.using our own build model or vgg19 as a embed model of faster RCNN to train the data
#### 2.Set the configuration:
####     enviroment :

-        CPU：G1050ti
-        system:win10,
-        language: python3.5 
-        tools:coco, pycharm,tensorflow

####     the process of set the configuration are too much and difficult to describe, so just ignore it here...
#### 3.Interface of the input picture and the program.

- for the Faster RCNN model, the input have to be the xml file. which have the information of the marked region of training picture. But we don't have it. So we have to use the program(MATLAB and PYTHON) and marked the region mannually to form the xml files. for these two ways, the program is faster but have lower accuracy(because there are too many conbinations of ship in the picture, program always can hardly mark the region very accurate :( .),but the mannually label and mark the region tooks a longer time. 
- after solving the problem of picture, we build some folders and files for accessing the model,this is important if use the open source code of Faster RCNN(it's a large project!). which are upload to the github.
- after setting the enviroment and file, the folder architecture was as follows:
<div align=center><img width="400" height="500" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/file_arch.PNG"/></div>

### Training:
#### 500 iterations' loss:
<div align=center><img width="400" height="200" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/loss-500.PNG"/></div>

#### 1000 iterations' loss:
<div align=center><img width="400" height="200" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/loss-1000.PNG"/></div>

#### 1000 iterations prediction demo:
<div align=center><img width="400" height="400" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/picFaster1000ite.jpg"/></div>

#### 1000 iterations prediction demo2:
<div align=center><img width="400" height="400" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/picFaster1000ite2.jpg"/></div>

- from these result above, we can find that there are two problems:First, it's not accurate enough to detect ships in the picture,  it doesn't cover the ship or even cannot detect if it's a ship here. Secend, the probability shows on the circle is relatively low at most of time, the probability always lower than 50%, it's not convincing for users.
- So later I made two main changes to our program. First, I modified the size of the layer after VGGnet, which originally is a 512 vector, but for our detection, we only need two classes:background and ship, so I try to drop some information to make the imformation contains ship more explicitly. Second, I modified the threshhold which determine the background and the foreground, originally, it's 0.5. And now I modified it to 0.6 and 0.4. which means, only if the IOU is larger than 0.6 we will say that this anchor contains a ship and if it's less than 0.4 we will think it as a background, otherwise we will just drop it, it's useless from my perspective.
- And this is the result after modified(we also increase the iteration of the training)

#### 2000 iterations predection demo:
<div align=center><img width="400" height="400" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/picFaster2000iter3.jpg"/></div>

#### 2500 iterations predection demo:
<div align=center><img width="400" height="400" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/picFaster2500iter3.jpg"/></div>

- from the demo above, we can find that the result is more accurate than the result before, the box is can cover most of or even the entire ship, and the probability there is much higher and convincing enough for now. But it still cannot detect all the ships within thee picture, some of them are because of the size of ship, some are because of the color or something else. 
- there are several approaches to modify the module: 1. keep adding the number of the training data. the size of the training set is kind of small, so this might be the most significant reason for the accuracy. 2. I wanna try to change the size of the anchor to seperate the foreground and the background, our ship, at most of time, is not that large as the default setting of the Faster RCNN, so I will try to change the size of anchor to make it more suitable for our ship.


demo of smaller anchor(the size is 5/8 of the original size):
<div align=center><img width="400" height="400" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/smallanchor_2000iter1.jpg"/></div>
<div align=center><img width="400" height="400" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/smallanchor_2000iter2.jpg"/></div>
<div align=center><img width="400" height="400" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/smallanchor_2000iter4.jpg"/></div>
<div align=center><img width="400" height="400" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/smalleranchor_2000iter3.jpg"/></div>


- as you can see from the latest demo above, smaller ships can be detected, and the box which contains the ship is kind of accurate enough now, but however, the larger ship might be recognized as two or more different ships. This means I have set the anchors too small to catch one big ship. So I need to modify the size of it. What's more, the accuracy which shows the probability that there's a ship is lower than before, this is resulting from the change of size of anchors. So the following jobs are to keep changing and finding the best size of anchors and increasing the training dataset.

#### latest result
- in our latest result, we trained less than 3000 pictures, and have trained 15,000 iterations and train one picture each iteration, which can be seen as training 15000 pictures as well. And we also modified the threshhold to judge bounding box and the size of the anchor again. result are as follows:

- the accuracy(whether there is a ship within the picture)
- test on test data(more than 10,000)
<div align=center><img width="400" height="100" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/test accuracy.PNG"/></div>
-- test on training data(less than 3,000)
<div align=center><img width="400" height="100" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/training accuracy.PNG"/></div>

- the accuracy(how many ship in the picture)
- test on test data(more than 10,000)
<div align=center><img width="400" height="100" src="https://github.com/Junoth/AirbusShipDetection/blob/master/Faster%20RCNN/test accuracy_shipnumber.PNG"/></div>

## ... to be continued
