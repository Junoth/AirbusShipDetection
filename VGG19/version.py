#load VGG19model and visiualization the output in each layer during forward propagation 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io 
import scipy.misc 
 
#convolution
def _conv_layer(input, weights, bias): 
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME') 
    return tf.nn.bias_add(conv, bias) 

#pooling
def _pool_layer(input): 
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME') 
    
#subtract pixel mean 
def preprocess(image, mean_pixel): 
    return image - mean_pixel 

#add pixel mean 
def unprocess(image, mean_pixel): 
    return image + mean_pixel 
    
#read
def imread(path): 
    return scipy.misc.imread(path).astype(np.float) 
    
#save 
def imsave(path, img): 
    img = np.clip(img, 0, 255).astype(np.uint8) 
    scipy.misc.imsave(path, img) 
print ("Functions for VGG ready") 
#define VGG's sturctual，store weight and offset 
def net(data_path, input_image):
    #get parameter for each layer
    layers = ( 
         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 
         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 
         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 
         'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 
         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 
         'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 
         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 
         'relu5_3', 'conv5_4', 'relu5_4' 
    ) 
    data = scipy.io.loadmat(data_path) 
    #sub mean 
    mean = data['normalization'][0][0][0] 
    mean_pixel = np.mean(mean, axis=(0, 1)) 
    #print(mean_pixel) 
    #get W&b  
    weights = data['layers'][0] 
    #print(weights) 
    net = {} 
    current = input_image 
    for i, name in enumerate(layers): 
        #:judge by the first 3 letter
        kind = name[:4] 
        if kind == 'conv': 
            kernels, bias = weights[i][0][0][0][0] 
            # matconvnet: weights are [width, height, in_channels, out_channels]\n", 
            # tensorflow: weights are [height, width, in_channels, out_channels]\n", 
            #transportation
            kernels = np.transpose(kernels, (1, 0, 2, 3)) 
            #trans bias to one-dimentional
            bias = bias.reshape(-1) 
            current = _conv_layer(current, kernels, bias) 
        elif kind == 'relu': 
            current = tf.nn.relu(current) 
        elif kind == 'pool': 
            current = _pool_layer(current) 
        net[name] = current 
    assert len(net) == len(layers) 
    return net, mean_pixel, layers 
print ("Network for VGG ready") 
#cwd  = os.getcwd() 
VGG_PATH = "E:/VScode/transstyle/imagenet-vgg-verydeep-19.mat" 
IMG_PATH = "E:/VScode/AirbusShipDetectionVGG19/project_train/train/00f34434e.jpg" 
input_image = imread(IMG_PATH) 
#shape 
shape = (1,input_image.shape[0],input_image.shape[1],input_image.shape[2]) 
#begin 
with tf.Session() as sess: 
    image = tf.placeholder('float', shape=shape) 
    #net function 
    nets, mean_pixel, all_layers = net(VGG_PATH, image) 
    #sub mean 
    input_image_pre = np.array([preprocess(input_image, mean_pixel)]) 
    layers = all_layers # For all layers \n", 
    # layers = ('relu2_1', 'relu3_1', 'relu4_1')\n", 
    for i, layer in enumerate(layers): 
        print ("[%d/%d] %s" % (i+1, len(layers), layer)) 
        features = nets[layer].eval(feed_dict={image: input_image_pre}) 
        print (" Type of 'features' is ", type(features)) 
        print (" Shape of 'features' is %s" % (features.shape,)) 
        # Plot response \n", 
        # print out each layer 
        if 1: 
            plt.figure(i+1, figsize=(10, 5)) 
            plt.matshow(features[0, :, :, 0], cmap=plt.cm.gray, fignum=i+1) 
            plt.title("" + layer) 
            plt.colorbar() 
            plt.show()
