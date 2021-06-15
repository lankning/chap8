#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, random, cv2, shutil, random
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from math import pi


# In[ ]:


def laplace_4(img):
    fil = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
    img = cv2.filter2D(img,-1,fil)
    img[img>1]=1
    img[img<0]=0
    return img


# In[ ]:


def rotate_and_scale(image, center=None): #1
    ceta = random.randint(0,90)
    # 缩放比例scale由角度ceta决定，注意要化为弧度制
    if ceta >= 0:
        scale = random.randint(int(10*np.sin(ceta/180*pi)+10*np.cos(ceta/180*pi)),20)/10
    else:
        scale = random.randint(int(-10*np.sin(ceta/180*pi)+10*np.cos(ceta/180*pi)),20)/10
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4
    M = cv2.getRotationMatrix2D(center, ceta, scale) #5
    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated


# In[ ]:


def load_imgs(x_paths, cateList):
    xList=[]
    yList=[]
    for x_path in x_paths:
        img = Image.open(x_path)
        img_arr = np.array(img)/255
        img_arr = cv2.resize(img_arr, (224, 224), interpolation=cv2.INTER_NEAREST)
        # img_arr = laplace_4(img_arr)
        # img_arr = rotate_and_scale(img_arr)
        xList.append(img_arr)
        cate = x_path.split("\\")[-2]
        label = cateList.index(cate)
        yList.append(label)
    return np.float32(np.array(xList)),np.array(yList)


# In[ ]:


def is_here(list1,list2):
    if len(list1)==len(list2):
        for i in list1:
            if i in list2:
                continue;
            else:
                return False
        return True
    else:
        return False


# In[ ]:


def generator(inpath, cates, batch_size, reshuffle_each_iteration=False):
    # read and sort train data from dir
    cateList = os.listdir(inpath)
    if (1-is_here(cates,cateList)):
        raise Exception("Invalid cates! The input categories are [",",".join(str(x) for x in cates),
                        "], while the categories found in the folder are ["",".join(str(x) for x in cateList),"]")
    else:
        catePath = [os.path.join(inpath,c) for c in cateList]
        x_List = []
        for c in catePath:
            imgList = os.listdir(c)
            x_List += [os.path.join(c,i) for i in imgList]
        i = 0
        while True:
            if reshuffle_each_iteration:
                if i == 0:
                    paths = x_List
                    random.shuffle(paths)
                    x_List = paths       
            x, y = load_imgs(x_List[i*batch_size:(i+1)*batch_size], cateList = cates)
            y = tf.one_hot(y, len(cates))
            yield x,y
            i = (i+1) % (len(x_List) // batch_size)


# In[ ]:


def MSE(y_true, y_pred):
    return tf.reduce_sum(tf.reduce_mean(tf.square(y_pred - y_true)))

def Cross_Entropy(y_true, y_pred):
    return -tf.reduce_sum(y_true*tf.math.log(y_pred+1e-10))

def CC_Pen(y_true, y_pred):
    cc = tf.keras.losses.categorical_crossentropy(y_true,y_pred)
    fibr = tf.constant([0.,0.,1.])
    p = cc*tf.reduce_mean(tf.abs(y_pred-fibr))
    loss = cc + p
    return loss


# In[ ]:


def divide_data(data_path='./Images', train_path='./Train', test_path='./Test', test_ratio=0.3):
    if bool(1-os.path.exists(train_path)): # train path是否存在
        os.mkdir(train_path)
    if bool(1-os.path.exists(test_path)): # test path是否存在
        os.mkdir(test_path)
    cates = os.listdir(data_path)
    for c in cates:
        cate_path = os.path.join(data_path,c)
        imgs = os.listdir(cate_path)
        random.shuffle(imgs)
        num = len(imgs) # 类别c的总数
        test_num = int(num*test_ratio)
        train_num = num - test_num
        print("Total number of {} = {}, number for training = {}, number for testing = {}.".format(c,num,train_num,test_num))
        # train path / test path 子目录是否存在
        if bool(1-os.path.exists(os.path.join(train_path,c))):
            print("create {} in {}".format(c,train_path))
            os.mkdir(os.path.join(train_path,c))
        if bool(1-os.path.exists(os.path.join(test_path,c))):
            print("create {} in {}".format(c,test_path))
            os.mkdir(os.path.join(test_path,c))
        # 复制测试数据
        for i in range(test_num):
            shutil.copy(os.path.join(cate_path,imgs[i]), os.path.join(test_path,c))
        # 复制训练数据
        for i in range(train_num):
            shutil.copy(os.path.join(cate_path,imgs[test_num+i]), os.path.join(train_path,c))
        print("Succeed to divide cate {}.\n--------------------------------------".format(c))


# In[ ]:

def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
 
    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)
 
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
 
    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
 
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
 
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
 
    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)
 
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


