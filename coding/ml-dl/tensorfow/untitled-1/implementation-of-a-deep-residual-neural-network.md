# Implementation of a Deep Residual Neural Network

## Imports

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
```

```python
models = tf.contrib.keras.models
layers = tf.contrib.keras.layers
initializers = tf.contrib.keras.initializers
regularizers = tf.contrib.keras.regularizers
```

## Pre-activation Bottleneck Residual Block

![](../../../../.gitbook/assets/pre-activation%20%281%29.png)

![](../../../../.gitbook/assets/bottleneck%20%281%29.png)

```python
def residual_block(input_tensor, filters, stage, reg=0.0, use_shortcuts=True):

    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'relu' + str(stage)
    merge_name = 'merge' + str(stage)

    # 1x1 conv
    # batchnorm-relu-conv
    # from input_filters to bottleneck_filters
    if stage>1: # first activation is just after conv1
        x = layers.BatchNormalization(name=bn_name+'a')(input_tensor)
        x = layers.Activation('relu', name=relu_name+'a')(x)
    else:
        x = input_tensor

    x = layers.Convolution2D(
            filters[0], (1,1),
            kernel_regularizer=regularizers.l2(reg),
            use_bias=False,
            name=conv_name+'a'
        )(x)

    # 3x3 conv
    # batchnorm-relu-conv
    # from bottleneck_filters to bottleneck_filters
    x = layers.BatchNormalization(name=bn_name+'b')(x)
    x = layers.Activation('relu', name=relu_name+'b')(x)
    x = layers.Convolution2D(
            filters[1], (3,3),
            padding='same',
            kernel_regularizer=regularizers.l2(reg),
            use_bias = False,
            name=conv_name+'b'
        )(x)

    # 1x1 conv
    # batchnorm-relu-conv
    # from bottleneck_filters  to input_filters
    x = layers.BatchNormalization(name=bn_name+'c')(x)
    x = layers.Activation('relu', name=relu_name+'c')(x)
    x = layers.Convolution2D(
            filters[2], (1,1),
            kernel_regularizer=regularizers.l2(reg),
            name=conv_name+'c'
        )(x)

    # merge output with input layer (residual connection)
    if use_shortcuts:
        x = layers.add([x, input_tensor], name=merge_name)

    return x
```

## Full Residual Network

```python
def ResNetPreAct(input_shape=(32,32,3), nb_classes=5, num_stages=5,
                 use_final_conv=False, reg=0.0):


    # Input
    img_input = layers.Input(shape=input_shape)

    #### Input stream ####
    # conv-BN-relu-(pool)
    x = layers.Convolution2D(
            128, (3,3), strides=(2, 2),
            padding='same',
            kernel_regularizer=regularizers.l2(reg),
            use_bias=False,
            name='conv0'
        )(img_input)
    x = layers.BatchNormalization(name='bn0')(x)
    x = layers.Activation('relu', name='relu0')(x)
#     x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool0')(x)

    #### Residual Blocks ####
    # 1x1 conv: batchnorm-relu-conv
    # 3x3 conv: batchnorm-relu-conv
    # 1x1 conv: batchnorm-relu-conv
    for stage in range(1,num_stages+1):
        x = residual_block(x, [32,32,128], stage=stage, reg=reg)


    #### Output stream ####
    # BN-relu-(conv)-avgPool-softmax
    x = layers.BatchNormalization(name='bnF')(x)
    x = layers.Activation('relu', name='reluF')(x)

    # Optional final conv layer
    if use_final_conv:
        x = layers.Convolution2D(
                64, (3,3),
                padding='same',
                kernel_regularizer=regularizers.l2(reg),
                name='convF'
            )(x)

    pool_size = input_shape[0] / 2
    x = layers.AveragePooling2D((pool_size,pool_size),name='avg_pool')(x)

    x = layers.Flatten(name='flat')(x)
    x = layers.Dense(nb_classes, activation='softmax', name='fc10')(x)

    return models.Model(img_input, x, name='rnpa')
```

## Inspect Model Architecture

```python
model = ResNetPreAct()
model.summary()
```

```text
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_3 (InputLayer)             (None, 32, 32, 3)     0                                            
____________________________________________________________________________________________________
conv0 (Conv2D)                   (None, 16, 16, 128)   3456        input_3[0][0]                    
____________________________________________________________________________________________________
bn0 (BatchNormalization)         (None, 16, 16, 128)   512         conv0[0][0]                      
____________________________________________________________________________________________________
relu0 (Activation)               (None, 16, 16, 128)   0           bn0[0][0]                        
____________________________________________________________________________________________________
conv1a (Conv2D)                  (None, 16, 16, 32)    4096        relu0[0][0]                      
____________________________________________________________________________________________________
bn1b (BatchNormalization)        (None, 16, 16, 32)    128         conv1a[0][0]                     
____________________________________________________________________________________________________
relu1b (Activation)              (None, 16, 16, 32)    0           bn1b[0][0]                       
____________________________________________________________________________________________________
conv1b (Conv2D)                  (None, 16, 16, 32)    9216        relu1b[0][0]                     
____________________________________________________________________________________________________
bn1c (BatchNormalization)        (None, 16, 16, 32)    128         conv1b[0][0]                     
____________________________________________________________________________________________________
relu1c (Activation)              (None, 16, 16, 32)    0           bn1c[0][0]                       
____________________________________________________________________________________________________
conv1c (Conv2D)                  (None, 16, 16, 128)   4224        relu1c[0][0]                     
____________________________________________________________________________________________________
merge1 (Add)                     (None, 16, 16, 128)   0           conv1c[0][0]                     
                                                                   relu0[0][0]                      
____________________________________________________________________________________________________
bn2a (BatchNormalization)        (None, 16, 16, 128)   512         merge1[0][0]                     
____________________________________________________________________________________________________
relu2a (Activation)              (None, 16, 16, 128)   0           bn2a[0][0]                       
____________________________________________________________________________________________________
conv2a (Conv2D)                  (None, 16, 16, 32)    4096        relu2a[0][0]                     
____________________________________________________________________________________________________
bn2b (BatchNormalization)        (None, 16, 16, 32)    128         conv2a[0][0]                     
____________________________________________________________________________________________________
relu2b (Activation)              (None, 16, 16, 32)    0           bn2b[0][0]                       
____________________________________________________________________________________________________
conv2b (Conv2D)                  (None, 16, 16, 32)    9216        relu2b[0][0]                     
____________________________________________________________________________________________________
bn2c (BatchNormalization)        (None, 16, 16, 32)    128         conv2b[0][0]                     
____________________________________________________________________________________________________
relu2c (Activation)              (None, 16, 16, 32)    0           bn2c[0][0]                       
____________________________________________________________________________________________________
conv2c (Conv2D)                  (None, 16, 16, 128)   4224        relu2c[0][0]                     
____________________________________________________________________________________________________
merge2 (Add)                     (None, 16, 16, 128)   0           conv2c[0][0]                     
                                                                   merge1[0][0]                     
____________________________________________________________________________________________________
bn3a (BatchNormalization)        (None, 16, 16, 128)   512         merge2[0][0]                     
____________________________________________________________________________________________________
relu3a (Activation)              (None, 16, 16, 128)   0           bn3a[0][0]                       
____________________________________________________________________________________________________
conv3a (Conv2D)                  (None, 16, 16, 32)    4096        relu3a[0][0]                     
____________________________________________________________________________________________________
bn3b (BatchNormalization)        (None, 16, 16, 32)    128         conv3a[0][0]                     
____________________________________________________________________________________________________
relu3b (Activation)              (None, 16, 16, 32)    0           bn3b[0][0]                       
____________________________________________________________________________________________________
conv3b (Conv2D)                  (None, 16, 16, 32)    9216        relu3b[0][0]                     
____________________________________________________________________________________________________
bn3c (BatchNormalization)        (None, 16, 16, 32)    128         conv3b[0][0]                     
____________________________________________________________________________________________________
relu3c (Activation)              (None, 16, 16, 32)    0           bn3c[0][0]                       
____________________________________________________________________________________________________
conv3c (Conv2D)                  (None, 16, 16, 128)   4224        relu3c[0][0]                     
____________________________________________________________________________________________________
merge3 (Add)                     (None, 16, 16, 128)   0           conv3c[0][0]                     
                                                                   merge2[0][0]                     
____________________________________________________________________________________________________
bn4a (BatchNormalization)        (None, 16, 16, 128)   512         merge3[0][0]                     
____________________________________________________________________________________________________
relu4a (Activation)              (None, 16, 16, 128)   0           bn4a[0][0]                       
____________________________________________________________________________________________________
conv4a (Conv2D)                  (None, 16, 16, 32)    4096        relu4a[0][0]                     
____________________________________________________________________________________________________
bn4b (BatchNormalization)        (None, 16, 16, 32)    128         conv4a[0][0]                     
____________________________________________________________________________________________________
relu4b (Activation)              (None, 16, 16, 32)    0           bn4b[0][0]                       
____________________________________________________________________________________________________
conv4b (Conv2D)                  (None, 16, 16, 32)    9216        relu4b[0][0]                     
____________________________________________________________________________________________________
bn4c (BatchNormalization)        (None, 16, 16, 32)    128         conv4b[0][0]                     
____________________________________________________________________________________________________
relu4c (Activation)              (None, 16, 16, 32)    0           bn4c[0][0]                       
____________________________________________________________________________________________________
conv4c (Conv2D)                  (None, 16, 16, 128)   4224        relu4c[0][0]                     
____________________________________________________________________________________________________
merge4 (Add)                     (None, 16, 16, 128)   0           conv4c[0][0]                     
                                                                   merge3[0][0]                     
____________________________________________________________________________________________________
bn5a (BatchNormalization)        (None, 16, 16, 128)   512         merge4[0][0]                     
____________________________________________________________________________________________________
relu5a (Activation)              (None, 16, 16, 128)   0           bn5a[0][0]                       
____________________________________________________________________________________________________
conv5a (Conv2D)                  (None, 16, 16, 32)    4096        relu5a[0][0]                     
____________________________________________________________________________________________________
bn5b (BatchNormalization)        (None, 16, 16, 32)    128         conv5a[0][0]                     
____________________________________________________________________________________________________
relu5b (Activation)              (None, 16, 16, 32)    0           bn5b[0][0]                       
____________________________________________________________________________________________________
conv5b (Conv2D)                  (None, 16, 16, 32)    9216        relu5b[0][0]                     
____________________________________________________________________________________________________
bn5c (BatchNormalization)        (None, 16, 16, 32)    128         conv5b[0][0]                     
____________________________________________________________________________________________________
relu5c (Activation)              (None, 16, 16, 32)    0           bn5c[0][0]                       
____________________________________________________________________________________________________
conv5c (Conv2D)                  (None, 16, 16, 128)   4224        relu5c[0][0]                     
____________________________________________________________________________________________________
merge5 (Add)                     (None, 16, 16, 128)   0           conv5c[0][0]                     
                                                                   merge4[0][0]                     
____________________________________________________________________________________________________
bnF (BatchNormalization)         (None, 16, 16, 128)   512         merge5[0][0]                     
____________________________________________________________________________________________________
reluF (Activation)               (None, 16, 16, 128)   0           bnF[0][0]                        
____________________________________________________________________________________________________
avg_pool (AveragePooling2D)      (None, 1, 1, 128)     0           reluF[0][0]                      
____________________________________________________________________________________________________
flat (Flatten)                   (None, 128)           0           avg_pool[0][0]                   
____________________________________________________________________________________________________
fc10 (Dense)                     (None, 5)             645         flat[0][0]                       
====================================================================================================
Total params: 96,133
Trainable params: 93,957
Non-trainable params: 2,176
____________________________________________________________________________________________________
```

### Next Lesson

#### Train and Evaluate ResNet

* Image classification task with Flower dataset

