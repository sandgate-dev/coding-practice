# Implementation of a CNN Fire module for SqueezeNet

## Imports

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
```

```python
layers =  tf.contrib.keras.layers
models = tf.contrib.keras.models
```

## Implementation of Fire Module

![](../../../../.gitbook/assets/fire_module.png)

```python
def fire_module(x, fire_id, squeeze=16, expand=64):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"
    s_id = 'fire' + str(fire_id) + '/'

    # Squeeze layer
    x = layers.Convolution2D(squeeze, (1,1), padding='valid', name=s_id + sq1x1)(x)
    x = layers.Activation('relu', name=s_id + relu + sq1x1)

    # Expand layer 1x1 filters
    left = layers.Convolution2D(expand, (1,1), padding='valid', name=s_id + exp1x1)(x)
    left = layers.Activation('relu', name=s_id + relu + exp1x1)(left)

    # Expand layer 3x3 filters
    right = layers.Convolution2D(expand, (3,3), padding='same', name=s_id + exp3x3)(x)
    right = layers.Activation('relu', name=s_id + relu + exp3x3)(right)

    # concatenate outputs
    x = layers.concatenate([left, right], axis=3, name=s_id + 'concat')


    return x
```

## Implementation of SqueezeNet

```python
def SqueezeNet(input_shape=(32,32,3), classes=10):

    img_input = layers.Input(shape=input_shape)

    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = layers.Activation('relu', name='relu_conv1')(x)
#     x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = layers.Dropout(0.5, name='drop9')(x)

    x = layers.Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = layers.Activation('relu', name='relu_conv10')(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Activation('softmax', name='loss')(x)

    model = models.Model(img_input, out, name='squeezenet')

    return model
```

## Inspect SqueezeNet Architecture

```python
squeeze_net = SqueezeNet()
squeeze_net.summary()
```

```text
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_3 (InputLayer)             (None, 32, 32, 3)     0                                            
____________________________________________________________________________________________________
conv1 (Conv2D)                   (None, 15, 15, 64)    1792        input_3[0][0]                    
____________________________________________________________________________________________________
relu_conv1 (Activation)          (None, 15, 15, 64)    0           conv1[0][0]                      
____________________________________________________________________________________________________
fire2/squeeze1x1 (Conv2D)        (None, 15, 15, 16)    1040        relu_conv1[0][0]                 
____________________________________________________________________________________________________
fire2/relu_squeeze1x1 (Activatio (None, 15, 15, 16)    0           fire2/squeeze1x1[0][0]           
____________________________________________________________________________________________________
fire2/expand1x1 (Conv2D)         (None, 15, 15, 64)    1088        fire2/relu_squeeze1x1[0][0]      
____________________________________________________________________________________________________
fire2/expand3x3 (Conv2D)         (None, 15, 15, 64)    9280        fire2/relu_squeeze1x1[0][0]      
____________________________________________________________________________________________________
fire2/relu_expand1x1 (Activation (None, 15, 15, 64)    0           fire2/expand1x1[0][0]            
____________________________________________________________________________________________________
fire2/relu_expand3x3 (Activation (None, 15, 15, 64)    0           fire2/expand3x3[0][0]            
____________________________________________________________________________________________________
fire2/concat (Concatenate)       (None, 15, 15, 128)   0           fire2/relu_expand1x1[0][0]       
                                                                   fire2/relu_expand3x3[0][0]       
____________________________________________________________________________________________________
fire3/squeeze1x1 (Conv2D)        (None, 15, 15, 16)    2064        fire2/concat[0][0]               
____________________________________________________________________________________________________
fire3/relu_squeeze1x1 (Activatio (None, 15, 15, 16)    0           fire3/squeeze1x1[0][0]           
____________________________________________________________________________________________________
fire3/expand1x1 (Conv2D)         (None, 15, 15, 64)    1088        fire3/relu_squeeze1x1[0][0]      
____________________________________________________________________________________________________
fire3/expand3x3 (Conv2D)         (None, 15, 15, 64)    9280        fire3/relu_squeeze1x1[0][0]      
____________________________________________________________________________________________________
fire3/relu_expand1x1 (Activation (None, 15, 15, 64)    0           fire3/expand1x1[0][0]            
____________________________________________________________________________________________________
fire3/relu_expand3x3 (Activation (None, 15, 15, 64)    0           fire3/expand3x3[0][0]            
____________________________________________________________________________________________________
fire3/concat (Concatenate)       (None, 15, 15, 128)   0           fire3/relu_expand1x1[0][0]       
                                                                   fire3/relu_expand3x3[0][0]       
____________________________________________________________________________________________________
pool3 (MaxPooling2D)             (None, 7, 7, 128)     0           fire3/concat[0][0]               
____________________________________________________________________________________________________
fire4/squeeze1x1 (Conv2D)        (None, 7, 7, 32)      4128        pool3[0][0]                      
____________________________________________________________________________________________________
fire4/relu_squeeze1x1 (Activatio (None, 7, 7, 32)      0           fire4/squeeze1x1[0][0]           
____________________________________________________________________________________________________
fire4/expand1x1 (Conv2D)         (None, 7, 7, 128)     4224        fire4/relu_squeeze1x1[0][0]      
____________________________________________________________________________________________________
fire4/expand3x3 (Conv2D)         (None, 7, 7, 128)     36992       fire4/relu_squeeze1x1[0][0]      
____________________________________________________________________________________________________
fire4/relu_expand1x1 (Activation (None, 7, 7, 128)     0           fire4/expand1x1[0][0]            
____________________________________________________________________________________________________
fire4/relu_expand3x3 (Activation (None, 7, 7, 128)     0           fire4/expand3x3[0][0]            
____________________________________________________________________________________________________
fire4/concat (Concatenate)       (None, 7, 7, 256)     0           fire4/relu_expand1x1[0][0]       
                                                                   fire4/relu_expand3x3[0][0]       
____________________________________________________________________________________________________
fire5/squeeze1x1 (Conv2D)        (None, 7, 7, 32)      8224        fire4/concat[0][0]               
____________________________________________________________________________________________________
fire5/relu_squeeze1x1 (Activatio (None, 7, 7, 32)      0           fire5/squeeze1x1[0][0]           
____________________________________________________________________________________________________
fire5/expand1x1 (Conv2D)         (None, 7, 7, 128)     4224        fire5/relu_squeeze1x1[0][0]      
____________________________________________________________________________________________________
fire5/expand3x3 (Conv2D)         (None, 7, 7, 128)     36992       fire5/relu_squeeze1x1[0][0]      
____________________________________________________________________________________________________
fire5/relu_expand1x1 (Activation (None, 7, 7, 128)     0           fire5/expand1x1[0][0]            
____________________________________________________________________________________________________
fire5/relu_expand3x3 (Activation (None, 7, 7, 128)     0           fire5/expand3x3[0][0]            
____________________________________________________________________________________________________
fire5/concat (Concatenate)       (None, 7, 7, 256)     0           fire5/relu_expand1x1[0][0]       
                                                                   fire5/relu_expand3x3[0][0]       
____________________________________________________________________________________________________
drop9 (Dropout)                  (None, 7, 7, 256)     0           fire5/concat[0][0]               
____________________________________________________________________________________________________
conv10 (Conv2D)                  (None, 7, 7, 10)      2570        drop9[0][0]                      
____________________________________________________________________________________________________
relu_conv10 (Activation)         (None, 7, 7, 10)      0           conv10[0][0]                     
____________________________________________________________________________________________________
global_average_pooling2d_3 (Glob (None, 10)            0           relu_conv10[0][0]                
____________________________________________________________________________________________________
loss (Activation)                (None, 10)            0           global_average_pooling2d_3[0][0] 
====================================================================================================
Total params: 122,986
Trainable params: 122,986
Non-trainable params: 0
____________________________________________________________________________________________________
```

### Next Lesson

#### Train and Evaluate SqueezeNet

* Image classification task with Cifar10

![](../../images/divider.png)

