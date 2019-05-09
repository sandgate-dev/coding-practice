# Training and Evaluating Xception Model

## Imports

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
```

```python
models = tf.contrib.keras.models
layers = tf.contrib.keras.layers
utils = tf.contrib.keras.utils
losses = tf.contrib.keras.losses
optimizers = tf.contrib.keras.optimizers 
metrics = tf.contrib.keras.metrics
preprocessing_image = tf.contrib.keras.preprocessing.image
applications = tf.contrib.keras.applications
```

## Load Pre-Trained Xception Model

```python
# load pre-trained Xception model and exclude top dense layer
base_model = applications.Xception(include_top=False,
                                   weights='imagenet',
                                   input_shape=(299,299,3),
                                   pooling='avg')
```

```python
print("Model input shape: {}\n".format(base_model.input_shape))
print("Model output shape: {}\n".format(base_model.output_shape))
print("Model number of layers: {}\n".format(len(base_model.layers)))
```

```text
Model input shape: (None, 299, 299, 3)

Model output shape: (None, 2048)

Model number of layers: 133
```

## Fine-tune Xception Model

```python
def fine_tune_Xception(base_model):

    # output of convolutional layers
    x = base_model.output

    # final Dense layer
    outputs = layers.Dense(4, activation='softmax')(x)

    # define model with base_model's input
    model = models.Model(inputs=base_model.input, outputs=outputs)

    # freeze weights of early layers
    # to ease training
    for layer in model.layers[:40]:
        layer.trainable = False

    return model
```

## Compile Model

```python
def compile_model(model):

    # loss
    loss = losses.categorical_crossentropy

    # optimizer
    optimizer = optimizers.RMSprop(lr=0.0001)

    # metrics
    metric = [metrics.categorical_accuracy]

    # compile model with loss, optimizer, and evaluation metrics
    model.compile(optimizer, loss, metric)

    return model
```

## Inspect Model Architecture

```python
model = fine_tune_Xception(base_model)
model = compile_model(model)
model.summary()
```

```text
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 299, 299, 3)   0                                            
____________________________________________________________________________________________________
block1_conv1 (Conv2D)            (None, 149, 149, 32)  864         input_1[0][0]                    
____________________________________________________________________________________________________
block1_conv1_bn (BatchNormalizat (None, 149, 149, 32)  128         block1_conv1[0][0]               
____________________________________________________________________________________________________
block1_conv1_act (Activation)    (None, 149, 149, 32)  0           block1_conv1_bn[0][0]            
____________________________________________________________________________________________________
block1_conv2 (Conv2D)            (None, 147, 147, 64)  18432       block1_conv1_act[0][0]           
____________________________________________________________________________________________________
block1_conv2_bn (BatchNormalizat (None, 147, 147, 64)  256         block1_conv2[0][0]               
____________________________________________________________________________________________________
block1_conv2_act (Activation)    (None, 147, 147, 64)  0           block1_conv2_bn[0][0]            
____________________________________________________________________________________________________
block2_sepconv1 (SeparableConv2D (None, 147, 147, 128) 8768        block1_conv2_act[0][0]           
____________________________________________________________________________________________________
block2_sepconv1_bn (BatchNormali (None, 147, 147, 128) 512         block2_sepconv1[0][0]            
____________________________________________________________________________________________________
block2_sepconv2_act (Activation) (None, 147, 147, 128) 0           block2_sepconv1_bn[0][0]         
____________________________________________________________________________________________________
block2_sepconv2 (SeparableConv2D (None, 147, 147, 128) 17536       block2_sepconv2_act[0][0]        
____________________________________________________________________________________________________
block2_sepconv2_bn (BatchNormali (None, 147, 147, 128) 512         block2_sepconv2[0][0]            
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 74, 74, 128)   8192        block1_conv2_act[0][0]           
____________________________________________________________________________________________________
block2_pool (MaxPooling2D)       (None, 74, 74, 128)   0           block2_sepconv2_bn[0][0]         
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 74, 74, 128)   512         conv2d_1[0][0]                   
____________________________________________________________________________________________________
add_1 (Add)                      (None, 74, 74, 128)   0           block2_pool[0][0]                
                                                                   batch_normalization_1[0][0]      
____________________________________________________________________________________________________
block3_sepconv1_act (Activation) (None, 74, 74, 128)   0           add_1[0][0]                      
____________________________________________________________________________________________________
block3_sepconv1 (SeparableConv2D (None, 74, 74, 256)   33920       block3_sepconv1_act[0][0]        
____________________________________________________________________________________________________
block3_sepconv1_bn (BatchNormali (None, 74, 74, 256)   1024        block3_sepconv1[0][0]            
____________________________________________________________________________________________________
block3_sepconv2_act (Activation) (None, 74, 74, 256)   0           block3_sepconv1_bn[0][0]         
____________________________________________________________________________________________________
block3_sepconv2 (SeparableConv2D (None, 74, 74, 256)   67840       block3_sepconv2_act[0][0]        
____________________________________________________________________________________________________
block3_sepconv2_bn (BatchNormali (None, 74, 74, 256)   1024        block3_sepconv2[0][0]            
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 37, 37, 256)   32768       add_1[0][0]                      
____________________________________________________________________________________________________
block3_pool (MaxPooling2D)       (None, 37, 37, 256)   0           block3_sepconv2_bn[0][0]         
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 37, 37, 256)   1024        conv2d_2[0][0]                   
____________________________________________________________________________________________________
add_2 (Add)                      (None, 37, 37, 256)   0           block3_pool[0][0]                
                                                                   batch_normalization_2[0][0]      
____________________________________________________________________________________________________
block4_sepconv1_act (Activation) (None, 37, 37, 256)   0           add_2[0][0]                      
____________________________________________________________________________________________________
block4_sepconv1 (SeparableConv2D (None, 37, 37, 728)   188672      block4_sepconv1_act[0][0]        
____________________________________________________________________________________________________
block4_sepconv1_bn (BatchNormali (None, 37, 37, 728)   2912        block4_sepconv1[0][0]            
____________________________________________________________________________________________________
block4_sepconv2_act (Activation) (None, 37, 37, 728)   0           block4_sepconv1_bn[0][0]         
____________________________________________________________________________________________________
block4_sepconv2 (SeparableConv2D (None, 37, 37, 728)   536536      block4_sepconv2_act[0][0]        
____________________________________________________________________________________________________
block4_sepconv2_bn (BatchNormali (None, 37, 37, 728)   2912        block4_sepconv2[0][0]            
____________________________________________________________________________________________________
conv2d_3 (Conv2D)                (None, 19, 19, 728)   186368      add_2[0][0]                      
____________________________________________________________________________________________________
block4_pool (MaxPooling2D)       (None, 19, 19, 728)   0           block4_sepconv2_bn[0][0]         
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 19, 19, 728)   2912        conv2d_3[0][0]                   
____________________________________________________________________________________________________
add_3 (Add)                      (None, 19, 19, 728)   0           block4_pool[0][0]                
                                                                   batch_normalization_3[0][0]      
____________________________________________________________________________________________________
block5_sepconv1_act (Activation) (None, 19, 19, 728)   0           add_3[0][0]                      
____________________________________________________________________________________________________
block5_sepconv1 (SeparableConv2D (None, 19, 19, 728)   536536      block5_sepconv1_act[0][0]        
____________________________________________________________________________________________________
block5_sepconv1_bn (BatchNormali (None, 19, 19, 728)   2912        block5_sepconv1[0][0]            
____________________________________________________________________________________________________
block5_sepconv2_act (Activation) (None, 19, 19, 728)   0           block5_sepconv1_bn[0][0]         
____________________________________________________________________________________________________
block5_sepconv2 (SeparableConv2D (None, 19, 19, 728)   536536      block5_sepconv2_act[0][0]        
____________________________________________________________________________________________________
block5_sepconv2_bn (BatchNormali (None, 19, 19, 728)   2912        block5_sepconv2[0][0]            
____________________________________________________________________________________________________
block5_sepconv3_act (Activation) (None, 19, 19, 728)   0           block5_sepconv2_bn[0][0]         
____________________________________________________________________________________________________
block5_sepconv3 (SeparableConv2D (None, 19, 19, 728)   536536      block5_sepconv3_act[0][0]        
____________________________________________________________________________________________________
block5_sepconv3_bn (BatchNormali (None, 19, 19, 728)   2912        block5_sepconv3[0][0]            
____________________________________________________________________________________________________
add_4 (Add)                      (None, 19, 19, 728)   0           block5_sepconv3_bn[0][0]         
                                                                   add_3[0][0]                      
____________________________________________________________________________________________________
block6_sepconv1_act (Activation) (None, 19, 19, 728)   0           add_4[0][0]                      
____________________________________________________________________________________________________
block6_sepconv1 (SeparableConv2D (None, 19, 19, 728)   536536      block6_sepconv1_act[0][0]        
____________________________________________________________________________________________________
block6_sepconv1_bn (BatchNormali (None, 19, 19, 728)   2912        block6_sepconv1[0][0]            
____________________________________________________________________________________________________
block6_sepconv2_act (Activation) (None, 19, 19, 728)   0           block6_sepconv1_bn[0][0]         
____________________________________________________________________________________________________
block6_sepconv2 (SeparableConv2D (None, 19, 19, 728)   536536      block6_sepconv2_act[0][0]        
____________________________________________________________________________________________________
block6_sepconv2_bn (BatchNormali (None, 19, 19, 728)   2912        block6_sepconv2[0][0]            
____________________________________________________________________________________________________
block6_sepconv3_act (Activation) (None, 19, 19, 728)   0           block6_sepconv2_bn[0][0]         
____________________________________________________________________________________________________
block6_sepconv3 (SeparableConv2D (None, 19, 19, 728)   536536      block6_sepconv3_act[0][0]        
____________________________________________________________________________________________________
block6_sepconv3_bn (BatchNormali (None, 19, 19, 728)   2912        block6_sepconv3[0][0]            
____________________________________________________________________________________________________
add_5 (Add)                      (None, 19, 19, 728)   0           block6_sepconv3_bn[0][0]         
                                                                   add_4[0][0]                      
____________________________________________________________________________________________________
block7_sepconv1_act (Activation) (None, 19, 19, 728)   0           add_5[0][0]                      
____________________________________________________________________________________________________
block7_sepconv1 (SeparableConv2D (None, 19, 19, 728)   536536      block7_sepconv1_act[0][0]        
____________________________________________________________________________________________________
block7_sepconv1_bn (BatchNormali (None, 19, 19, 728)   2912        block7_sepconv1[0][0]            
____________________________________________________________________________________________________
block7_sepconv2_act (Activation) (None, 19, 19, 728)   0           block7_sepconv1_bn[0][0]         
____________________________________________________________________________________________________
block7_sepconv2 (SeparableConv2D (None, 19, 19, 728)   536536      block7_sepconv2_act[0][0]        
____________________________________________________________________________________________________
block7_sepconv2_bn (BatchNormali (None, 19, 19, 728)   2912        block7_sepconv2[0][0]            
____________________________________________________________________________________________________
block7_sepconv3_act (Activation) (None, 19, 19, 728)   0           block7_sepconv2_bn[0][0]         
____________________________________________________________________________________________________
block7_sepconv3 (SeparableConv2D (None, 19, 19, 728)   536536      block7_sepconv3_act[0][0]        
____________________________________________________________________________________________________
block7_sepconv3_bn (BatchNormali (None, 19, 19, 728)   2912        block7_sepconv3[0][0]            
____________________________________________________________________________________________________
add_6 (Add)                      (None, 19, 19, 728)   0           block7_sepconv3_bn[0][0]         
                                                                   add_5[0][0]                      
____________________________________________________________________________________________________
block8_sepconv1_act (Activation) (None, 19, 19, 728)   0           add_6[0][0]                      
____________________________________________________________________________________________________
block8_sepconv1 (SeparableConv2D (None, 19, 19, 728)   536536      block8_sepconv1_act[0][0]        
____________________________________________________________________________________________________
block8_sepconv1_bn (BatchNormali (None, 19, 19, 728)   2912        block8_sepconv1[0][0]            
____________________________________________________________________________________________________
block8_sepconv2_act (Activation) (None, 19, 19, 728)   0           block8_sepconv1_bn[0][0]         
____________________________________________________________________________________________________
block8_sepconv2 (SeparableConv2D (None, 19, 19, 728)   536536      block8_sepconv2_act[0][0]        
____________________________________________________________________________________________________
block8_sepconv2_bn (BatchNormali (None, 19, 19, 728)   2912        block8_sepconv2[0][0]            
____________________________________________________________________________________________________
block8_sepconv3_act (Activation) (None, 19, 19, 728)   0           block8_sepconv2_bn[0][0]         
____________________________________________________________________________________________________
block8_sepconv3 (SeparableConv2D (None, 19, 19, 728)   536536      block8_sepconv3_act[0][0]        
____________________________________________________________________________________________________
block8_sepconv3_bn (BatchNormali (None, 19, 19, 728)   2912        block8_sepconv3[0][0]            
____________________________________________________________________________________________________
add_7 (Add)                      (None, 19, 19, 728)   0           block8_sepconv3_bn[0][0]         
                                                                   add_6[0][0]                      
____________________________________________________________________________________________________
block9_sepconv1_act (Activation) (None, 19, 19, 728)   0           add_7[0][0]                      
____________________________________________________________________________________________________
block9_sepconv1 (SeparableConv2D (None, 19, 19, 728)   536536      block9_sepconv1_act[0][0]        
____________________________________________________________________________________________________
block9_sepconv1_bn (BatchNormali (None, 19, 19, 728)   2912        block9_sepconv1[0][0]            
____________________________________________________________________________________________________
block9_sepconv2_act (Activation) (None, 19, 19, 728)   0           block9_sepconv1_bn[0][0]         
____________________________________________________________________________________________________
block9_sepconv2 (SeparableConv2D (None, 19, 19, 728)   536536      block9_sepconv2_act[0][0]        
____________________________________________________________________________________________________
block9_sepconv2_bn (BatchNormali (None, 19, 19, 728)   2912        block9_sepconv2[0][0]            
____________________________________________________________________________________________________
block9_sepconv3_act (Activation) (None, 19, 19, 728)   0           block9_sepconv2_bn[0][0]         
____________________________________________________________________________________________________
block9_sepconv3 (SeparableConv2D (None, 19, 19, 728)   536536      block9_sepconv3_act[0][0]        
____________________________________________________________________________________________________
block9_sepconv3_bn (BatchNormali (None, 19, 19, 728)   2912        block9_sepconv3[0][0]            
____________________________________________________________________________________________________
add_8 (Add)                      (None, 19, 19, 728)   0           block9_sepconv3_bn[0][0]         
                                                                   add_7[0][0]                      
____________________________________________________________________________________________________
block10_sepconv1_act (Activation (None, 19, 19, 728)   0           add_8[0][0]                      
____________________________________________________________________________________________________
block10_sepconv1 (SeparableConv2 (None, 19, 19, 728)   536536      block10_sepconv1_act[0][0]       
____________________________________________________________________________________________________
block10_sepconv1_bn (BatchNormal (None, 19, 19, 728)   2912        block10_sepconv1[0][0]           
____________________________________________________________________________________________________
block10_sepconv2_act (Activation (None, 19, 19, 728)   0           block10_sepconv1_bn[0][0]        
____________________________________________________________________________________________________
block10_sepconv2 (SeparableConv2 (None, 19, 19, 728)   536536      block10_sepconv2_act[0][0]       
____________________________________________________________________________________________________
block10_sepconv2_bn (BatchNormal (None, 19, 19, 728)   2912        block10_sepconv2[0][0]           
____________________________________________________________________________________________________
block10_sepconv3_act (Activation (None, 19, 19, 728)   0           block10_sepconv2_bn[0][0]        
____________________________________________________________________________________________________
block10_sepconv3 (SeparableConv2 (None, 19, 19, 728)   536536      block10_sepconv3_act[0][0]       
____________________________________________________________________________________________________
block10_sepconv3_bn (BatchNormal (None, 19, 19, 728)   2912        block10_sepconv3[0][0]           
____________________________________________________________________________________________________
add_9 (Add)                      (None, 19, 19, 728)   0           block10_sepconv3_bn[0][0]        
                                                                   add_8[0][0]                      
____________________________________________________________________________________________________
block11_sepconv1_act (Activation (None, 19, 19, 728)   0           add_9[0][0]                      
____________________________________________________________________________________________________
block11_sepconv1 (SeparableConv2 (None, 19, 19, 728)   536536      block11_sepconv1_act[0][0]       
____________________________________________________________________________________________________
block11_sepconv1_bn (BatchNormal (None, 19, 19, 728)   2912        block11_sepconv1[0][0]           
____________________________________________________________________________________________________
block11_sepconv2_act (Activation (None, 19, 19, 728)   0           block11_sepconv1_bn[0][0]        
____________________________________________________________________________________________________
block11_sepconv2 (SeparableConv2 (None, 19, 19, 728)   536536      block11_sepconv2_act[0][0]       
____________________________________________________________________________________________________
block11_sepconv2_bn (BatchNormal (None, 19, 19, 728)   2912        block11_sepconv2[0][0]           
____________________________________________________________________________________________________
block11_sepconv3_act (Activation (None, 19, 19, 728)   0           block11_sepconv2_bn[0][0]        
____________________________________________________________________________________________________
block11_sepconv3 (SeparableConv2 (None, 19, 19, 728)   536536      block11_sepconv3_act[0][0]       
____________________________________________________________________________________________________
block11_sepconv3_bn (BatchNormal (None, 19, 19, 728)   2912        block11_sepconv3[0][0]           
____________________________________________________________________________________________________
add_10 (Add)                     (None, 19, 19, 728)   0           block11_sepconv3_bn[0][0]        
                                                                   add_9[0][0]                      
____________________________________________________________________________________________________
block12_sepconv1_act (Activation (None, 19, 19, 728)   0           add_10[0][0]                     
____________________________________________________________________________________________________
block12_sepconv1 (SeparableConv2 (None, 19, 19, 728)   536536      block12_sepconv1_act[0][0]       
____________________________________________________________________________________________________
block12_sepconv1_bn (BatchNormal (None, 19, 19, 728)   2912        block12_sepconv1[0][0]           
____________________________________________________________________________________________________
block12_sepconv2_act (Activation (None, 19, 19, 728)   0           block12_sepconv1_bn[0][0]        
____________________________________________________________________________________________________
block12_sepconv2 (SeparableConv2 (None, 19, 19, 728)   536536      block12_sepconv2_act[0][0]       
____________________________________________________________________________________________________
block12_sepconv2_bn (BatchNormal (None, 19, 19, 728)   2912        block12_sepconv2[0][0]           
____________________________________________________________________________________________________
block12_sepconv3_act (Activation (None, 19, 19, 728)   0           block12_sepconv2_bn[0][0]        
____________________________________________________________________________________________________
block12_sepconv3 (SeparableConv2 (None, 19, 19, 728)   536536      block12_sepconv3_act[0][0]       
____________________________________________________________________________________________________
block12_sepconv3_bn (BatchNormal (None, 19, 19, 728)   2912        block12_sepconv3[0][0]           
____________________________________________________________________________________________________
add_11 (Add)                     (None, 19, 19, 728)   0           block12_sepconv3_bn[0][0]        
                                                                   add_10[0][0]                     
____________________________________________________________________________________________________
block13_sepconv1_act (Activation (None, 19, 19, 728)   0           add_11[0][0]                     
____________________________________________________________________________________________________
block13_sepconv1 (SeparableConv2 (None, 19, 19, 728)   536536      block13_sepconv1_act[0][0]       
____________________________________________________________________________________________________
block13_sepconv1_bn (BatchNormal (None, 19, 19, 728)   2912        block13_sepconv1[0][0]           
____________________________________________________________________________________________________
block13_sepconv2_act (Activation (None, 19, 19, 728)   0           block13_sepconv1_bn[0][0]        
____________________________________________________________________________________________________
block13_sepconv2 (SeparableConv2 (None, 19, 19, 1024)  752024      block13_sepconv2_act[0][0]       
____________________________________________________________________________________________________
block13_sepconv2_bn (BatchNormal (None, 19, 19, 1024)  4096        block13_sepconv2[0][0]           
____________________________________________________________________________________________________
conv2d_4 (Conv2D)                (None, 10, 10, 1024)  745472      add_11[0][0]                     
____________________________________________________________________________________________________
block13_pool (MaxPooling2D)      (None, 10, 10, 1024)  0           block13_sepconv2_bn[0][0]        
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 10, 10, 1024)  4096        conv2d_4[0][0]                   
____________________________________________________________________________________________________
add_12 (Add)                     (None, 10, 10, 1024)  0           block13_pool[0][0]               
                                                                   batch_normalization_4[0][0]      
____________________________________________________________________________________________________
block14_sepconv1 (SeparableConv2 (None, 10, 10, 1536)  1582080     add_12[0][0]                     
____________________________________________________________________________________________________
block14_sepconv1_bn (BatchNormal (None, 10, 10, 1536)  6144        block14_sepconv1[0][0]           
____________________________________________________________________________________________________
block14_sepconv1_act (Activation (None, 10, 10, 1536)  0           block14_sepconv1_bn[0][0]        
____________________________________________________________________________________________________
block14_sepconv2 (SeparableConv2 (None, 10, 10, 2048)  3159552     block14_sepconv1_act[0][0]       
____________________________________________________________________________________________________
block14_sepconv2_bn (BatchNormal (None, 10, 10, 2048)  8192        block14_sepconv2[0][0]           
____________________________________________________________________________________________________
block14_sepconv2_act (Activation (None, 10, 10, 2048)  0           block14_sepconv2_bn[0][0]        
____________________________________________________________________________________________________
global_average_pooling2d_1 (Glob (None, 2048)          0           block14_sepconv2_act[0][0]       
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 4)             8196        global_average_pooling2d_1[0][0] 
====================================================================================================
Total params: 20,869,676
Trainable params: 19,170,396
Non-trainable params: 1,699,280
____________________________________________________________________________________________________
```

## Image Preprocessing And Augmentation

```python
def preprocess_image(x):
    x /= 255.
    x -= 0.5
    x *= 2.

    # 'RGB'->'BGR'
    x = x[..., ::-1]
    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x


train_datagen = preprocessing_image.ImageDataGenerator(
    preprocessing_function=preprocess_image,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = preprocessing_image.ImageDataGenerator(preprocessing_function=preprocess_image)
```

```python
BASE_DIR = "/Users/marvinbertin/Github/marvin/ImageNet_Utils"

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "imageNet_dataset/train"),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "imageNet_dataset/validation"),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)
```

```text
Found 2677 images belonging to 4 classes.
Found 668 images belonging to 4 classes.
```

## Train Model on ImageNet Dataset

```python
history = model.fit_generator(
    train_generator,
    steps_per_epoch=80,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=20)
```

```text
Epoch 1/10
80/80 [==============================] - 121s - loss: 0.9725 - categorical_accuracy: 0.5895 - val_loss: 12.2145 - val_categorical_accuracy: 0.2422
Epoch 2/10
80/80 [==============================] - 119s - loss: 0.8422 - categorical_accuracy: 0.6576 - val_loss: 3.3612 - val_categorical_accuracy: 0.2441
Epoch 3/10
80/80 [==============================] - 118s - loss: 0.8019 - categorical_accuracy: 0.6707 - val_loss: 1.3862 - val_categorical_accuracy: 0.2220
Epoch 4/10
80/80 [==============================] - 119s - loss: 0.7753 - categorical_accuracy: 0.6800 - val_loss: 4.0558 - val_categorical_accuracy: 0.1732
Epoch 5/10
80/80 [==============================] - 119s - loss: 0.7529 - categorical_accuracy: 0.6950 - val_loss: 2.9780 - val_categorical_accuracy: 0.2661
Epoch 6/10
80/80 [==============================] - 118s - loss: 0.7216 - categorical_accuracy: 0.7130 - val_loss: 1.6897 - val_categorical_accuracy: 0.5528
Epoch 7/10
80/80 [==============================] - 119s - loss: 0.6669 - categorical_accuracy: 0.7364 - val_loss: 4.1187 - val_categorical_accuracy: 0.1984
Epoch 8/10
80/80 [==============================] - 119s - loss: 0.6815 - categorical_accuracy: 0.7285 - val_loss: 1.1006 - val_categorical_accuracy: 0.6142
Epoch 9/10
80/80 [==============================] - 118s - loss: 0.6641 - categorical_accuracy: 0.7290 - val_loss: 2.0925 - val_categorical_accuracy: 0.5087
Epoch 10/10
80/80 [==============================] - 119s - loss: 0.6518 - categorical_accuracy: 0.7409 - val_loss: 1.2735 - val_categorical_accuracy: 0.5165
```

## Plot Accuracy And Loss Over Time

```python
def plot_accuracy_and_loss(history):
    plt.figure(1, figsize= (15, 10))

    # plot train and test accuracy
    plt.subplot(221)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('SqueezeNet accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # plot train and test loss
    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('SqueezeNet loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.show()
```

```python
plot_accuracy_and_loss(history)
```

![](../../../../.gitbook/assets/output_20_0.png)

## Save Model Weights And Configuration

```python
# save model architecture
model_json = model.to_json()
open('xception_model.json', 'w').write(model_json)

# save model's learned weights
model.save_weights('image_classifier_xception.h5', overwrite=True)
```

### Next Lesson

#### CGAN: Conditional Generative Adversarial Networks

![](../../images/divider.png)

