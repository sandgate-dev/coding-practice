# Training and Evaluating ResNet Model

## Import

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
```

```python
models = tf.contrib.keras.models
layers = tf.contrib.keras.layers
initializers = tf.contrib.keras.initializers
regularizers = tf.contrib.keras.regularizers
losses = tf.contrib.keras.losses
optimizers = tf.contrib.keras.optimizers 
metrics = tf.contrib.keras.metrics
preprocessing_image = tf.contrib.keras.preprocessing.image
```

## ResNet Model

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

```python
def ResNetPreAct(input_shape=(32,32,3), nb_classes=5, num_stages=5,
                 use_final_conv=False, reg=0.0):


    # Input
    img_input = layers.Input(shape=input_shape)

    #### Input stream ####
    # conv-BN-relu-pool
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
    # FxF conv: batchnorm-relu-conv
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

## Compile Model

```python
def compile_model(model):

    # loss
    loss = losses.categorical_crossentropy

    # optimizer
    optimizer = optimizers.Adam(lr=0.0001)

    # metrics
    metric = [metrics.categorical_accuracy, metrics.top_k_categorical_accuracy]

    # compile model with loss, optimizer, and evaluation metrics
    model.compile(optimizer, loss, metric)

    return model
```

## Image Preprocessing And Augmentation

```python
train_datagen = preprocessing_image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

test_datagen = preprocessing_image.ImageDataGenerator(rescale=1./255)
```

```python
BASE_DIR = "/Users/marvinbertin/Desktop/tmp"

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "flower_dataset/train"),
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "flower_dataset/validation"),
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical')
```

```text
Found 2939 images belonging to 5 classes.
Found 731 images belonging to 5 classes.
```

```python
model = ResNetPreAct(input_shape=(32, 32, 3), nb_classes=5, num_stages=5,
                     use_final_conv=False, reg=0.005)

model = compile_model(model)
```

## Train Model on Flower Dataset

```python
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=20)
```

```text
Epoch 1/10
100/100 [==============================] - 97s - loss: 4.7355 - categorical_accuracy: 0.3716 - top_k_categorical_accuracy: 1.0000 - val_loss: 4.7847 - val_categorical_accuracy: 0.2453 - val_top_k_categorical_accuracy: 1.0000
Epoch 2/10
100/100 [==============================] - 94s - loss: 4.2881 - categorical_accuracy: 0.5327 - top_k_categorical_accuracy: 1.0000 - val_loss: 4.6325 - val_categorical_accuracy: 0.2472 - val_top_k_categorical_accuracy: 1.0000
Epoch 3/10
100/100 [==============================] - 92s - loss: 3.9952 - categorical_accuracy: 0.5626 - top_k_categorical_accuracy: 1.0000 - val_loss: 4.5625 - val_categorical_accuracy: 0.2441 - val_top_k_categorical_accuracy: 1.0000
Epoch 4/10
100/100 [==============================] - 93s - loss: 3.7385 - categorical_accuracy: 0.5889 - top_k_categorical_accuracy: 1.0000 - val_loss: 4.4722 - val_categorical_accuracy: 0.2315 - val_top_k_categorical_accuracy: 1.0000
Epoch 5/10
100/100 [==============================] - 99s - loss: 3.5150 - categorical_accuracy: 0.6122 - top_k_categorical_accuracy: 1.0000 - val_loss: 4.0694 - val_categorical_accuracy: 0.3118 - val_top_k_categorical_accuracy: 1.0000
Epoch 6/10
100/100 [==============================] - 113s - loss: 3.3142 - categorical_accuracy: 0.6223 - top_k_categorical_accuracy: 1.0000 - val_loss: 3.5542 - val_categorical_accuracy: 0.4142 - val_top_k_categorical_accuracy: 1.0000
Epoch 7/10
100/100 [==============================] - 90s - loss: 3.1440 - categorical_accuracy: 0.6408 - top_k_categorical_accuracy: 1.0000 - val_loss: 3.2635 - val_categorical_accuracy: 0.5354 - val_top_k_categorical_accuracy: 1.0000
Epoch 8/10
100/100 [==============================] - 87s - loss: 2.9911 - categorical_accuracy: 0.6518 - top_k_categorical_accuracy: 1.0000 - val_loss: 2.9651 - val_categorical_accuracy: 0.6409 - val_top_k_categorical_accuracy: 1.0000
Epoch 9/10
100/100 [==============================] - 100s - loss: 2.8415 - categorical_accuracy: 0.6620 - top_k_categorical_accuracy: 1.0000 - val_loss: 2.8743 - val_categorical_accuracy: 0.6079 - val_top_k_categorical_accuracy: 1.0000
Epoch 10/10
100/100 [==============================] - 119s - loss: 2.7194 - categorical_accuracy: 0.6677 - top_k_categorical_accuracy: 1.0000 - val_loss: 2.7313 - val_categorical_accuracy: 0.6646 - val_top_k_categorical_accuracy: 1.0000
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

![](../../../../.gitbook/assets/output_17_0.png)

## Save Model Weights And Configuration

```python
# save model architecture
model_json = model.to_json()
open('resnet_model.json', 'w').write(model_json)

# save model's learned weights
model.save_weights('image_classifier_resnet.h5', overwrite=True)
```

### Next Lesson

#### Xception: Depthwise Separable Convolutions

![](../../images/divider.png)

