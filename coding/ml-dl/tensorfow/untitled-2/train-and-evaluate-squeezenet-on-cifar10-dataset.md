# Train and Evaluate SqueezeNet on Cifar10 Dataset

## Imports

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

```python
datasets = tf.contrib.keras.datasets
layers =  tf.contrib.keras.layers
models = tf.contrib.keras.models
losses = tf.contrib.keras.losses
optimizers = tf.contrib.keras.optimizers 
metrics = tf.contrib.keras.metrics
preprocessing_image = tf.contrib.keras.preprocessing.image
utils = tf.contrib.keras.utils
callbacks = tf.contrib.keras.callbacks
```

## Load Cifar10 Datset

```python
(x_train, y_train), (x_test, y_test)= datasets.cifar10.load_data()
```

## Image Preprocessing and Augmentation

```python
train_datagen = preprocessing_image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

test_datagen = preprocessing_image.ImageDataGenerator(rescale=1./255)
```

## One Hot Encoding of Target Vector

```python
y_train = utils.to_categorical(y_train, num_classes=10)
y_test = utils.to_categorical(y_test, num_classes=10)
```

## Data Generator

```python
train_generator = train_datagen.flow(x=x_train, y=y_train, batch_size=32, shuffle=True)

test_generator = test_datagen.flow(x=x_test, y=y_test, batch_size=32, shuffle=True)
```

## SqueezeNet Model

```python
def fire_module(x, fire_id, squeeze=16, expand=64):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"
    s_id = 'fire' + str(fire_id) + '/'

    channel_axis = 3

    x = layers.Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = layers.Activation('relu', name=s_id + relu + sq1x1)(x)

    left = layers.Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = layers.Activation('relu', name=s_id + relu + exp1x1)(left)

    right = layers.Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = layers.Activation('relu', name=s_id + relu + exp3x3)(right)

    x = layers.concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x

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

## Compile Model

```python
def compile_model(model):

    # loss
    loss = losses.categorical_crossentropy

    # optimizer
    optimizer = optimizers.RMSprop(lr=0.0001)

    # metrics
    metric = [metrics.categorical_accuracy, metrics.top_k_categorical_accuracy]

    # compile model with loss, optimizer, and evaluation metrics
    model.compile(optimizer, loss, metric)

    return model
```

## Train Model on Cifar10

```python
sn = SqueezeNet()
sn = compile_model(sn)
```

```python
history = sn.fit_generator(
    train_generator,
    steps_per_epoch=400,
    epochs=10,
    validation_data=test_generator,
    validation_steps=200)
```

```text
Epoch 1/10
400/400 [==============================] - 59s - loss: 2.2349 - categorical_accuracy: 0.1488 - top_k_categorical_accuracy: 0.6128 - val_loss: 2.1264 - val_categorical_accuracy: 0.2295 - val_top_k_categorical_accuracy: 0.7531
Epoch 2/10
400/400 [==============================] - 54s - loss: 2.0378 - categorical_accuracy: 0.2335 - top_k_categorical_accuracy: 0.7801 - val_loss: 1.9289 - val_categorical_accuracy: 0.2762 - val_top_k_categorical_accuracy: 0.8327
Epoch 3/10
400/400 [==============================] - 57s - loss: 1.9405 - categorical_accuracy: 0.2548 - top_k_categorical_accuracy: 0.8183 - val_loss: 1.9037 - val_categorical_accuracy: 0.2917 - val_top_k_categorical_accuracy: 0.8344
Epoch 4/10
400/400 [==============================] - 54s - loss: 1.8825 - categorical_accuracy: 0.2755 - top_k_categorical_accuracy: 0.8338 - val_loss: 1.8343 - val_categorical_accuracy: 0.2914 - val_top_k_categorical_accuracy: 0.8531
Epoch 5/10
400/400 [==============================] - 55s - loss: 1.8404 - categorical_accuracy: 0.2909 - top_k_categorical_accuracy: 0.8469 - val_loss: 1.8556 - val_categorical_accuracy: 0.3044 - val_top_k_categorical_accuracy: 0.8396
Epoch 6/10
400/400 [==============================] - 67s - loss: 1.8109 - categorical_accuracy: 0.3133 - top_k_categorical_accuracy: 0.8521 - val_loss: 1.7528 - val_categorical_accuracy: 0.3272 - val_top_k_categorical_accuracy: 0.8736
Epoch 7/10
400/400 [==============================] - 55s - loss: 1.7910 - categorical_accuracy: 0.3205 - top_k_categorical_accuracy: 0.8558 - val_loss: 1.7415 - val_categorical_accuracy: 0.3412 - val_top_k_categorical_accuracy: 0.8720
Epoch 8/10
400/400 [==============================] - 55s - loss: 1.7703 - categorical_accuracy: 0.3304 - top_k_categorical_accuracy: 0.8655 - val_loss: 1.9308 - val_categorical_accuracy: 0.3233 - val_top_k_categorical_accuracy: 0.8170
Epoch 9/10
400/400 [==============================] - 55s - loss: 1.7438 - categorical_accuracy: 0.3446 - top_k_categorical_accuracy: 0.8690 - val_loss: 1.6670 - val_categorical_accuracy: 0.3596 - val_top_k_categorical_accuracy: 0.8911
Epoch 10/10
400/400 [==============================] - 61s - loss: 1.7154 - categorical_accuracy: 0.3483 - top_k_categorical_accuracy: 0.8788 - val_loss: 1.6629 - val_categorical_accuracy: 0.3811 - val_top_k_categorical_accuracy: 0.8912
```

## Plot Accuracy and Loss Over Time

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

![](../../../../.gitbook/assets/output_21_0.png)

## Save Model Configuration and Weights

```python
# save model architecture
model_json = sn.to_json()
open('squeeze_net_model.json', 'w').write(model_json)

# save model's learned weights
sn.save_weights('image_classifier_squeeze_net.h5', overwrite=True)
```

### Next Lesson

#### ResNet: Residual Learning for Image Recognition

![](../../images/divider.png)

