# Loading and Exploring MNIST Dataset

## Image Classification Task

MNIST is a classic computer-vision dataset used for handwritten digits recognition. The dataset consists of:

* black and white images of handwritten digits
* 10 classes
* 28x28 pixels
* 7 000 images per classes
* 60 000 images in the training set
* 10 000 images in the test set

![](../../../../.gitbook/assets/mnist.png)

![](../../images/mnist.png)

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
```

## Download and Load MNIST Dataset

```python
(x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
```

## Training Tensor Shape

```python
x_train.shape
```

```text
(60000, 28, 28)
```

## Testing Tensor Shape

```python
x_test.shape
```

```text
(10000, 28, 28)
```

## Ploting Helper Function

```python
def plot_10_by_10_images(images):

    # figure size
    fig = plt.figure(figsize=(10,10))

    # plot image grid
    for x in range(10):
        for y in range(10):
            ax = fig.add_subplot(10, 10, 10*y+x+1)
            plt.imshow(images[10*y+x], cmap='Greys')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()
```

## Explore MNIST Dataset

```python
plot_10_by_10_images(x_train[:100])
```

### Next Lesson

#### ACGAN Architecture

* Discriminator
* Generator
* Label Conditioning

