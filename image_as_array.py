def imageAsArray(imagePath):
    import numpy as np
    import cv2

    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(image)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32') / 255

    return np.reshape(image, (784, 1))

bin = imageAsArray('numbers/8_m.jpg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(bin.reshape((28, 28)), cmap=cm.Greys_r)
plt.show()