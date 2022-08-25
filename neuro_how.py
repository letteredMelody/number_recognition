print ("Let's get a network object\n")
import network
net = network.Network([784, 30, 10])

print ("train the Network object using MNIST DATABASE:")
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net.SGD(training_data, 30, 10, 3.0, test_data = test_data)

# [screenshot 0]

print("\ncheck how trained network can recognize the digits")
print("let's see how the trained network processes test data\n")

raw_data, answer = test_data[0]

print(net.feedforward(raw_data))

# [screenshot 1]

print("\nto abtain the answer we should take the biggest number:")

import numpy as np
print(np.argmax(net.feedforward(raw_data)))

# [screenshot 2]

# let's looks at the test data network just recognized for us

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(raw_data.reshape((28, 28)), cmap=cm.Greys_r)
plt.show()

# [screenshot 3]
