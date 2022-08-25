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

import image_as_array

while True:
    a = raw_input("What pic do you want?")
    bin = image_as_array.imageAsArray('numbers/' + a)

    print(net.feedforward(bin))

    # [screenshot 1]

    print("Your answer is:")

    import numpy as np
    print(np.argmax(net.feedforward(bin)))

    # [screenshot 2]

    # let's looks at the picture network just recognized for us

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    plt.imshow(bin.reshape((28, 28)), cmap=cm.Greys_r)
    plt.show()

    # [screenshot 3]
#to stop the program, ctrl + c