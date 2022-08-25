def trainedNetwork():

    # Network should be trained

    ## Let's get a network object
    import network
    net = network.Network([784, 30, 10])

    ## train the Network object using MNIST DATABASE
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    return net
