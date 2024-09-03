import numpy as np




class FCLayer:
    def __init__(self, in_features, out_features, activation:ActivationFunction):

        '''
            Fully connected layer with activation function
        
        '''

        #initilize weights and biases for the layer randomly
        
        limit = np.sqrt(6/float(in_features))
        self.weights = np.random.uniform(low=-limit, high=limit, size=(in_features, out_features))
        self.biases = np.ones((1, out_features))

        #set activation function
        self.activation = activation

    def forward(self, X, cache=True):

        '''
            Compute this layers output on X and cache values for backprop

            X = (batchSize, size) flattened input
        
        '''

        #X is of size (batchSize, inputSizes)
        #caching activation (w^T * x + b)
        if cache:
            self.input = X
            self.z = np.dot(X, self.weights) + self.biases
            self.a = self.activation.apply(self.z)

            #pass through activation function to get output
            return self.a
        else:
            return self.activation.apply(np.dot(X, self.weights) + self.biases)


    def backward(self, nextGrad, optimizer:Optimizer, timeStep:int):

        '''
            Compute the gradient of this layers weights, biases and inputs w.r.t to the loss function
            nextGrad : the gradient of the loss w.r.t this layers output
        '''
        batchSize = nextGrad.shape[0]

        dA = nextGrad

        #chain rule on the activation function gradient
        dZ = dA * self.activation.derivative(self.z)

        #get weight and bias gradient
        dW = np.dot(self.input.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        #gradient w.r.t input
        dX = np.dot(dZ, self.weights.T)

        dW /= batchSize
        db /= batchSize

        self.weights, self.biases = optimizer.step(timeStep, self.weights, self.biases, dW, db)
        return dX
    


class DeepNN:
    
    def __init__(self, height, width, depth, seed):
        
        self.weights = np.ones(height, width, depth)
        self.biases
        
    
        
        