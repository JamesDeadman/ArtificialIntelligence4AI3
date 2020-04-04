import numpy as np
import matplotlib.pyplot as plt

#Number of iterations to train the neural net
TRAINING_ITERATIONS = 400

#Class for any Neural Network with a single hidden layer
class NeuralNetwork:
    def __init__(self, input, desiredOutput):
        self.input = input
        self.desiredOutput = desiredOutput

        self.weights1 = np.random.rand(self.input.shape[1], len(input))
        self.weights2 = np.random.rand(len(input), 1) 
        self.output = np.zeros(self.desiredOutput.shape)

    def FeedForward(self):
        self.hiddenLayer = NeuralNetwork.Sigmoid(np.dot(self.input, self.weights1))
        self.output = NeuralNetwork.Sigmoid(np.dot(self.hiddenLayer, self.weights2))

    def BackProp(self):
        dWeights2 = np.dot(self.hiddenLayer.T, (2 * (self.output - self.desiredOutput) * NeuralNetwork.SigmoidDerivative(self.output)))
        dWeights1 = np.dot(self.input.T, (np.dot(2 * (self.output - self.desiredOutput) * NeuralNetwork.SigmoidDerivative(self.output), self.weights2.T) * NeuralNetwork.SigmoidDerivative(self.hiddenLayer)))

        self.weights2 = self.weights2 - dWeights2
        self.weights1 = self.weights1 - dWeights1

    def Train(self, iterations):
        self.lossValues = []

        for i in range(iterations):
            FlameDetectionNeuralNet.FeedForward()
            FlameDetectionNeuralNet.BackProp()
            loss = NeuralNetwork.ComputeLoss(FlameDetectionNeuralNet.output, valveOutput)
            self.lossValues.append(loss)

        self.finalLoss = loss

    def Report(self):
        print("Weights, Input -> Hidden Layer")
        print(self.weights1)
        
        print("Weights, Hidden Layer -> Output")
        print(self.weights2)

        print("Output")
        print(self.output)
    
        print("FinalLoss:")
        print(self.finalLoss)

    def Plot(self):
        plt.plot(self.lossValues)
        plt.show()

    @staticmethod
    def Sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def SigmoidDerivative(sigma):
        return sigma * (1.0 - sigma)

    @staticmethod
    def ComputeLoss(actual, desired):
        return ((desired - actual) ** 2).sum()


# Training data input values
sensorInputs = np.array([[0,0,0],
                         [0,0,1],
                         [0,1,0],
                         [0,1,1],
                         [1,0,0],
                         [1,0,1],
                         [1,1,0],
                         [1,1,1]])

# Training data expected results
valveOutput = np.array([[0], [0], [0], [1], [0], [1], [1], [1]])

# Create and train the Neual Net
FlameDetectionNeuralNet = NeuralNetwork(sensorInputs, valveOutput)
FlameDetectionNeuralNet.Train(TRAINING_ITERATIONS)

# Display the results
FlameDetectionNeuralNet.Report()
FlameDetectionNeuralNet.Plot()