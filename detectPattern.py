#!/usr/bin/python3
import numpy as np
import copy
import random

'''
Class for each layer in the network
'''
class Layer:

	'''
	Function to initialize class attributes
	- Weights are initialized to random values
	- Biases are initialized to zeros

	params:
		- inputLength: length of the input
		- numNeurons: number of neurons in the layer
	'''
	def __init__(self, inputLength, numNeurons):
		self.weights = 0.1 * np.random.randn(inputLength, numNeurons)
		self.biases = np.zeros((1, numNeurons))

	'''
	Function to propagate forward in the network

	params:
		inputData: matrix holding input data
	'''
	def forwardPropagation(self, inputData):
		self.input = inputData
		self.output = np.dot(self.input, self.weights)+self.biases

		return self.output

	'''
	Function to propagate backwards to adjust weights and biases at a given learning rate

	params:
		- outError: matrix of output error
		- learningRate: value for learning rate
	'''
	def backwardPropagation(self, outError, learningRate):
		inError = np.dot(outError, self.weights.T)
		weightsError = np.dot(self.input.T, outError)

		self.weights -= learningRate * weightsError
		self.biases -= learningRate * outError

		return inError


'''
Rectified Linear (ReLU) activation function

params:
	- Z: matrix to perform activation on
'''
def relu(Z):
	return np.maximum(0,Z)

'''
Rectified Linear (ReLU) function's derivative

params:
	- Z: matrix to find derivative of
'''
def reluDerivative(Z):
	dZ = copy.deepcopy(Z)
	dZ[Z <= 0] = 0
	dZ[Z > 0] = 1
	return dZ

'''
Sigmoid function to evaluate probablities for the output layer

params:
	- Z: matrix to perform activation on
'''
def sigmoid(Z):
	return 1/(1+np.exp(-Z))

'''
Sigmoid function's derivative

params:
	- Z: matrix to find derivative of
'''
def sigmoidDerivative(Z):
	return (np.exp(-Z))/np.power(1+np.exp(-Z), 2)


'''
Class for activation layer in the network
'''
class Activation:

	'''
	Function to initialize values in the class

	params:
		- activation: activation function to use for this layer
		- activationDerivative: derivative of the activation function
	'''
	def __init__(self, activation=relu, activationDerivative=reluDerivative):
		self.activation = activation
		self.activationDerivative = activationDerivative

	'''
	Function to propagate forward in the network

	params:
		inputData: matrix holding input data
	'''
	def forwardPropagation(self, inputData):
		self.input = inputData
		self.output = self.activation(self.input)
		return self.output

	'''
	Function to propagate backwards to dense layer

	params:
		- outError: matrix of output error
		- learningRate: value for learning rate 
						(Dummy param to match function in Layer class)
	'''
	def backwardPropagation(self, outError, learningRate):
		return self.activationDerivative(self.input) * outError


'''
Mean Squared Error function to find the prediction error

params:
	- pred: matrix holding predicted values
	- true: matrix holding actual values 
'''
def meanSquaredError(pred, true):
	return np.mean(np.power(pred - true, 2))

'''
Mean Squared Error function's derivative to help backward propagation

params:
	- pred: matrix holding predicted values
	- true: matrix holding actual values 
'''
def meanSquaredErrorDerivative(pred, true):
	return (2/true.size)*(pred-true)


'''
Class that consists of all layers and carries out training and testing of the network
'''
class Network:

	'''
	Function that intializes class attributes

	params:
		- loss: loss function to use in the network
		- lossDerivative: derivative of chosen loss function
	'''
	def __init__(self, loss=meanSquaredError, lossDerivative=meanSquaredErrorDerivative):
		self.layers = []
		self.loss = loss
		self.lossDerivative = lossDerivative

	'''
	Function to add layer in the network

	params:
		layer: Layer or Activation object to be added to the network
	'''
	def addLayer(self, layer):
		self.layers.append(layer)

	'''
	Function to train network at a given learning rate

	params:
		- xTrain: matrix holding training data
		- yTrain: matrix holding answers to training data
		- learningRate: value holding the learning rate of the network
	'''
	def train(self, xTrain, yTrain, epochs, learningRate):
		xTrain = np.array(xTrain)
		xTrain = xTrain.reshape(xTrain.shape[0], 1, 1024)

		for _ in range(epochs):
			for i in range(len(xTrain)):
				output = xTrain[i]
				for layer in self.layers:
					output = layer.forwardPropagation(output)
				error = self.lossDerivative(output, np.array(yTrain[i]))
				for layer in reversed(self.layers):
					error = layer.backwardPropagation(error, learningRate)


	'''
	Function to get probabilities from the trained network for a given dataset

	params:
		- xTest: dataset to compute predictions for
	'''
	def getPredictions(self, xTest):
		result = []
		for i in range(len(xTest)):
			output = xTest[i]
			for layer in self.layers:
				output = layer.forwardPropagation(output)
			result.append(output.tolist()[0])
		return result

	'''
	Function interpret computed probabilities and evaluate the network from a given dataset

	params:
		- xTest: matrix holding testing data
		- yTest: matrix holding answers to testing data
	'''
	def predict(self, xTest, yTest, printRes=False):
		results = self.getPredictions(xTest)
		correctCount = 0
		if printRes:
			print('Prediction		Actual')
		for i in range(len(yTest)):
			pred = results[i].index(max(results[i]))
			true = yTest[i].index(max(yTest[i]))
			
			if printRes:
				print(str(pred)+'		'+str(true))
			
			if pred == true:
				correctCount += 1
		print('Accuracy: '+str(correctCount)+'/'+str(len(yTest)) + ' ('+str(round(100*(correctCount/len(yTest)), 2))+' %)')
		return correctCount


'''
Function to create input and target encoded files

params:
	origFilePath: original dataset file path
	inputFilePath: input file path
	targetFilePath: target file path
'''
def encodeFiles(origFilePath, inputFilePath, targetFilePath):
	with open(origFilePath) as file:
		lines = file.readlines()
		num = 21+32
		i = 21
		inputFile = open(inputFilePath, 'w')
		targetFile = open(targetFilePath, 'w')
		for line in lines[21:]:
			line = line.replace('\n','')
			if i == num:
				num += 33
				inputFile.write('\n')
				out = ['0' for _ in range(10)]
				out[int(line.split()[0])] = '1'
				out = "".join(out)
				targetFile.write(out+'\n')
			else:
				inputFile.write(line)
			i+=1
		inputFile.close()
		targetFile.close()


'''
Function to read input and target files to a list

params:
	inputFilePath: input file path
	targetFilePath: target file path
'''
def readFiles(inputFilePath, targetFilePath):
	img = []
	actual = []

	with open(inputFilePath) as inputFile:
		lines = inputFile.readlines()
		for line in lines:
			line = line.replace("\n", "")
			pixels = []
			for val in line:
				pixels.append(int(val))
			img.append(pixels)

	with open(targetFilePath) as targetFile:
		lines = targetFile.readlines()
		for line in lines:
			line = line.replace("\n", "")
			actual.append(list(map(int, list(line))))

	return img, actual


def main():
	# create input and target files
	encodeFiles("optdigits-orig.windep", "input.txt", "target.txt")

	# read input and target files
	img, actual = readFiles("input.txt", "target.txt")

	# create training set
	xTrain = img[:1300]
	yTrain = actual[:1300]

	# create validation set
	xValidate = img[1300:1500]
	yValidate = actual[1300:1500]

	# create testing set
	xTest = img[1500:]
	yTest = actual[1500:]

	# the code below was used to validate models with different hyperparameters
	bestModel = (None,0,0,0)
	for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
		for e in [1, 3, 5]:
			print("Learning Rate: "+str(lr))
			print("Epochs: "+str(e))

			# create neural network
			nn = Network()

			# add input layer
			nn.addLayer(Layer(len(xTrain[0]), 500))

			# add hidden layers
			nn.addLayer(Activation())
			nn.addLayer(Layer(500, 500))
			nn.addLayer(Activation())
			#nn.addLayer(Activation(activation=sigmoid, activationDerivative=sigmoidDerivative))
			nn.addLayer(Layer(500, len(yTrain[0])))
			
			# add output layer
			nn.addLayer(Activation(activation=sigmoid, activationDerivative=sigmoidDerivative))

			# train neural network
			nn.train(xTrain, yTrain, epochs=e, learningRate=0.05)

			# validate model
			correctCount = nn.predict(xValidate, yValidate)
			print("**********************************************")

			if bestModel[1] < correctCount:
				bestModel = (copy.deepcopy(nn), correctCount, lr, e)
	
	print("Best Model:")
	print("Learning Rate: "+str(bestModel[2]))
	print("Epochs: "+str(bestModel[3]))
	print("Validation Accuracy: "+str(bestModel[1])+"/"+str(len(xValidate))+"("+str(round(100*(bestModel[1]/len(xValidate)), 2))+" %)")
	

	# test and evaluate network
	print("\n----Testing----")
	bestModel[0].predict(xTest, yTest, printRes=True)

if __name__ == "__main__":
	main()