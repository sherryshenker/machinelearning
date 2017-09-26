'''
Sherry Shenker
Problem Set 6
March 9th, 2017

Syntax to run code:
python3 sshenker_pset6.py [trainx_filename] [trainy_filename]
[testx_filename] [testy_filename] eta nhidden num_epochs [predictx_filename]
'''

import sys
import random
from math import exp
import numpy as np
import matplotlib.pyplot as plt

class NN:
	def __init__(self,trainx,trainy,testx,testy,nhidden,num_epochs,eta,predictx,h_x,h_y):
		self.eta = float(eta)
		self.num_epochs = int(num_epochs)
		self.nhidden = int(nhidden)
		self.trainx = trainx
		self.trainy = trainy
		self.testx = testx
		self.testy = testy
		self.holdout_x = h_x
		self.holdout_y = h_y
		self.weights = self.initialize_weights(784,10)
		self.deltas = [[0 for i in range(self.nhidden+1)],[0 for i in range(10)]]
		self.outputs = [[0 for i in range(self.nhidden+1)],[0 for i in range(10)]]
		j = []
		with open(predictx,'r') as f:
			for line in f:
				p = line.split(",")
				l = [float(i) for i in p]
				l.append(1) #bias neuron
				j.append(l)
		self.predictx = np.asarray(j)


	def initialize_weights(self,num_inputs,num_output):
		'''
		initialize weights to random values between 0 and 1
		in the hidden layer, there is nhidden lists with 785 weights in each list
		in the output later, there are 10 lists with nhidden+1 weights in each list
		the last weight is the weight of the bias
		'''
		hidden_layer = []
		for n in range(self.nhidden+1):
			weights = []
			for i in range(num_inputs+1):
				weights.append(random.uniform(-1,1))
			hidden_layer.append(weights)
		output_layer = []
		for n in range(num_output):
			weights = []
			for i in range(self.nhidden+1):
				weights.append(random.uniform(-1,1))
			output_layer.append(weights)

		return [np.asarray(hidden_layer),np.asarray(output_layer)]

	def forward_propagate(self,row):
		'''
		iterate through layers of network to calculate output.
		output of one layer is input of the next layer
		'''
		inputs = row
		for layer_index in range(len(self.weights)):
			outputs = []
			for output_index in range(len(self.weights[layer_index])):
				activation = self.activate(inputs,output_index,layer_index)
				output = self.transfer(activation)
				outputs.append(output)
			self.outputs[layer_index] = outputs
			inputs = outputs
		return outputs

	def activate(self,inputs,output_index,layer_index):
		'''
		activate all neurons in a layer
		'''
		return np.dot(self.weights[layer_index][output_index,:],inputs)

	def transfer(self,activation):
		return 1.0 / (1.0 + exp(-activation))

	def train_network(self):
		for e in range(self.num_epochs):
			print("epoch: {}".format(e))
			error = 0
			for index,img in enumerate(self.trainx):
				outputs = self.forward_propagate(img)
				correct = int(self.trainy[index][0])
				expected = [0 for i in range(10)]
				expected[correct] = 1
				error += np.sum(np.linalg.norm([expected,outputs],axis=0))
				self.back_prop(expected)
				self.update_weights(img)
			holdout_error = self.test_h()[1]/len(self.holdout_y)
			print("holdout error: {}".format(holdout_error))
		predictions,test_error =  self.test()
		print("num epochs: {} error: {}".format(self.num_epochs,test_error))
		return test_error


	def back_prop(self,expected):
		'''
		back propagation to calculate delta values and then update weights
		'''
		errors = np.subtract(self.outputs[1],expected)
		deltas = np.multiply(errors,self.transfer_derivative(self.outputs[1]))
		self.deltas[1] = deltas

		#back propagate to hidden layer
		hidden_deltas = []
		for neuron in range(self.nhidden+1):
			error = np.dot(self.weights[1][:,neuron],deltas)
			hidden_deltas.append(error)
		self.deltas[0] = np.multiply(hidden_deltas,self.transfer_derivative(self.outputs[0]))
	def transfer_derivative(self,output):
		j = [x*(1-x) for x in output ]
		return j

	def update_weights(self,img):
		'''
		update weights using gradient descent
		'''

		for i in range(self.nhidden+1):
			self.weights[0][i,:] = np.subtract(self.weights[0][i,:],self.eta*self.deltas[0][i]*img)

		for output_index in range(10):
			self.weights[1][output_index,:] = np.subtract(self.weights[1][output_index,:],self.eta*self.deltas[1][output_index]*np.array(self.outputs[0]))


	def test_h(self):
		'''
		use forward propagation to predict on the holdout set
		and calculate error
		'''
		predictions = []
		error = 0
		for row_index,row in enumerate(self.holdout_x):
			probs = self.forward_propagate(row)
			p = np.argmax(probs)
			predictions.append(p)
			if p != self.holdout_y[row_index]:
				error += 1

		return predictions, error

	def test(self):
		'''
		use forward propagation to predict on the holdout set
		and calculate error
		'''
		predictions = []
		error = 0
		for row_index,row in enumerate(self.testx):

			probs = self.forward_propagate(row)
			p = np.argmax(probs)
			predictions.append(p)
			if p != self.testy[row_index]:
				error += 1

		return predictions, error

	def predict(self):
		'''
		use forward propagation to predict on the test set
		'''
		predictions = []
		for row in enumerate(self.predictx):
			probs = self.forward_propagate(row)
			p = np.argmax(probs)
			predictions.append(p)
		return predictions


def train_net(trainx,trainy,testx,testy,eta,nhidden,num_epochs,predictx):
	'''
	wrapper function for training the network and predicting on the test set
	'''
	tr_x,tr_y,h_x,h_y = read_data(trainx,trainy,"train")
	test_x,test_y = read_data(testx,testy,"test")



	''' code for finding optimal parameters
	errors = {}
	for n in np.arange(4,7,1):
		for e in np.arange(0.1,0.4,0.05):
			for h in [32,64,128,256]:
				net = NN(tr_x,tr_y,test_x,test_y,h,n,e,predictx,h_x,h_y)
				error = net.train_network()
				errors["{}_{}_{}".format(n,h,e)] = error'''


	net = NN(tr_x,tr_y,test_x,test_y,nhidden,num_epochs,eta,predictx,h_x,h_y)
	error = net.train_network()

	print("test set error: {}".format(error))



def read_data(x,y,t):
	'''
	converts file containing digit pixels and file containing digit labels
	into two numpy arrays
	'''
	with open(x,'r') as f:
		data_list = []
		for line in f:
			l = line.split(",")
			l = [float(i) for i in l]
			l.append(1) #bias neuron
			data_list.append(l)
	with open(y,'r') as f:
		labels = []
		for line in f:
			l = line.split(",")
			l = [float(i) for i in l]
			labels.append(l)
	if t == "train":
		n = int(len(data_list))
		i = int(n*0.8)

		#split data into training and holdout set

		indexes = random.sample(range(n),i)
		train_x = np.take(np.asarray(data_list),indexes,axis=0)
		train_y = np.take(np.asarray(labels),indexes,axis=0)
		all_i = set(range(n))
		holdout = list(all_i.difference(indexes))
		h_x = np.take(np.asarray(data_list),holdout,axis=0)
		h_y = np.take(np.asarray(labels),holdout,axis=0)

		return train_x,train_y,h_x,h_y
	else:
		return np.asarray(data_list),np.asarray(labels)

if __name__ == '__main__':
	train_net(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8])
