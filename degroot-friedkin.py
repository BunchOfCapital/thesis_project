#Written by Joel Wildman (22984156)

#DeGroot model is row stochastic weight matrix where each edge has a weight > 0 && < 1
#Friedkin model introduces self-confidence

import numpy as np


def generate_network(size):
	#adj_matrix = np.zeros(size, size)
	adj_matrix = np.ones((size, size))
	for i in range(size):
		adj_matrix[i][i] = 0 #no self loops

	#generate row-stochastic weight matrix
	w_matrix = np.random.random((size, size))
	w_matrix = w_matrix/w_matrix.sum(axis=1)[:,None]
	#print("um?")
	#print(w_matrix)

	#opinions for each node are between 0 and 1
	opinions = np.random.random(size)

	return adj_matrix, w_matrix, opinions

#algorithm for iterating within a single issue:
"""
from https://ieeexplore.ieee.org/document/7170871

yi(s,t+1)=wii(s)yi(s,t)+∑j=1j≠inwij(s)yj(s,t) where
yi is opinion of node i
wii is self confidence
wij is weight node i gives to node j

note that the weight matrix is stochastic so 
wij(s)=(1−xi(s))cij
where xi is self confidence
cij is neighbours?

"""
def iterate_network(network, weights, opinions):
	new_opinions = np.zeros(network.shape[0])

	#for each node in the network
	for i in range(network.shape[0]):
		#self weight * current opinion
		current_standing = weights[i][i] * opinions[i]

		#find sum of adjacent nodes' opinions, adjusted to remain stochastic
		opinion_sum = 0
		for j in range(network.shape[1]):
			#I have no idea why this is necessary or what its use is
			if (weights[i][j] == 0):
				print("fuck")
				print(weights)
			interpersonal_weight = (1 - weights[i][i]) / weights[i][j]
			print("interpersonal_weight = ", interpersonal_weight)
			#print("interpersonal_weight is: ", interpersonal_weight)
			#Relative interaction matrix C is 0 along main diagonal
			if (i == j):
				interpersonal_weight = 0
			print("(1-weights[i][i]) = ", (1-weights[i][i]))
			opinion_sum += weights[i][j] * opinions[j]
			print("opinion sum = ", opinion_sum)

		#perform final calculation
		new_opinion = current_standing + (opinion_sum - current_standing)/2
		new_opinions[i] = new_opinion

	return new_opinions

#algorithm for iterating once issue has concluded:
"""
original:
pi(s,t+1)=wii(s)pi(s,t)+∑j=1,j≠inwji(s)pj(s,t) where
pi is perceived social power
wii is self confidence
wji is other nodes weighting of current node

adjusted for finite time steps:
xi(s+1)=xi(s)xi(s)+∑j=1n(1−xj(s))cjixj(s)
xi(s+1) is self weight and == pi(s, T) for T time steps
cji is interpersonal weights from neighbours


"""
def iterate_confidence(network, weights):
	new_weights = np.zeros((weights.shape[0], weights.shape[1]))
	return weights

#runs DeGroot-Friedkin simulation for a given number of iterations and issues
def run_dgf(network_size, num_iterations, num_issues):
	network, weights, opinions = generate_network(network_size)

	for i in range(num_issues):
		for j in range(num_iterations):
			opinions = iterate_network(network, weights, opinions)
			print(opinions)
		weights = iterate_confidence(network, weights)

	return network, weights, opinions

_, _, opinions = run_dgf(10, 100, 1)
#print(opinions)