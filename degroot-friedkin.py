#Written by Joel Wildman (22984156)

#DeGroot model is row stochastic weight matrix where each edge has a weight > 0 && < 1
#Friedkin model introduces self-confidence

import numpy as np


def generate_network(size):
	adj_matrix = np.zeros(size, size)
	w_matrix = np.zeros(size, size)
	return adj_matrix, w_matrix

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
def iterate_network(network, weights):
	new_network = np.zeros(network.shape[0], network.shape[1])
	return new_network

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
	new_weights = np.zeros(weights.shape[0], weights.shape[1])
	return new_weights

#runs DeGroot-Friedkin simulation for a given number of iterations and issues
def run_dgf(network_size, num_iterations, num_issues):
	network, weights = generate_network(network_size)

	for i in range(num_issues):
		for j in range(num_iterations):
			network = iterate_network(network, weights)
		weights = iterate_confidence(network, weights)

	return network, weights