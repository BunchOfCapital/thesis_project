#Written by Joel Wildman (22984156)

#DeGroot model is row stochastic weight matrix where each edge has a weight > 0 && < 1
#Friedkin model introduces self-confidence

import numpy as np


def generate_network(size):
	adj_matrix = np.zeros(size, size)
	w_matrix = np.zeros(size, size)
	return adj_matrix, w_matrix

def iterate_network(network, weights):
	new_network = np.zeros(network.shape[0], network.shape[1])
	return new_network

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