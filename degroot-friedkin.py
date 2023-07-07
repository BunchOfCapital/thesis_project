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
def gen_interpersonal_weights(weights):
	interpersonal_weights = np.zeros((weights.shape[0], weights.shape[1]))

	for i in range(weights.shape[0]):
		for j in range(weights.shape[1]):
			interpersonal_weight = weights[i][j] / (1 - weights[i][i])
			if (i == j):
				continue
			else:
				interpersonal_weights[i][j] = interpersonal_weight

	#make matrix stochastic
	for i in range(weights.shape[0]):
		row_sum = np.sum(interpersonal_weights[i])
		adjustment = 1 / row_sum
		interpersonal_weights[i] = interpersonal_weights[i] * adjustment

	return interpersonal_weights




def iterate_network(network, weights, opinions):
	new_opinions = np.zeros(network.shape[0])
	interpersonal_weights = gen_interpersonal_weights(weights)

	#for each node in the network
	for i in range(network.shape[0]):
		#self confidence * current opinion
		current_standing = weights[i][i] * opinions[i]

		#find sum of adjacent nodes' opinions
		opinion_sum = 0
		#print("stochastic?: ", weights[i].sum())
		for j in range(network.shape[1]):
			opinion_sum += interpersonal_weights[i][j] * opinions[j]

		#perform final calculation
		new_opinion = current_standing + opinion_sum * (1 - weights[i][i])
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
	new_weights = weights.copy()

	#for complete system with full knowledge:
		#calculate left eigenvector of the weight matrix 
		#eigvals, eigvecs = np.linalg.eig(weights.T)
		#left_vec = eigvecs[:, 0].T
		#update weights to reflect this

		#adjust to remain stochastic



	#for finite system:

	interpersonal_weights = gen_interpersonal_weights(weights)
	#assume timestep T = 1 because it doesn't work if it isnt'
	for i in range(network.shape[0]):
		#with T=1 p(s, T) becomes w[i][i]
		prior_opinion = weights[i][i] * weights[i][i]

		#sum incoming weights
		influence_sum = 0
		for j in range(network.shape[1]):
			influence_sum += interpersonal_weights[j][i] * (1-weights[j][j]) * weights[j][j]

		new_weights[i][i] = prior_opinion + influence_sum


	#now scale the rest of the weights matrix to ensure it remains row stochastic
	for i in range(network.shape[0]):

		row_sum = np.sum(new_weights[i])
		difference = 1 - row_sum

		#we don't want to adjust the self weights, so we have to subtract it's adjustment from the other weights
		adjustment = difference / (network.shape[0] - 1)
		adjustment = (1 - new_weights[i][i]) / (row_sum - new_weights[i][i])

		if (adjustment < 0):
			print("Catastrophic calculation error")
			exit()
			break

		for j in range(network.shape[1]):
			if (i == j):
				continue
			else:
				new_weights[i][j] = new_weights[i][j] * adjustment
		#print("stochastic?: ", np.sum(new_weights[i]))
	print(new_weights)

	return new_weights

#runs DeGroot-Friedkin simulation for a given number of iterations and issues
def run_dgf(network_size, num_iterations, num_issues):
	network, weights, opinions = generate_network(network_size)

	for i in range(num_issues):
		for j in range(num_iterations):
			opinions = iterate_network(network, weights, opinions)
			#print(opinions)
		weights = iterate_confidence(network, weights)

	return network, weights, opinions

_, _, opinions = run_dgf(10, 10000, 10)
#print(opinions)
