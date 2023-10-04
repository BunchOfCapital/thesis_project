#Written by Joel Wildman (22984156)


#algorithm as specified in https://epubs.siam.org/doi/10.1137/130913250

#DeGroot model is row stochastic weight matrix where each edge has a weight > 0 && < 1
#Friedkin model introduces self-confidence


import numpy as np
import sys
from matplotlib import pyplot as plt
import numpy.ma as ma


def parse_file(filename,):
	values = np.loadtxt(filename, dtype=int)
	# new_values = []
	# for i in range(values.shape[0]):
	# 	if (np.random.rand() < 0.65 ):
	# 		new_values.append(values[i])
	
	return values

#this dataset has 4039 nodes
def gen_facebook_net(size=4039):
	edges = parse_file("facebook_combined.txt")
	network = np.ones((size,size), dtype=int)
	print(edges.shape)
	for i in range(edges.shape[0]):
		#this graph is undirected
		network[edges[i][0]][edges[i][1]] = 0
		network[edges[i][1]][edges[i][0]] = 0
	#add self loops to represent self-weight
	for i in range(size):
		network[i][i] = 0
	return network


def generate_network(size, facebook=False):
	

	if (facebook):
		#facebook graph:
		adj_matrix = gen_facebook_net()
		w_matrix = np.random.random((size, size))
		masked_w = ma.masked_array(w_matrix, adj_matrix)

		#make it row-stochastic 
		masked_w = masked_w/masked_w.sum(axis=1)[:,None]

		#wipe the remaining weights for clarity
		masked_w[masked_w.mask] = 0

		final_weights = np.array(masked_w)

	else:
		#complete graph:
		adj_matrix = np.zeros((size, size))
		for i in range(size):
			adj_matrix[i][i] = 0 #no self loops

		#generate weight matrix
		w_matrix = np.random.random((size, size))

		#make it row-stochastic
		w_matrix = w_matrix/w_matrix.sum(axis=1)[:,None]

		final_weights = w_matrix



	

	#opinions for each node are between 0 and 1
	opinions = np.random.random(size)


	return adj_matrix, final_weights, opinions

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

in short, matrix c is row stochastic, with a 0 diagonal

"""
def gen_interpersonal_weights(network, weights):
	interpersonal_weights = np.zeros((weights.shape[0], weights.shape[1]))
	for i in range(weights.shape[0]):
		for j in range(weights.shape[1]):
			if (network[i][j] == 0 and i != j):
				interpersonal_weight = weights[i][j] / (1 - weights[i][i])
				interpersonal_weights[i][j] = interpersonal_weight

	#make matrix row stochastic
	masked_w = ma.masked_array(interpersonal_weights, network)
	for i in range(weights.shape[0]):
		row_sum = masked_w[i].sum()
		if (row_sum == 0.0):
			print("no connections?")
			exit()
		adjustment = 1 / row_sum
		interpersonal_weights[i] = interpersonal_weights[i] * adjustment

	return interpersonal_weights




def iterate_network(network, weights, opinions):
	new_opinions = np.zeros(network.shape[0])
	interpersonal_weights = gen_interpersonal_weights(network, weights)

	#for each node in the network
	for i in range(network.shape[0]):
		#self confidence * current opinion
		current_standing = weights[i][i] * opinions[i]

		#find sum of adjacent nodes' opinions
		opinion_sum = 0
		#print("stochastic?: ", weights[i].sum())
		for j in range(network.shape[1]):
			if (network[i][j] == 0):
				opinion_sum += interpersonal_weights[i][j] * opinions[j]

		#perform final calculation
		new_opinion = current_standing + (opinion_sum * (1 - weights[i][i]))
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

	interpersonal_weights = gen_interpersonal_weights(network, weights)
	#assume timestep T = 1 because it doesn't work if it isnt'
	for i in range(network.shape[0]):
		#with T=1 p(s, T) becomes w[i][i]
		prior_opinion = weights[i][i] * weights[i][i]

		#sum incoming weights
		influence_sum = 0
		for j in range(network.shape[1]):
			if (network[j][i] == 0):
				influence_sum += interpersonal_weights[j][i] * (1-weights[j][j]) * weights[j][j] 
		
		#note that this value can be greater than 1, which I don't think is meant to happen
		new_weights[i][i] = prior_opinion + influence_sum


	#now scale the rest of the weights matrix to ensure it remains row stochastic
	for i in range(network.shape[0]):

		masked_w = ma.masked_array(new_weights, network)
		row_sum = masked_w[i].sum()

		#we don't want to adjust the self weights, so we have to subtract it's adjustment from the other weights
		#adjustment = (1 - new_weights[i][i]) / (row_sum - new_weights[i][i])

		#nvm yes we do
		adjustment = 1 / row_sum


		if (adjustment < 0):
			print("Catastrophic calculation error")
			print(adjustment)
			exit()

		for j in range(network.shape[1]):
			if (network[i][j] == 0):
				new_weights[i][j] = new_weights[i][j] * adjustment
		#print("stochastic?: ", np.sum(new_weights[i]))

	return new_weights

#runs DeGroot-Friedkin simulation for a given number of iterations and issues
def run_dgf(network_size, num_iterations, num_issues, facebook=False):
	network, weights, opinions = generate_network(network_size, facebook=facebook)

	self_weights = np.zeros((num_issues, network_size), dtype=float)

	print("Initial Opinion Vector: \n", opinions)
	print("Initial Weights Vector: \n", weights)
	plotted = False
	for i in range(num_issues):
		print("Beginning Issue ", i)
		converged = False
		opinion_record = np.zeros((num_iterations, network_size), dtype=float)
		for j in range(num_iterations):
			print("beginning iteration ", j)
			opinions = iterate_network(network, weights, opinions)
			opinion_record[j] = opinions[:]
			#print(opinions)
			if (not converged and len(set(np.around(opinions, decimals=10))) == 1):
				print("Opinion Convergence achieved on iteration ", j, " of issue ", i)
				converged = True

		if not plotted:
			plt.plot(opinion_record, linestyle='dashed')
			plt.title("Random sample of agent opinions ("+str(network_size)+" nodes)")
			plt.savefig(fname="opinions_f"+str(facebook)+".pdf", format='pdf')
			plt.show()
			plotted=True

		weights = iterate_confidence(network, weights)
		print(weights)
		for p in range(network_size):
			self_weights[i][p] = weights[p][p]

	print("Final Opinion Vector: \n", opinions)
	print("Final Weights Vector: \n", weights)
	print(self_weights.shape)
	sample = np.random.randint(low=0, high=network_size-1, size=30)
	plt.plot(self_weights[:,sample], linestyle='dashed')
	plt.title("Random sample of agent self-weights (" + str(network_size) + " nodes, " + str(num_iterations) + " iterations per issue)")
	plt.savefig(fname="self_weights_f"+str(facebook)+"_2.pdf", format='pdf')
	plt.show()

	return network, weights, opinions


if __name__ == "__main__":
	if (len(sys.argv) < 4):
		print("Please enter network size, number of iterations and number of issues")
		exit()
	else:
		if ("-f" in sys.argv):
			_, _, opinions = run_dgf(4039, int(sys.argv[2]), int(sys.argv[3]), facebook=True)
		else: 
			_, _, opinions = run_dgf(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
