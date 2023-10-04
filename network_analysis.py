import numpy as np
from matplotlib import pyplot as plt
import math

#TODO: degree distribution
#TODO: graphs
#TODO: index-based analysis


def main(file_name, num_nodes):
	input_data = np.loadtxt(file_name, dtype=int)

	new_values = []
	for i in range(input_data.shape[0]):
		if (np.random.rand() < 0.3 ):
			new_values.append(input_data[i])
	
	data =  np.array(new_values)
	
	degrees = np.zeros(num_nodes)
	for edge in data:
		degrees[int(edge[0])] += 1
		degrees[int(edge[1])] += 1

	#easy stats
	print("Average degree is ", np.mean(degrees))
	print("Standard Degree Deviation is ", np.std(degrees))
	print("Median Degree Value is ", np.median(degrees))
	print("Highest Degree Value is ", np.max(degrees), " of node ", np.argmax(degrees))
	print("Lowest Degree Value is ", np.min(degrees))

	#cliquishness
	#first make adjacency matrix
	adj_mat = np.zeros((num_nodes,num_nodes))
	for edge in data:
		adj_mat[int(edge[0])][int(edge[1])] = 1

	#calc cliques
	cliquishness = np.zeros(num_nodes)
	for node in range(num_nodes):
		if( node % 400 == 0):
			print("Clique calc ", np.round(node/num_nodes, 1), " done")
		#no reflexive edges so we're not included in this list
		num_neighbours = np.sum(adj_mat[node])
		num_possible_edges = (num_neighbours*(num_neighbours-1)) / 2
		if (num_possible_edges == 0):
			cliquishness[node] = 1.0
			continue
		num_edges = 0

		for neighbour in range(num_nodes):
			if (adj_mat[node][neighbour] == 1):
				for neighbours_neighbour in range(num_nodes):
					if (adj_mat[neighbour][neighbours_neighbour] == 1):
						if (adj_mat[node][neighbours_neighbour] == 1):
							num_edges +=1

		if (num_possible_edges < num_edges):
			print(num_possible_edges, num_edges)
			exit()
		cliquishness[node] = num_edges / num_possible_edges

	print("Average Cliquishness is ", np.mean(cliquishness))




	fig, ax = plt.subplots(ncols=2, nrows=2)
	ax[0][0].set_title("Node Degrees")
	im1 = ax[0][0].scatter(np.arange(num_nodes), degrees, marker=',',lw=0,s=1)
	ax[0][1].set_title("Node Degree Count Histogram")
	im2 = ax[0][1].hist(degrees, bins=100, facecolor = '#2ab0ff', edgecolor='#169acf')
	ax[1][0].set_title("Cliquishness")
	im3 = ax[1][0].scatter(np.arange(num_nodes), cliquishness, marker=',',lw=0,s=1)
	ax[1][1].set_title("Cliquishness Count Histogram")
	im4 = ax[1][1].hist(cliquishness, bins=100, facecolor = 'firebrick', edgecolor='orange')
	plt.show()


if __name__ == "__main__":
	main("facebook_combined.txt", 4039)