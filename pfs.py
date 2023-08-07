#Written by Joel Wildman (22984156)
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import random
import sys

#TODO: agent in position 0 is not swapped with the rest of low class
#TODO: diagonal connections in lattice

# PFS model operates on a lattice divided into social ranks
# this approach is not compatible with interaction graphs like follower graphs
#each node possesses a private opinion x(i), public opinion z(i), threshold for falsification y(i),
# threshold for internalising k(i) = 3, number of previous falsifications φ, and previously expressed opinion z*(i)

internalising_threshold = 3

SR = 0
INT_OPINION = 1
EXT_OPINION = 2
FAKER_THRESHOLD = 3 
PREV_FAKES = 4
PREV_EXPRESSION = 5

#generates a lattice of nodes in the form of an adjacency matrix
def gen_lattice(size):
	#size must be a square number
	if (size ** 0.5 % 1.0 != 0.0):
		print("size must be a square number for a square lattice ( s = n*n )")
		exit()
	lattice_length = int(size ** 0.5)
	network = np.zeros((size, size))

	for i in range(size):
		#left and right nodes
		if (i % lattice_length == 0): #if it's in the first column, wrap around
			network[i][i+1] = 1 # right node
			network[i][i+lattice_length-1] = 1 #left node
		elif (i % lattice_length == lattice_length - 1): #if it's in the last column, wrap around
			network[i][i-(lattice_length-1)] = 1
			network[i][i-1] = 1
		else:
			network[i][i+1] = 1
			network[i][i-1] = 1

		#nodes above and below are connected
		if (i/lattice_length < 1): #if node is in first row, only connect below
			network[i][i+lattice_length] = 1 

		elif ((i+1)/lattice_length >= lattice_length-1): #if node is in last row, only connect above
			network[i][i-lattice_length] = 1
		else:
			network[i][i-lattice_length] = 1
			network[i][i+lattice_length] = 1

	#diagonals
	print(network)
	return network


def gen_agents(size, segregation):

	if (size ** 0.5 % 1.0 != 0.0):
		print("size must be a square number for a square lattice ( s = n*n )")
		exit()

	#social rank of each agent follows a series of uniform distributions 
	sr = []
	#this distribution is on page 9 of the paper, page 393 of the journal
	sr_distribution = [	0.010625,
						0.03125,
						0.083125,
						0.125,
						0.15375,
						0.1925,
						0.15375,
						0.125,
						0.083125,
						0.03125,
						0.010625]
	for proportion in range(len(sr_distribution)):
		#avoid filling the sr list with empty arrays
		if (int(sr_distribution[proportion]*size) == 0):
			continue
		#there are 11 uniform sr distributions, starting at U(0) and increasing by 0.1 from U(0.05, 0.15)
		sr_range = (max((proportion*0.1)-0.05, 0.0), min((proportion*0.1)+0.05, 1.0))
		sr.append(np.random.uniform(low=sr_range[0], high=sr_range[1], size=int(sr_distribution[proportion]*size + 1) ))
	sr = np.hstack(np.array(sr, dtype=object))
	
	if (sr.size != size):
		print("be aware new array does not match input array size, snipping")
		print(len(sr), size)
		sr = sr[:size]

	#this 2d matrix will store all agent data
	agents = np.zeros((sr.size, 6))
	agents[:, SR] = sr

	#at t=0 internal opinion x(i)=sr(i)
	agents[:, INT_OPINION] = np.copy(sr)

	#at t=0 external opinion z(i) exactly represents their internal opinion
	agents[:, EXT_OPINION][agents[:, INT_OPINION] > 0.5] = 1

	#at t=0 threshold for falsification y(i) is a uniform distribution
	#need to use a loop here to generate a new rand() seed for each
	for i in range(sr.size):
		if (agents[i][INT_OPINION] < 0.5):
			agents[i][FAKER_THRESHOLD] = np.random.rand() * 0.5 + 0.5
		else:
			agents[i][FAKER_THRESHOLD] = np.random.rand() * 0.5

	#at t=1 number of previous falsifications φ=0 (unnecessary operation, kept for clarity)
	agents[:, PREV_FAKES] = np.zeros(agents[:, INT_OPINION].size) 

	#at t=1 previous expressed opinion z*(i)=z(i)
	agents[:, PREV_EXPRESSION] = np.copy(agents[:, EXT_OPINION])

	#now rearrange agents to fit class posiions (high, medium, low, medium) bands (see page 394)
	#manually find the borders of medium class
	low_border = 0
	high_border = 0
	for i in range(sr.size):
		if (sr[i] > 0.35 and low_border == 0):
			low_border = i-1
		if (sr[i] > 0.65 and high_border == 0):
			high_border = i-1
	midpoint = low_border*2
	print(low_border, midpoint, high_border)
	#we take half the medium class and swap it with the low class
	tmp = np.copy(agents[0:low_border][:])
	agents[0:low_border][:] = agents[low_border:midpoint][:]
	agents[low_border:midpoint][:] = tmp

	#now apply segregation level, by placing agents in a separate class area
	if (segregation < 1.0 and segregation > 0.0):
		for i in range(sr.size):
			if (np.random.rand() > segregation):
				#swap this agent with another class agent
				#we know class divisions occur at sum(sr_distribution[:4]) and sum(sr_distribution[:7])

				#low class to middle or high area
				if (agents[i][SR] <= 0.35):
					exchange = random.choice(np.arange(agents.shape[0])[np.r_[0:low_border,midpoint:agents.shape[0]]])
					tmp = np.copy(agents[exchange])
					agents[exchange] = agents[i]
					agents[i] = tmp
				#middle class to low or high area
				elif (agents[i][SR] <= 0.65):
					exchange = random.choice(np.arange(agents.shape[0])[np.r_[low_border:midpoint, high_border:agents.shape[0]]])
					tmp = np.copy(agents[exchange])
					agents[exchange] = agents[i]
					agents[i] = tmp
				else:
					exchange = random.choice(np.arange(high_border))
					tmp = np.copy(agents[exchange])
					agents[exchange] = agents[i]
					agents[i] = tmp
	return agents


def draw_agents(agents):
	picture = np.zeros((int(agents.shape[0] ** 0.5), int(agents.shape[0] ** 0.5),  3))
	#picture[agents[:, SR] < 0.35] = np.array([200, 50, 20])
	#print(picture.shape)
	for i in range(agents.shape[0]):
		if (agents[i][SR] < 0.35):
			picture[i%int((agents.shape[0] ** 0.5))][int(i/agents.shape[0] ** 0.5)] = 0.2
		elif (agents[i][SR] < 0.65):
			picture[i%int((agents.shape[0] ** 0.5))][int(i/agents.shape[0] ** 0.5)] = 0.5
		elif (agents[i][SR] > 0.65):
			picture[i%int((agents.shape[0] ** 0.5))][int(i/agents.shape[0] ** 0.5)] = 0.8
	#print(picture)
	fig, ax = plt.subplots()
	#colour map doesn't work for some reason
	a = plt.imshow(picture, cmap='seismic', interpolation='nearest', vmin=0, vmax=1)

	patches = [	mpatches.Patch(facecolor=(0.2, 0.2, 0.2)), 
				mpatches.Patch(facecolor=(0.5, 0.5, 0.5)),
				mpatches.Patch(facecolor=(0.8, 0.8, 0.8))]
	plt.legend(patches, ["Low Class", "Middle Class", "High Class"])
	plt.show()
	return



# social ranks are U[0, 1], with low rank being [0, 0.35], medium [0.35, 0.65], high [0.65, 1]
# these ranks do not change over the course of the simulation

#public opinion z(i) is 1 or 0

#private opinion x(i) is continuous [0, 1]
# initial (t=0) private opinion = f(sr(i)), that is social rank of that individual
# I think this is bullshit and will comment so in the thesis

#threshold for preference falsification y(i) dictates how willing an agent is to fake their belief
# U[0.5, 1] if x(i) < 0.5 and U[0, 0.5) if x(i) ≥ 0.5
#will be updated as simulation progresses

# segregation level (S) [0, 1] dictates how much mixing there is between social classes when t=0
# if S < 1, a random sample of agents are exchanged with another agent from outside their class

#simulation:
#roundstart:
# update coherence heuristic to match private opinion with public one (to a degree)
	# if φ is the number of previous falsifications and k(i) is a threshold for internalizing z(i)
	# if φ > k,		x(i) = (x(i) + z(i))/2 (average external & internal)
			#		y(i) = 1 - x(i)
			#		φ = 0
	# else			x(i) and y(i) do not change
#note that k=3 for all agents
# φ
# 	φ will be set to 0 if agent i does not falsify opinion or coherence heurestic is triggered
#	φ is incremented by 1 is the falsification is valid, nothing happens if it is not valid

# Apply Goffman heuristic
# 	if there are enough high rank conflicting opinions in the area, an agent will fake their opinion
#	however this does not count towards the coherence heuristic

def iterate_network(network, agents):
	#apply coherence heuristic for each node:
	for i in range(agents):
		

def main(size, segregation):
	network = gen_lattice(size)
	agents = gen_agents(size, segregation)
	draw_agents(agents)
	return

if __name__ == "__main__":
	if (len(sys.argv) != 3):
		print("Please enter a network size and segregation level")
		exit()
	main(int(sys.argv[1]), float(sys.argv[2]))
