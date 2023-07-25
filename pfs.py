#Written by Joel Wildman (22984156)
import numpy as np

# PFS model operates on a lattice divided into social ranks
# this approach is not compatible with interaction graphs like follower graphs

#generates a lattice of nodes in the form of an adjacency matrix
def gen_lattice(size):
	#size must be a square number
	lattice_length = int(size ** 0.5)
	network = np.zeros((size, size))

	for i in range(size):
		#nodes to the left and right are connected
		if (i % lattice_length == 0): #if it's the first in the row, wrap around
			network[i][i+1] = 1 
			network[i][i+lattice_length-1] = 1
		elif (i % lattice_length == lattice_length - 1): #if it's the last in the row, wrap around
			network[i][i-(lattice_length-1)] = 1
			network[i][i-1] = 1
		else:
			network[i][i+1] = 1
			network[i][i-1] = 1

		#nodes above and below are connected

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
# 
