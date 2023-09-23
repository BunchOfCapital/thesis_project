#Written by Joel Wildman (22984156)
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import random
import sys
import math

#algorithm as specified in  https://link.springer.com/article/10.1007/s11135-019-00909-2

#TODO: agent in position 0 is not swapped with the rest of low class
#TODO: previously expressed opinion is deprecated, remove
#TODO: make neighbourhood lookup use a dictionary

#TODO: FIX OPINION UPDATE HAPPENING TOO EASILY FOR SOME REASON

# PFS model operates on a lattice divided into social ranks
# this approach is not compatible with interaction graphs like follower graphs
# to somewhat bridge this gap I have treated the lattice as an adjacency matrix, 4x4 neihgbourhoods are replaced with nodes within 3 jumps
# each node possesses a private opinion x(i), public opinion z(i), threshold for falsification y(i), provisional threshold y*(i)
# threshold for internalising k(i) = 3, number of previous falsifications φ, and previously expressed opinion z*(i)

internalising_threshold = 3
edge_chance = 0

SR = 0
INT_OPINION = 1
EXT_OPINION = 2
FAKER_THRESHOLD = 3 
PROV_THRESHOLD = 4
PREV_FAKES = 5
PREV_EXPRESSION = 6

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
	return network

def parse_file(filename,):
	values = np.loadtxt(filename, dtype=int)
	new_values = []
	for i in range(values.shape[0]):
		if (np.random.rand() < edge_chance ):
			new_values.append(values[i])
	
	return np.array(new_values)

#this dataset has 4039 nodes
def gen_facebook_net(size):
	edges = parse_file("facebook_combined.txt")
	network = np.zeros((size,size))
	for i in range(edges.shape[0]):
		network[edges[i][0]][edges[i][1]] = 1
	return network


# agent notes

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

def gen_agents(size, segregation):

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
	agents = np.zeros((sr.size, 7))
	agents[:, SR] = sr

	#at t=0 internal opinion x(i)=sr(i)
	agents[:, INT_OPINION] = np.copy(sr)

	#at t=0 external opinion z(i) exactly represents their internal opinion
	agents[:, EXT_OPINION][agents[:, INT_OPINION] > 0.5] = 1

	#at t=0 threshold for falsification y(i) is a uniform distribution
	#need to use a loop here to generate a new rand() seed for each
	for i in range(sr.size):
		if (agents[i][INT_OPINION] < 0.5):
			agents[i][FAKER_THRESHOLD] = round(np.random.rand() * 0.5 + 0.5, 2)
		else:
			agents[i][FAKER_THRESHOLD] = round(np.random.rand() * 0.5, 2)

	#provisional threshold for falsification is unused at t=0
	agents[:, PROV_THRESHOLD] = np.copy(agents[:, FAKER_THRESHOLD])

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
	#print(low_border, midpoint, high_border)

	#now we shuffle agents within a class to make it truly random
	np.random.shuffle(agents[0:low_border])
	np.random.shuffle(agents[low_border:high_border])
	np.random.shuffle(agents[high_border:])

	#we take half the medium class and swap it with the low class
	tmp = np.copy(agents[0:low_border][:])
	agents[0:low_border][:] = agents[low_border:midpoint][:]
	agents[low_border:midpoint][:] = tmp

	#now apply segregation level, by placing agents in a separate class area
	if (segregation <= 1.0 and segregation >= 0.0):
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


def get_pictures(all_agents):
	if (all_agents.shape[0] ** 0.5 % 1.0 != 0.0):
		print("Agents are not on a lattice, snipping to fit square image")
		closest_square = int(math.sqrt(all_agents.shape[0]))**2
		agents = all_agents[:closest_square, :closest_square]
		print(agents.shape)
	else:
		agents = all_agents
		

	ranks = np.zeros((int(agents.shape[0] ** 0.5), int(agents.shape[0] ** 0.5)))
	internal = np.zeros((int(agents.shape[0] ** 0.5), int(agents.shape[0] ** 0.5)))
	external = np.zeros((int(agents.shape[0] ** 0.5), int(agents.shape[0] ** 0.5)))

	for i in range(agents.shape[0]):
		internal[i%int((agents.shape[0] ** 0.5))][int(i/agents.shape[0] ** 0.5)] = agents[i][INT_OPINION]

		external[i%int((agents.shape[0] ** 0.5))][int(i/agents.shape[0] ** 0.5)] = float(agents[i][EXT_OPINION])
		#if an agent is faking positive, colour high
		if (agents[i][INT_OPINION] < 0.5 and agents[i][EXT_OPINION] == 1):
			external[i%int((agents.shape[0] ** 0.5))][int(i/agents.shape[0] ** 0.5)] = 0.7
		#if an agent is faking negative, colour low
		if (agents[i][INT_OPINION] > 0.5 and agents[i][EXT_OPINION] == 0):
			external[i%int((agents.shape[0] ** 0.5))][int(i/agents.shape[0] ** 0.5)] = 0.3

		if (agents[i][SR] < 0.35):
			ranks[i%int((agents.shape[0] ** 0.5))][int(i/agents.shape[0] ** 0.5)] = 0.2
		elif (agents[i][SR] < 0.65):
			ranks[i%int((agents.shape[0] ** 0.5))][int(i/agents.shape[0] ** 0.5)] = 0.5
		elif (agents[i][SR] > 0.65):
			ranks[i%int((agents.shape[0] ** 0.5))][int(i/agents.shape[0] ** 0.5)] = 0.8
	
	return ranks, internal, external 


#simulation notes:
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


def find_neighbours(network, agent, depth, neighbours):
	if depth >= 3:
		return []
	current_depth = depth + 1
	neighbours = set([])
	for i in range(network.shape[0]):
		if (network[agent][i] == 1 and i not in neighbours):
			neighbours.add(i)
			neighbours.update(find_neighbours(network, i, current_depth, neighbours))
	return list(neighbours)

def find_neighbourhood(neighbourhoods, agent):
	for neighbourhood in neighbourhoods:
		if agent in neighbourhood:
			return neighbourhood
	return []

def create_neighbourhood(neighbours, neighbourhoods):
	neighbourhood = []
	for agent in neighbours:
		if (len(find_neighbourhood(neighbourhoods, agent)) == 0):
			neighbourhood.append(agent)
	return neighbourhood


#NOTE THAT THIS CALCULATES AVERAGE *INTERNAL* OPINION
#this contradicts the paper, but matches the findings, likely this is also present in the original code
def get_general_opinion(agents):
	total_sum = agents[:,INT_OPINION].sum()
	return total_sum/agents.shape[0]

def gen_square_neighbourhoods():
	#create a list of neighbourhoods that will be added to later
	neighbourhoods = [[] for i in range(100)]
	#this is the original 4x4 neighbourhood idea
	offset = int(random.random()*1600)
	for i in range(1600):
		hood = []
		horizontal_group = int(i/4)%10
		vertical_group = int(int(i/40)/4)
		neighbourhoods[horizontal_group+(vertical_group*10)].append((i+offset)%1600)
	return neighbourhoods


def iterate_network(network, agents, P, stats_mode=False, stats_buffer={}):

	if (stats_mode):
		faked_supports_g = 0
		faked_supports_no_g= 0
		opinion_changes = 0


	#take the general opinion from the previous round for use later
	general_opinion = round(get_general_opinion(agents),2)
	#this is modified by some noise
	noise = np.random.normal(loc=0.5, scale=0.05, size=agents.shape[0])
	percieved_opinions = general_opinion + noise - 0.5
	percieved_opinions = np.clip(percieved_opinions, 0.0, 1.0)

	#create a list of neighbourhoods that will be added to later
	neighbourhoods = []

	#neighbourhoods = gen_square_neighbourhoods()

	reference_opinions = 0

	#iterate over all agents, updating in random order
	agent_list = np.arange(agents.shape[0])
	np.random.shuffle(agent_list)
	for i in agent_list:

		# firstapply coherence heuristic for each node:
		if (agents[i][PREV_FAKES] >= internalising_threshold):
			agents[i][INT_OPINION] = (agents[i][EXT_OPINION] + agents[i][INT_OPINION])/2
			agents[i][FAKER_THRESHOLD] = round(1 - agents[i][INT_OPINION], 2)
			agents[i][PREV_FAKES] = 0
			if (stats_mode):
				opinion_changes += 1
		else:
			#no change
			agents[i][INT_OPINION] = agents[i][INT_OPINION]
			agents[i][FAKER_THRESHOLD] = agents[i][FAKER_THRESHOLD]


		#second apply Goffman Heuristic to each node ("impression management")
		#we need highest rank agent in the neighbourhood (within n steps)
		neighbours = find_neighbourhood(neighbourhoods, i)
		if (len(neighbours) == 0):
			neighbours = find_neighbours(network, i, 0, [])
			#make sure there's no overlap in neighbourhoods
			neighbourhoods.append(create_neighbourhood(neighbours, neighbourhoods))

		#sort by social rank
		influences = sorted(list(neighbours), key=lambda agent: agents[agent][SR], reverse=True)
		#see equation 3 (p. 397)
		#we provisionally adjust the falsification threshold for these agents
		if (agents[i][SR] < 0.5):
			#goffman heuristic is only possible if the agent has neighbours 
			if (len(influences) != 0):
				goffman_influence = abs(agents[i][FAKER_THRESHOLD] - agents[i][INT_OPINION]) / (1 + abs(agents[i][SR] - agents[influences[0]][SR]) )
				if (goffman_influence < 0):
					print("Goffman has calculated incorrectly", goffman_influence)
					exit()
				if (agents[i][INT_OPINION] < 0.5 and agents[influences[0]][EXT_OPINION] == 1):
					agents[i][PROV_THRESHOLD] = round(agents[i][INT_OPINION] + goffman_influence, 2)
				if (agents[i][INT_OPINION] >= 0.5 and agents[influences[0]][EXT_OPINION]  == 0):
					agents[i][PROV_THRESHOLD] = round(agents[i][INT_OPINION] - goffman_influence, 2)

		#expected opinion of the agent is based on their neighbours
		if (len(neighbours) != 0):
			expected_opinion = 0
			for j in range(len(neighbours)):
				if (agents[neighbours[j]][EXT_OPINION] == 1):
					expected_opinion += 1.0
			if (expected_opinion == 0):
				expected_opinion = 0 #no change (but also no division by 0)
			else:
				expected_opinion = round(expected_opinion/len(neighbours), 2)
		#if an agent has no neighbours at all, it's opinion will be purely influenced by global expressions
		else:
			expected_opinion = percieved_opinions[i]
		

		#calculate final reference opinion for this agent, using variable P which scales influence of global & local opinions
		#higher P means agents are influenced more by the global opinion and less by their neighbours
		reference_opinion = round((P * percieved_opinions[i]) + ((1 - P) * expected_opinion),2)
		reference_opinions += reference_opinion

		#the agent will now express an opinion
		if (reference_opinion >= agents[i][PROV_THRESHOLD]):
			agents[i][EXT_OPINION] = 1
			
			if (agents[i][INT_OPINION] < 0.5):
				if (agents[i][PROV_THRESHOLD] > agents[i][FAKER_THRESHOLD]):
					print("catastrophic calculation error (probably a rounding issue)")
					print("provisional: ", agents[i][PROV_THRESHOLD])
					print("actual: ", agents[i][FAKER_THRESHOLD])
				#if an agent has falsified its opinions due to goffman heuristic, do not count it towards φ
				if (reference_opinion < agents[i][FAKER_THRESHOLD]):
					agents[i][PREV_FAKES] += 0
					if (stats_mode):
						faked_supports_g += 1 
				else:
					#otherwise it increases doubt
					agents[i][PREV_FAKES] += 1
					if (stats_mode):
						faked_supports_no_g += 1

			#if no falsification occurred, reset counter
			else:
				agents[i][PREV_FAKES] = 0
		else:
			agents[i][EXT_OPINION] = 0

			if (agents[i][INT_OPINION] >= 0.5):
				#high opinion (but low rank) agents can also trigger goffmans heuristic to protect themselves
				if(reference_opinion > agents[i][FAKER_THRESHOLD]):
					agents[i][PREV_FAKES] += 0
					if (stats_mode):
						faked_supports_g += 1

				else:
					agents[i][PREV_FAKES] += 1
					if (stats_mode):
						faked_supports_no_g += 1
			else:
				agents[i][PREV_FAKES] = 0


		#provisional threshold is reset if it has been used this round
		if (agents[i][PROV_THRESHOLD] != agents[i][FAKER_THRESHOLD]):
			agents[i][PROV_THRESHOLD] = agents[i][FAKER_THRESHOLD]

	if (stats_mode):
		#calculate average neighbourhood size and save to buffer
		neighbourhood_sizes = sum(len(hood) for hood in neighbourhoods)
		avg_size = neighbourhood_sizes/len(neighbourhoods)
		stats_buffer["hood_sizes"].append(avg_size)
		stats_buffer["num_hoods"].append(len(neighbourhoods))

		#add opinion change and faked opinion stats to buffer
		stats_buffer["opinion_changes"].append(opinion_changes)
		stats_buffer["faked_supports_g"].append(faked_supports_g)
		stats_buffer["faked_supports_no_g"].append(faked_supports_no_g)

		print("average global reference opinion is ", reference_opinions/agents.shape[0])

	return agents


def main(size, segregation, P, iterations, sample_data=False, stats_mode=False):

	#GENERATE NECESSARY DATA
	print("Generating network...")
	if (sample_data):
		#facebook dataset contains 4039 nodes
		network = gen_facebook_net(4039)
		print(network.shape)
	else:
		network = gen_lattice(size)

	print("Generating agents...")
	if (sample_data):
		agents = gen_agents(network.shape[0], segregation)
	else:
		agents = gen_agents(size, segregation)

	# plt.ion()
	# ranks, internal, external = get_pictures(agents)
	# fig1, ax = plt.subplots(nrows=2, ncols=3)

	if (stats_mode):
		print("Running in stats mode")
		stats_buffer = {"hood_sizes":[],
						"num_hoods":[],
						"faked_supports_g":[], 
						"faked_supports_no_g":[], 
						"avg_int_opinion":[],
						"avg_ext_opinion":[], 
						"median_int_opinion":[], 
						"opinion_changes":[]
						}
	else:
		stats_buffer={}

	#INITIALIZE GRAPHS
	# im1 = ax[0][0].imshow(ranks, cmap='magma', interpolation='nearest', vmin=0, vmax=1)
	# im2 = ax[0][1].imshow(internal, cmap='magma', interpolation='nearest', vmin=0, vmax=1)
	# im3 = ax[0][2].imshow(external, cmap='hot', interpolation='nearest', vmin=0, vmax=1)

	# #histograms
	# im4 = ax[1][0].hist(agents[:,SR], bins=50, range=(0.0, 1.0), facecolor = '#2ab0ff', edgecolor='#169acf')
	# im5 = ax[1][1].hist(agents[:,INT_OPINION], bins=50, range=(0.0, 1.0), facecolor = '#2ab0ff', edgecolor='#169acf')
	# im6 = ax[1][2].hist(agents[:,EXT_OPINION], bins=2, range=(0.0,1.0), facecolor = '#2ab0ff', edgecolor='#169acf')

	# patches = [	mpatches.Patch(facecolor=(0.2, 0.2, 0.2)), 
	# 			mpatches.Patch(facecolor=(0.5, 0.5, 0.5)),
	# 			mpatches.Patch(facecolor=(0.8, 0.8, 0.8))]
	# ax[0][0].legend(patches, ["Low Class", "Middle Class", "High Class"])
	# ax[0][0].set_title("Social Ranks")
	# ax[0][1].set_title("Internal Opinions")
	# ax[0][2].set_title("External Opinions")
	# iterlabel = plt.text(0.0, 20.0, "Iteration 0")
	# fig1.tight_layout()
	# plt.show()


	#EXECUTE SIMULATION
	for i in range(iterations):
		print("Running iteration ", i+1, "...")

		#update charts
		# ranks, internal, external = get_pictures(agents)
		# iterlabel.set_text("Iteration " + str(i+1))
		# im1.set_data(ranks)
		# im2.set_data(internal)
		# im3.set_data(external)
		# fig1.canvas.draw()
		# fig1.canvas.flush_events()

		agents = iterate_network(network, agents, P, stats_mode, stats_buffer)

		external_average = (agents[:,EXT_OPINION].sum())/agents.shape[0]
		internal_average = agents[:,INT_OPINION].sum()/agents.shape[0]
		print("Internal avg: ", internal_average, " External avg: ", external_average)

		if (stats_mode):
			stats_buffer["avg_int_opinion"].append(internal_average)
			stats_buffer["avg_ext_opinion"].append(external_average)
			median_agent = np.argsort(agents[:,INT_OPINION])[int(agents.shape[0]/2)]
			stats_buffer["median_int_opinion"].append(agents[:,INT_OPINION][median_agent])

	# fig2, ax2 = plt.subplots(nrows=1, ncols=2)
	# ax2[0].set_title("Internal Opinions")
	# ax2[1].set_title("External/Expressed Opinions")
	# final_opinions = ax2[0].hist(agents[:,INT_OPINION], bins=50, range=(0.0, 1.0), facecolor = '#2ab0ff', edgecolor='#169acf')
	# final_expressions = ax2[1].hist(agents[:,EXT_OPINION], bins=2, range=(0.0,1.0), facecolor = '#2ab0ff', edgecolor='#169acf')
	# plt.show()

	#PLOT STATISTICS
	if (stats_mode):
		#save stats to file
		# int_hist = np.histogram(agents[:,INT_OPINION], bins=50)
		# file = open("data/pfs_p" + str(P) + "_s" + str(segregation) + "_i" + str(iterations), "a")
		# file.write("Average internal opinions: \n" + str(stats_buffer["avg_int_opinion"]) + "\n")
		# file.write("Average external opinions: \n" + str(stats_buffer["avg_ext_opinion"]) + "\n")
		# file.write("Faked supports due to Goffman: \n" + str(stats_buffer["faked_supports_g"]) + "\n")
		# file.write("Faked supports not due to Goffman: \n" + str(stats_buffer["faked_supports_no_g"]) + "\n")
		# file.write("Opinion changes each round: \n" + str(stats_buffer["opinion_changes"]) + "\n")
		# file.write("Final internal opinion distribution: \n" + str(int_hist[0]) + "\n" + str(int_hist[1]) + "\n\n")
		# file.close()

		file2 = open("int_op_gb_heatmap_"+ str(edge_chance), 'a')
		if (P == 1.0):
			entry = str(stats_buffer["avg_int_opinion"][-1]) + "\n"
		else:
			entry = str(stats_buffer["avg_int_opinion"][-1]) + ","

		file2.write(entry)
		file2.close()


		# fig3, ax3 = plt.subplots(nrows=3, ncols=2)
		# ax3[0][0].set_title("Faked Supports")
		# ax3[0][1].set_title("Average Neighbourhood Size")
		# ax3[1][0].set_title("Average Internal Opinion")
		# ax3[1][1].set_title("Average External Opinion")
		# ax3[2][0].set_title("Median Internal Opinion")
		# ax3[2][1].set_title("Number of Opinion Changes")

		# patches = [	mpatches.Patch(facecolor="blue"), 
		# 			mpatches.Patch(facecolor="orange")]
		# ax3[0][0].legend(patches, ["Goffman Fakes", "Valid Fakes"])

		# ax3[0][0].bar(np.arange(iterations) - 0.2, stats_buffer["faked_supports_g"],label="Goffman Fakes", width=0.4)
		# ax3[0][0].bar(np.arange(iterations) + 0.2, stats_buffer["faked_supports_no_g"],label="Other Fakes", width=0.4)
		# ax3[0][1].bar(np.arange(iterations), stats_buffer["hood_sizes"])
		# ax3[1][0].plot(stats_buffer["avg_int_opinion"])
		# ax3[1][0].set_yticks(np.linspace(0.0, 1.0, num=9))
		# ax3[1][1].plot( stats_buffer["avg_ext_opinion"])
		# ax3[1][1].set_yticks(np.linspace(0.0, 1.0, num=9))
		# ax3[2][0].plot(stats_buffer["median_int_opinion"])
		# ax3[2][0].set_yticks(np.linspace(0.0, 1.0, num=9))
		# ax3[2][1].plot(stats_buffer["opinion_changes"])

		# fig3.tight_layout()
		# #fig3.savefig("data/pfs_p" + str(P) + "_s" + str(segregation) + "_i" + str(iterations) + ".png")
		# plt.show()

	print("Average Neighbourhood Sizes:")
	print(stats_buffer["hood_sizes"])

	#input()
	return

if __name__ == "__main__":
	if (len(sys.argv) < 5):
		print("Please enter a network size, segregation level, P value, and iteration number (optional -f for facebook data, -s for stats mode)")
		exit()
	elif (len(sys.argv) == 5):
		#run custom values
		main(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
		exit()
	elif ("-f" in sys.argv or "-s" in sys.argv):
		#run with options
		for chance in [0.7, 0.5, 0.3]:
			edge_chance = chance
			print("Running with chance ", edge_chance)
			for i in range(11):
				for j in range(11):
					main(1600, 0.1*i, 0.1*j, 30, sample_data=True, stats_mode=True)
		#main(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sample_data=("-f" in sys.argv), stats_mode=("-s" in sys.argv))
	else:
		print("Too many arguments")
		exit()
