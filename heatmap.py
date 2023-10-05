from matplotlib import pyplot as plt
import numpy as np
import sys

filename = sys.argv[1]
file = open(filename, "r")
final_map = np.zeros((11, 11))
for data in range(10):
	fname = "int_op_og_heatmap_" + str(data)
	file = open(fname, "r")
	contents = file.readlines()
	for i in range(len(contents)):
		contents[i] = contents[i].split(",")

	for i in range(len(contents)):
		for j in range(len(contents[i])):
			contents[i][j] = float(contents[i][j])
	print(contents)
	final_map += np.array(contents)
#contents.reverse()
final_map = final_map/10
fig, ax = plt.subplots()

heatmap = plt.pcolor(final_map, cmap='Greys')
plt.colorbar(heatmap)
plt.title("Average Internal Opinion after 30 Iterations (Lattice Network) ")
ax.set_yticks(np.round(np.linspace(0, 10, num=11), 1), labels=np.round(np.linspace(0.0, 1.0, num=11), 1))
ax.set_xticks(np.round(np.linspace(0, 10, num=11), 1), labels=np.round(np.linspace(0.0, 1.0, num=11), 1))
plt.xlabel("P")
plt.ylabel("S")
plt.show()