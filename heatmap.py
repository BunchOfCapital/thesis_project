from matplotlib import pyplot as plt
import numpy as np
import sys

filename = sys.argv[1]
file = open(filename, "r")

contents = file.readlines()
for i in range(len(contents)):
	contents[i] = contents[i].split(",")

for i in range(len(contents)):
	for j in range(len(contents[i])):
		contents[i][j] = float(contents[i][j])
print(contents)
#contents.reverse()

fig, ax = plt.subplots()

heatmap = plt.pcolor(contents, cmap='Greys')
plt.colorbar(heatmap)
plt.title("Average Internal Opinion after 30 Iterations (Facebook Network) ")
ax.set_yticks(np.round(np.linspace(0, 10, num=11), 1), labels=np.round(np.linspace(0.0, 1.0, num=11), 1))
ax.set_xticks(np.round(np.linspace(0, 10, num=11), 1), labels=np.round(np.linspace(0.0, 1.0, num=11), 1))
plt.xlabel("P")
plt.ylabel("S")
plt.show()