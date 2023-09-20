from matplotlib import pyplot as plt

file = open("int_op_lattice_heatmap", "r")

contents = file.readlines()
for i in range(len(contents)):
	contents[i] = contents[i].split(",")

for i in range(len(contents)):
	for j in range(len(contents[i])):
		contents[i][j] = float(contents[i][j])
print(contents)
#contents.reverse()
plt.imshow(contents, cmap='Greys', interpolation='nearest', vmin=0.4, vmax=0.7, origin="lower")
plt.show()