import numpy as np

def parse_file(filename):
	values = np.loadtxt(filename, dtype=int)
	print(values.shape)
	return values