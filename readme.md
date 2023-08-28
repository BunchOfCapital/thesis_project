# Opinion Dynamics Models in Empirical Networks
## Aim
These models are implementations of existing published literature on Opinion Dynamics Models, with the intent to apply them to existing social media datasets and evaluate noteable trends. The social media dataset included in the repository is available from the Standford Network Analysis Project website at https://snap.stanford.edu/data/ego-Facebook.html.

More information on the execution of the models can be found in the comments of their respective files. To fully understand how they work and the justification behind their environments, update rules and initial conditions, please refer to the original literature. 

## How To

The two models, the Preference Falsification Simulation (pfs.py) and the DeGroot-Friedkin model (degroot-friedkin.py) can be run from the command line with python3.

#### To run pfs.py, please use:
```
python3 pfs.py <network size> <segregation value> <P value> <number of iterations> [-f]
```
Notes:

- Network size must be a square number to allow for a square lattice network
- Segregation value [0-1] refers to how random the distribution of agents are within the lattice
- P value refers to whether agents place more weight on local or global opinions
- Number of iterations refers to the execution length of the program
- -f option will use the facebook dataset file as the simulation environment rather than a lattice. (This will override the <network size> option with the closest square number to the facebook dataset size (3969))


#### To run degroot-friedkin.py, please use:
```
python3 degroot-friedkin.py <network size> <number of iterations per issue> <number of issues>
```
Notes: 
- The opinions of each node will not be reset between issues