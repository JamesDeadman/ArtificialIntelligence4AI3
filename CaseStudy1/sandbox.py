import numpy as np

#print()


# Following are configuration parameters for GA 
DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # DNA crossover probability 
MUTATION_RATE = 0.000    # DNA mutation probability
N_GENERATIONS = 100      # Number of generation for iteration 
X_BOUND = [0, 5]         # upper and lower bounds for independent variable x


# To find the maximum of this function
def F(x): return np.sin(10*x)*x + np.cos(2*x)*x

# find non-zero fitness for selection
# in order to avoid the fitness being zero,
# it is: 1) subtracted by the minimal fitness, and 2)added by a small positive number 1e-3
def get_fitness(pred): return pred + 1e-3 - np.min(pred)


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop): return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]

# pop=np.random.randint(2, size=(10))
print(np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool))
