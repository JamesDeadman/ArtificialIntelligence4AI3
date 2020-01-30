"""
Apply Genetic Algorithm (GA) to find a maximum point in a function.

"""
import numpy as np
import matplotlib.pyplot as plt

# Following are configuration parameters for GA 
DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # DNA crossover probability 
MUTATION_RATE = 0.003    # DNA mutation probability
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


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

plt.ion()       # # Turn the interactive plotting mode on
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))

for i in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))    # compute function value by extracting DNA

    # something about plotting
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)
    if i < (N_GENERATIONS - 1): sca.remove()
    
    # GA operators for evolution
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child

plt.ioff()  # Turn the interactive plotting mode off
plt.show()
