"""
Visualize Genetic Algorithm to find a maximum point in a function.

"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 20            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 100
X_BOUND = [0, 5]         # x upper and lower bounds
Y_BOUND = [0, 5]         # y upper and lower bounds

def F(X,Y): return (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
    # to find the maximum of this function

# find non-zero fitness for selection
def get_fitness(F_values): return F_values + 1e-3 - np.min(F_values)

# convert binary DNA for X to decimal and normalize it to a range(0, 5)
def translateDNA_X(pop): 
    pop_X=pop[:, 0:10]
    return pop_X.dot(2 ** np.arange(np.int(DNA_SIZE/2))[::-1]) / float(2**(DNA_SIZE/2)-1) * X_BOUND[1]

# convert binary DNA for Y to decimal and normalize it to a range(0, 5)
def translateDNA_Y(pop): 
    pop_Y=pop[:, 10:20]
    return pop_Y.dot(2 ** np.arange(np.int(DNA_SIZE/2))[::-1]) / float(2**(DNA_SIZE/2)-1) * Y_BOUND[1]

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


############################################################
pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   

for i in range(N_GENERATIONS):
    F_values = F(translateDNA_X(pop), translateDNA_Y(pop))    # compute function value by extracting DNA
  
    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child
    
pop_best_X_DNA = pop[np.argmax(fitness), 0:10] 
pop_best_X_real = pop_best_X_DNA.dot(2 ** np.arange(np.int(DNA_SIZE/2))[::-1]) / float(2**(DNA_SIZE/2)-1) * X_BOUND[1]

pop_best_Y_DNA = pop[np.argmax(fitness), 10:20] 
pop_best_Y_real = pop_best_Y_DNA.dot(2 ** np.arange(np.int(DNA_SIZE/2))[::-1]) / float(2**(DNA_SIZE/2)-1) * X_BOUND[1]

print("Best real X value: ", pop_best_X_real)
print("Best real Y value: ", pop_best_Y_real)
print("Maximal function value: ", F(pop_best_X_real, pop_best_Y_real))

