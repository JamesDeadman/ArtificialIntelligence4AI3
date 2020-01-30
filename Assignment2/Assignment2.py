import math
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-3

class Chromosome:

    #Construct a random chromosome
    def __init__(self, sequence, id, dnaSize, minValue, maxValue):
        self.sequence = sequence
        self.id = id
        self.dnaSize = dnaSize
        self.minValue = minValue
        self.maxValue = maxValue
        self.values = np.random.randint(2, size=(dnaSize))

    #Construct a random chromosome by crossing two parents
    @classmethod
    def Spawn(cls, sequence, parentA, parentB):
        sequence = parentA.sequence.copy()
        crossPoints = np.random.randint(0, 2, size=parentA.dnaSize).astype(np.bool)
        sequence[crossPoints] = parentB.sequence[crossPoints]
        return cls(sequence, parentA.id, parentA.dnaSize, parentA.minValue, parentA.maxValue)
    
    #Calculate the decimal value
    def GetValue(self):
        return self.values.dot(2 ** np.arange(self.dnaSize)[::]) / float(2 ** self.dnaSize - 1) * (self.maxValue - self.minValue) + self.minValue

    def GetId(self):
        return self.id

    #Mutate each bit
    def Mutate(self):
        mutationRate = self.sequence.GetMutationRate()
        for point in range(self.dnaSize):
            if np.random.rand() < mutationRate:
                self.values[point] = 1 if self.values[point] == 0 else 0

    #Find the mutation rate of the owning sequence
    def GetMutationRate(self):
        return self.sequence.GetMutationRate()


class Sequence:
    def __init__(self, population, parentA = None, parentB = None):
        self.population = population
        if(parentA is None): # this is a new sequence, generate chromosomes from the environment
            self.chromosomes = { x.GetId(): x for x in population.environment.GenerateChromosomes(self) }
        else: # this is a child of two parent sequences
            self.chromosomes = { }
            for k in parentA.chromosomes.keys() & parentB.chromosomes.keys():
                self.chromosomes[k] = Chromosome.Spawn(self, parentA.chromosomes[k], parentB.chromosomes[k])

    #Mutate each chromosome
    def Mutate(self):
        for chromosome in self.chromosomes:
             chromosome.mutate()

    #Calculate the value of this sequence
    def CalculateValue(self):
        self.value = self.population.environment.Calculate(self.chromosomes)

    #Return the value of this sequence
    def GetValue(self):
        return self.value

    def GetFitness(self):
        return self.fitness

    #Determine the fitness of this sequence TODO: move to the environment
    def CalculateFitness(self):
        self.fitness = self.population.GetLowestValue() - self.GetValue() + EPSILON

    #Return the mutation rate
    def GetMutationRate(self):
        return self.population.GetMutationRate()


class Population:
    def __init__(self, environment, populationSize):
        self.environment = environment
        self.populationSize = populationSize
        self.sequences = []
        for i in range(populationSize):
            self.sequences.append(Sequence(self))

    def GetMutationRate(self):
        return self.environment.GetMutationRate()

    def CalculateValues(self):
        self.lowestValueSequence = None
        for sequence in self.sequences:
            sequence.CalculateValue()
            if(self.lowestValueSequence is None or self.lowestValueSequence.GetValue() < sequence.GetValue()):
                self.lowestValueSequence = sequence

    def CalculateFitness(self):
        self.bestFit = None
        for sequence in self.sequences:
            sequence.CalculateFitness()
            if(self.bestFit is None or self.bestFit.GetFitness() < sequence.GetFitness()):
                self.bestFit = sequence
        self.totalFitness = sum(s.GetFitness() for s in self.sequences)

    #Build the next generation of sequences with weighting based on fitness
    def Select(self):
        fitness = np.array(list(s.GetFitness() for s in self.sequences))
        self.sequences = np.random.choice(self.sequences, size=self.populationSize, replace=True, p=(fitness / self.totalFitness))

    def Spawn(self):
        for i in range(self.populationSize):
            if np.random.rand() < environment.spawnRate:
                j = np.random.randint(0, self.populationSize, size=1)
                self.sequences[i] = Sequence(self, self.sequences[i], self.sequences[j])

    def Mutate(self):
        for sequence in self.sequences:
            sequence.Mutate()

    def GetBestFit(self):
        return self.bestFit

    def GetLowestValue(self):
        return self.lowestValueSequence.GetValue()

class Environment:
    R = 30
    V = 6.5e-6

    def __init__(self, mutationRate, spawnRate):
        self.mutationRate = mutationRate
        self.spawnRate = spawnRate

    def Calculate(self, chromosomes):
        Mu = chromosomes['Mu'].GetValue()
        Ms = chromosomes['Ms'].GetValue()
        Kt = chromosomes['Kt'].GetValue()
        K = chromosomes['K'].GetValue()
        C = chromosomes['C'].GetValue()
        return math.sqrt(math.pi * self.R * self.V * ((Kt * C) / (2 * Ms ** (3 / 2) * K * (1 / 2)) + ((Mu + Ms) * K ** 2) / (2 * C * Ms ** 2)))

    def GenerateChromosomes(self, sequence):
        chromosomes = []
        chromosomes.append(Chromosome(sequence, 'Mu', 10, 25.0, 40.0))
        chromosomes.append(Chromosome(sequence, 'Ms', 10, 400.0, 550.0))
        chromosomes.append(Chromosome(sequence, 'Kt', 10, 420000.0, 700000.0))
        chromosomes.append(Chromosome(sequence, 'K', 10, 60000.0, 90000.0))
        chromosomes.append(Chromosome(sequence, 'C', 10, 1900.0, 3100.0))
        return chromosomes

    def GetMutaitonRate(self):
        return self.mutationRate


class Optimization:
    def __init__(self, environment, numberOfGenerations, populationSize):
        self.numberOfGenerations = numberOfGenerations
        self.environment = environment
        self.population = Population(environment, populationSize)

    def Run(self):
        for i in range(self.numberOfGenerations):
            self.population.CalculateValues()
            self.population.CalculateFitness()
            self.population.Select()
            self.population.Spawn()
            self.population.Mutate()
            bestFit = self.population.GetBestFit()

            bestFit.GetValue()


environment = Environment(0.003, 0.8)
optimization = Optimization(environment, 100, 100)

optimization.Run()
