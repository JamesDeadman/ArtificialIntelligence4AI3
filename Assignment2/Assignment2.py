""" 
    Assignment 2 - James Deadman 400040400

    Note about the code:

        I developed this application using OOP in order to maximize potential 
        for re-use.  Any function can be optimized and visualized by extending 
        the Environment and SequencePlotter classes and changing a few lines
        related to console output in the main function and Solver class.
        
        OOP Class concepts and relationships:
        A Sequence is a collection of Chromosomes and exists as a member of a 
        Population within an Environment which determins the conditions for
        fitness and rates of mutation and spawning.  The GA process is driven by 
        a Solver and is visualized using a SequencePlotter.
        
        Visuaization was included because the optimized values seemed suspicious;
        they are all tending to the min or max of their respective domains.  Since 
        we're dealing with a function of 5 variables, I ploted the function in 
        2D space with each variable's full domain in the horizontal axis with 
        the other variables at the optimal best-fit value and the function result 
        in the Y-Axis.  This visualization would not prove the minimum for all
        functions since each variable is considered individually.  I used a 2nd
        application to verify the results using brute-force scanning and found the 
        the results to be consistent with GA.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum


#GA prameters
MUTATION_RATE = 0.003
SPAWN_RATE = 0.8
POPULATION = 100
GENERATIONS = 500
EPSILON = 1e-3


#A generic binary sequence representing a float value within a given range
class Chromosome:
    #Construct a random chromosome
    def __init__(self, sequence, id, dnaSize, minValue, maxValue, values = None):
        self.sequence = sequence
        self.id = id
        self.dnaSize = dnaSize
        self.minValue = minValue
        self.maxValue = maxValue

        if values is None:
            self.values = np.random.randint(2, size=(dnaSize))
        else:
            self.values = values
   

    #Construct a random chromosome by crossing two parents
    @classmethod
    def Spawn(cls, sequence, parentA, parentB):
        values = parentA.values.copy()
        crossPoints = np.random.randint(0, 2, size=parentA.dnaSize).astype(np.bool)
        values[crossPoints] = parentB.values[crossPoints]
        return cls(sequence, parentA.id, parentA.dnaSize, parentA.minValue, parentA.maxValue, values)
    
    #Calculate the decimal value based on the binary value of the sequence
    def CalculateDecValue(self):
        self.decValue = self.values.dot(2 ** np.arange(self.dnaSize)[::-1])
        if(self.decValue < 0):
            print(self.decValue)

    #Calculate the float value based on the decimal value and the range
    def CalculateFloatValue(self):
        self.floatValue = self.decValue / float(2 ** self.dnaSize - 1) * (self.maxValue - self.minValue) + self.minValue

    #Calculate the decimal and float values
    def CalculateValue(self):
        self.CalculateDecValue()
        self.CalculateFloatValue()

    #Calculate the decimal value
    def GetValue(self):
        return self.floatValue

    #Return the domain for plotting
    def GetDomainLineSpace(self):
        return np.linspace(self.minValue, self.maxValue, 200)

    #Getter for the Id
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


#A generic collection of chromosomes to represent a set of variables resulting in a single function value defined in Environment
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
        for chromosome in self.chromosomes.values():
             chromosome.Mutate()

    #Calculate the value of this sequence
    def CalculateValue(self):
        for chromosome in self.chromosomes.values():
             chromosome.CalculateValue()
        self.value = self.population.environment.Calculate(self.chromosomes)

    #Return the value of this sequence
    def GetValue(self):
        return self.value

    def GetFitness(self):
        return self.fitness

    #Determine the fitness of this sequence TODO: move to the environment
    def CalculateFitness(self):
        highest = self.population.GetHighestValue()
        lowest = self.population.GetLowestValue()
        self.fitness = (highest - self.GetValue()) / (highest - lowest) + EPSILON

    #Return the mutation rate
    def GetMutationRate(self):
        return self.population.GetMutationRate()


#Represents a population for GA
class Population:
    def __init__(self, environment, populationSize):
        self.environment = environment
        self.populationSize = populationSize
        self.sequences = []
        for i in range(populationSize):
            self.sequences.append(Sequence(self))

    #Getter for the mutation rate owned by the environment
    def GetMutationRate(self):
        return self.environment.GetMutationRate()

    #Calculate the values of each sequence, keep track of the highest and lowest values
    def CalculateValues(self):
        self.lowestValueSequence = None
        self.highestValueSequence = None
        for sequence in self.sequences:
            sequence.CalculateValue()
            if(self.lowestValueSequence is None or self.lowestValueSequence.GetValue() > sequence.GetValue()):
                self.lowestValueSequence = sequence
            if(self.highestValueSequence is None or self.highestValueSequence.GetValue() < sequence.GetValue()):
                self.highestValueSequence = sequence

    #Calculate the fitness of each sequence, keep track of the best fit and sum of the fitness to be used in mate selection weighting
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

    #Mate sequence pairs within the population randomly with probability proportional to fit
    def Spawn(self):
        for i in range(self.populationSize):
            if np.random.rand() < self.environment.spawnRate:
                j = np.random.randint(0, self.populationSize)
                self.sequences[i] = Sequence(self, self.sequences[i], self.sequences[j])

    #Randomly mutate each sequence
    def Mutate(self):
        for sequence in self.sequences:
            sequence.Mutate()

    #Getter for the best fit sequence
    def GetBestFit(self):
        return self.bestFit
    
    #Getter for the sequence with the lowest value
    def GetLowestValue(self):
        return self.lowestValueSequence.GetValue()

    #Getter for the sequence with the highest value
    def GetHighestValue(self):
        return self.highestValueSequence.GetValue()


#VehicleSuspensionSystem - application specific class, contains the function defined in Assignment 2
class VehicleSuspensionSystem:
    R = 30.0
    V = 6.5e-6
    def __init__(self, Mu, Ms, Kt, K, C):
        self.Mu = Mu
        self.Ms = Ms
        self.Kt = Kt
        self.K = K
        self.C = C

    #Calculate and record the acceleration value
    def Calculate(self):
        self.sprungMassAcceleration = VehicleSuspensionSystem.CalculateSprungMassAcceleration(self.R, self.V, self.Mu, self.Ms, self.Kt, self.K, self.C)

    #Acceleration function as defined in Assignment 2
    @staticmethod
    def CalculateSprungMassAcceleration(R, V, Mu, Ms, Kt, K, C):
        return math.sqrt(math.pi * R * V * ((Kt * C) / (2 * (Ms ** (3 / 2)) * (K ** (1 / 2))) + ((Mu + Ms) * (K ** 2)) / (2 * C * (Ms ** 2))))

    #Getter for the acceleration value
    def GetSprungMassAcceleration(self):
        return self.sprungMassAcceleration


#The problem and conditions to be used by the GA solver
class Environment:
    def __init__(self, mutationRate, spawnRate):
        self.mutationRate = mutationRate
        self.spawnRate = spawnRate

    #Calculate the value for a given sequence
    def Calculate(self, chromosomes):
        vss = VehicleSuspensionSystem(chromosomes['Mu'].GetValue(), chromosomes['Ms'].GetValue(), chromosomes['Kt'].GetValue(), chromosomes['K'].GetValue(), chromosomes['C'].GetValue())
        vss.Calculate()
        return vss.GetSprungMassAcceleration()

    #Create a set of chromosomes for each variable defined in Assignment 2
    def GenerateChromosomes(self, sequence):
        chromosomes = []
        chromosomes.append(Chromosome(sequence, 'Mu', 16, 25.0, 40.0))
        chromosomes.append(Chromosome(sequence, 'Ms', 24, 400.0, 550.0))
        chromosomes.append(Chromosome(sequence, 'Kt', 24, 420000.0, 700000.0))
        chromosomes.append(Chromosome(sequence, 'K', 24, 60000.0, 90000.0))
        chromosomes.append(Chromosome(sequence, 'C', 24, 1900.0, 3100.0))
        return chromosomes

    def GetMutationRate(self):
        return self.mutationRate


#Function visualization
class SequencePlotter:
    def __init__(self, sequence):
        self.sequence = sequence

    def PlotAll(self):
        MuDomain = self.sequence.chromosomes['Mu'].GetDomainLineSpace()
        MsDomain = self.sequence.chromosomes['Ms'].GetDomainLineSpace()
        KtDomain = self.sequence.chromosomes['Kt'].GetDomainLineSpace()
        KDomain = self.sequence.chromosomes['K'].GetDomainLineSpace()
        CDomain = self.sequence.chromosomes['C'].GetDomainLineSpace()

        V = VehicleSuspensionSystem.V
        R = VehicleSuspensionSystem.R

        Mu = self.sequence.chromosomes['Mu'].GetValue()
        Ms = self.sequence.chromosomes['Ms'].GetValue()
        Kt = self.sequence.chromosomes['Kt'].GetValue()
        K = self.sequence.chromosomes['K'].GetValue()
        C = self.sequence.chromosomes['C'].GetValue()
        
        F = np.vectorize(VehicleSuspensionSystem.CalculateSprungMassAcceleration)
        plt.subplots_adjust(wspace = 0.3, hspace = 0.5)

        plt.subplot(3, 2, 1)
        plt.plot(MuDomain, F(V, R, MuDomain, Ms, Kt, K, C))

        plt.xlabel("Mu")
        plt.ylabel("a")

        plt.subplot(3, 2, 2)
        plt.plot(MsDomain, F(V, R, Mu, MsDomain, Kt, K, C))
        plt.xlabel("Ms")
        plt.ylabel("a")

        plt.subplot(3, 2, 3)
        plt.plot(KtDomain, F(V, R, Mu, Ms, KtDomain, K, C))
        plt.xlabel("Kt")
        plt.ylabel("a")

        plt.subplot(3, 2, 4)
        plt.plot(KDomain, F(V, R, Mu, Ms, Kt, KDomain, C))
        plt.xlabel("K")
        plt.ylabel("a")

        plt.subplot(3, 2, 5)
        plt.plot(CDomain, F(V, R, Mu, Ms, Kt, K, CDomain))
        plt.xlabel("C")
        plt.ylabel("a")

        plt.show()


#Top level container for the GA optimization process
class Solver:
    def __init__(self, environment, numberOfGenerations, populationSize):
        self.numberOfGenerations = numberOfGenerations
        self.environment = environment
        self.population = Population(environment, populationSize)
        self.bestFit = None

    def GetBestFit(self):
        return self.bestFit

    #Execute the GA process
    def Run(self):
        for i in range(self.numberOfGenerations):
            self.population.CalculateValues()
            self.population.CalculateFitness()
            self.population.Select()
            self.population.Spawn()
            self.population.Mutate()

            generationBestFit = self.population.GetBestFit()

            Mu = generationBestFit.chromosomes['Mu'].GetValue()
            Ms = generationBestFit.chromosomes['Ms'].GetValue()
            Kt = generationBestFit.chromosomes['Kt'].GetValue()
            K = generationBestFit.chromosomes['K'].GetValue()
            C = generationBestFit.chromosomes['C'].GetValue()

            print("Generation Best Fit i: %d, a: %f, Mu: %f, Ms: %f, Kt: %f, K: %f, C: %f" % (i, generationBestFit.GetValue(), Mu, Ms, Kt, K, C))
        
            if(self.bestFit == None or self.bestFit.GetValue() > generationBestFit.GetValue()):
                self.bestFit = generationBestFit

#Application entry point
def main():
    environment = Environment(MUTATION_RATE, SPAWN_RATE)
    solver = Solver(environment, GENERATIONS, POPULATION)

    solver.Run()

    bestFit = solver.bestFit
    Mu = bestFit.chromosomes['Mu'].GetValue()
    Ms = bestFit.chromosomes['Ms'].GetValue()
    Kt = bestFit.chromosomes['Kt'].GetValue()
    K = bestFit.chromosomes['K'].GetValue()
    C = bestFit.chromosomes['C'].GetValue()

    print("Overall Best Fit a: %f, Mu: %f, Ms: %f, Kt: %f, K: %f, C: %f" % (bestFit.GetValue(), Mu, Ms, Kt, K, C))

    plotter = SequencePlotter(solver.bestFit)
    plotter.PlotAll()


main()