import math

bestFitA = None
R = 30.0
V = 6.5e-6

def CalculateSprungMassAcceleration(R, V, Mu, Ms, Kt, K, C):
    return math.sqrt(math.pi * R * V  * ((Kt * C) / (2 * (Ms ** (3 / 2)) * (K ** (1 / 2))) + ((Mu + Ms) * K ** 2) / (2 * C * (Ms ** 2))))

for Mu in range(25, 40, 5):
    print("Mu: %f" % Mu)
    for Ms in range(400, 550, 25):
        for Kt in range(420000, 700000, 1000):
            for K in range(60000, 90000, 100):
                for C in range(1900, 3100, 100):
                    value = CalculateSprungMassAcceleration(R, V, float(Mu), float(Ms), float(Kt), float(K), float(C))
                    if(bestFitA == None or bestFitA > value):
                        bestFitA = value
                        print("a: %f, Mu: %f, Ms: %f, Kt: %f, K: %f, C: %f" % (bestFitA, Mu, Ms, Kt, K, C))
print("done")