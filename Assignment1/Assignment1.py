#Class to represent any trapaziodal or triangular identity function
class IdentityFunction:
    def __init__(self, name, a, m, n, b):
        self.name = name
        self.a = a
        self.m = m
        self.n = n
        self.b = b

    @classmethod
    def MakeSymmetricalTriangle(cls, name, a, b):
        m = (a + b) / 2
        return cls(name, a, m, m, b)

    @classmethod
    def MakeTriangle(cls, name, a, m, b):
        return cls(name, a, m, m, b)

    @classmethod
    def MakeTrapazoid(cls, name, a, m, n, b):
        return cls(name, a, m, n, b)

    def GetName(self):
        return self.name
    
    def Calculate(self, v):
        if v < self.a:
            return 0
        elif v < self.m:
            return (v - self.a) / (self.m / self.a)
        elif v < self.n:
            return 1
        elif v < self.b:
            return 1 - (v - self.b) / (self.n - self.b)
        else:
            return 0

#Class for operators related to fuzzy logic
class FuzzyOperator:
    @staticmethod
    def FuzzAnd(a, b):
        return max(a, b)

    @staticmethod 
    def FuzzOr(a, b):
        return min(a, b)

