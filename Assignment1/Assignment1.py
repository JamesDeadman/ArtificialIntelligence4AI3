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
    
    def GetMidPoint(self):
        return (self.m + self.n) / 2

    def Calculate(self, v):
        if v < self.a:
            return 0
        elif v < self.m:
            return (v - self.a) / (self.m - self.a)
        elif v < self.n:
            return 1
        elif v < self.b:
            return (self.b - v) / (self.b - self.n)
        else:
            return 0

#Class for operators related to fuzzy logic
class FuzzyOperator:
    @staticmethod
    def FuzzAnd(a, b):
        return min(a, b)

    @staticmethod 
    def FuzzOr(a, b):
        return max(a, b)


#Use fuzzy logic to determine the valve position
def GetValvePosition(EggDirt, FlowRate):
    # Define the membership functions for the Egg Wash system
    EggDirtLow = IdentityFunction.MakeSymmetricalTriangle("Low", 0.0, 0.5)
    EggDirtAcceptable = IdentityFunction.MakeSymmetricalTriangle("Acceptable", 0.25, 0.75)
    EggDirtHigh = IdentityFunction.MakeSymmetricalTriangle("High", 0.5, 1.0)

    FlowRateLow = IdentityFunction.MakeSymmetricalTriangle("Low", 0.0, 0.75)
    FlowRateHigh = IdentityFunction.MakeSymmetricalTriangle("High", 0.25, 1.0)

    ValvePositionLow = IdentityFunction.MakeSymmetricalTriangle("Low", 0.0, 0.5)
    ValvePositionMedium = IdentityFunction.MakeSymmetricalTriangle("Medium", 0.25, 0.75)
    ValvePositionHigh = IdentityFunction.MakeSymmetricalTriangle("High", 0.5, 1.0)

    # Step 1 Fuzzification
    #DOM - Degree of membership
    EggDirtLowDOM = EggDirtLow.Calculate(EggDirt)
    EggDirtAcceptableDOM = EggDirtAcceptable.Calculate(EggDirt)
    EggDirtHighDOM = EggDirtHigh.Calculate(EggDirt)

    print("Egg Dirt Low DOM: %.2f" % EggDirtLowDOM)
    print("Egg Dirt Acceptable DOM: %.2f" % EggDirtAcceptableDOM)
    print("Egg Dirt High DOM: %.2f" % EggDirtHighDOM)

    FlowRateLowDOM = FlowRateLow.Calculate(FlowRate)
    FlowRateHighDOM = FlowRateHigh.Calculate(FlowRate)

    print("Flow Rate Low DOM: %.2f" % FlowRateLowDOM)
    print("Flow Rate High DOM: %.2f" % FlowRateHighDOM)

    #Aggregation
    PositionLow1 = FuzzyOperator.FuzzAnd(EggDirtLowDOM, FlowRateLowDOM)
    PositionLow2 = FuzzyOperator.FuzzAnd(EggDirtLowDOM, FlowRateHighDOM)
    PositionMedimum1 = EggDirtAcceptableDOM
    PositionHigh1 = FuzzyOperator.FuzzAnd(EggDirtHighDOM, FlowRateLowDOM)
    PositionHigh2 = FuzzyOperator.FuzzAnd(EggDirtHighDOM, FlowRateHighDOM)

    #Composition
    PositionLowDOM = FuzzyOperator.FuzzOr(PositionLow1, PositionLow2)
    PositionMedimumDOM = PositionMedimum1
    PositionHighDOM = FuzzyOperator.FuzzOr(PositionHigh1, PositionHigh2)

    print("Valve Position Low DOM: %.2f" % PositionLowDOM)
    print("Valve Position Medium DOM: %.2f" % PositionMedimumDOM)
    print("Valve Position High DOM: %.2f" % PositionHighDOM)

    #Defuzzification
    ValvePosition = PositionLowDOM * ValvePositionLow.GetMidPoint() + PositionMedimumDOM * ValvePositionMedium.GetMidPoint() + PositionHighDOM * ValvePositionHigh.GetMidPoint()
    
    return ValvePosition


#Values from input
EggDirt = 0.5
FlowRate = 0.2

print("Input Egg Dirt: %.2f" % EggDirt)
print("Input Flow Rate: %.2f" % FlowRate)
print()

ValvePosition = GetValvePosition(EggDirt, FlowRate)

print()
print("Output Valve Position: %.2f" % ValvePosition)
