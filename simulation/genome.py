import globalVars as globs
import random 
import copy
import hyperParameters as hp
import math
import numpy as np 

class Genome:
    def __init__(
        self, 
        genome = {},
        initialMetabolicReservoirValues = {}, 
        totalFitness = 0, 
    ):
        self.genome = genome
        self.initialMetabolicReservoirValues = initialMetabolicReservoirValues
        if len(self.initialMetabolicReservoirValues) == 0:
            self.initialMetabolicReservoirValues = {
                0: random.randint(0, globs.MAX_INITIAL_RESERVOIR_VALUE ),
            }
        self.totalFitness = totalFitness
        self.aspectRatioFitness = 0
        self.cellRatioFitness = 0
        self.sizeFitness = 0

        self.resHits = []
        for i in range(4):
            self.resHits.append(
                {
                    "metFin": 0,
                    "metInf": 0,
                    "infFin": 0,
                    "infInf": 0
                }
            )

        self.currentMetabolicReservoirValues = copy.deepcopy(self.initialMetabolicReservoirValues)

        self.lookupTableOutputs = []

    def queryGenome(self, stemNeighbors, nerveNeighbors, skinNeighbors):
        k = ""
        m = "" 
        for mr in self.currentMetabolicReservoirValues:
            if mr == 0:
                m += "0"
            else:
                m += "1"

        if hp.useReservoirsAsInputs:
            k = str(stemNeighbors) + str(nerveNeighbors) + str(skinNeighbors) + m
        else:
            k = str(stemNeighbors) + str(nerveNeighbors) + str(skinNeighbors)

        r = random.random()
        if r < hp.genomeLookupNoise / 2:
            k = str(int(k) - 1).replace('9', '8')
        elif r < hp.genomeLookupNoise:
            k = str(int(k) + 1).replace('9', '0')

        if k not in self.genome:
           self.genome[k] = self.generateRandomOutcome()
    
        return self.genome[k]

    def generateRandomOutcome(self):
        directionalBiasOrder = list(range(8))
        random.shuffle(directionalBiasOrder)
        if hp.onlyFinite:
            return {
                "nMR": 0,
                "dMR": 0,
                "dbo": directionalBiasOrder,
                "dT": random.choice([globs.STEM, globs.SKIN, globs.NERVE]),
                "thresh": abs(round(np.random.normal(5, 5))),
                "SA": random.choice([0, 1, 2, 3]),
                "rad": abs(round(np.random.normal(0, 2)))
            }
        else:
            return {
                "nMR": random.randint(0, len(self.initialMetabolicReservoirValues) * 2 - 1),
                "dMR": random.randint(0, len(self.initialMetabolicReservoirValues) * 2 - 1),
                "dbo": directionalBiasOrder,
                "dT": random.choice([globs.STEM, globs.SKIN, globs.NERVE]),
                "thresh": abs(round(np.random.normal(5, 5))),
                "SA": random.choice([0, 1, 2, 3]),
                "rad": abs(round(np.random.normal(0, 2)))
            }
        
    def fillReservoirs(self):
        self.currentMetabolicReservoirValues = copy.deepcopy(self.initialMetabolicReservoirValues)
    
    def useMetabolicReservoir(self, r):
        if hp.onlyInfinite:
            return True

        if self.currentMetabolicReservoirValues[r] > 0:
            self.currentMetabolicReservoirValues[r] -= 1
            return True
        else:
            return False
    
    def updateReservoirHits(self, purpose, limit):
        resType = ""
        if purpose == "metabolic":
            resType = "met"
        else:
            resType = "inf"

        if limit == "finite":
            resType += "Fin"
        else:
            resType += "Inf"

        self.resHits[self.getPhase()][resType] += 1

        
    def mutate(self):
        if random.randint(0, 100) < hp.reservoirMutationPercentage:
            # if random.randint(0, 100) < 50:
            resToMutate = random.choice(list(self.initialMetabolicReservoirValues.keys()))
            self.initialMetabolicReservoirValues[resToMutate] += int(np.random.normal(0, 50, 1)[0])
            # else:
                # resToMutate = random.choice(list(self.initialInformationalReservoirValues.keys()))
                # self.initialInformationalReservoirValues[resToMutate] = random.randint(0, globs.MAX_INITIAL_RESERVOIR_VALUE)
        else:
            numMutations = math.ceil(hp.genomeMutationRate / 100 * len(self.genome))
            for _ in range(numMutations):
                keyToMutate = random.choice(list(self.genome.keys()))
                self.genome[keyToMutate] = self.generateRandomOutcome()

    def setTotalFitness(self, f):
        self.totalFitness = f
        if self.totalFitness < 0:
            self.totalFitness = 0
    
    def setAspectRatioFitness(self, f):
        self.aspectRatioFitness = f
    
    def setCellRatioFitness(self, f):
        self.cellRatioFitness = f

    def setSizeFitness(self, f):
        self.sizeFitness = f