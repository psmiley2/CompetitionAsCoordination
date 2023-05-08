import globalVars as globs
import random
import hyperParameters as hp
import json

class Cell:
    def __init__(self, g, cellType, metabolicReservoir, x,  y, threshold, signalAction, signalRadius):
        self.genome = g
        self.type = cellType
        self.metabolicReservoir = metabolicReservoir
        self.x = x
        self.y = y
        self.threshold = threshold
        self.signalAction = signalAction
        self.signalCount = 0
        self.signalRadius = signalRadius
    
    # returns the new cell so that board can update
    def divide(self, neighbors):
        freeNeighbors = neighbors["empty"]

        if self.metabolicReservoir < globs.FINITE_METABOLIC_RESERVOIR_COUNT:
            self.genome.updateReservoirHits("metabolic", "finite")
            # finite metabolic
            depleted = not self.genome.useMetabolicReservoir(self.metabolicReservoir)
            if random.random() < hp.reservoirAssessmentNoise:
                depleted = not depleted
        else:
            self.genome.updateReservoirHits("metabolic", "infinite")


        genomeOutput = self.genome.queryGenome(neighbors[globs.STEM], neighbors[globs.NERVE], neighbors[globs.SKIN])

        if hp.printLookupTableOutputs:
            def getDirectionFromXY(dbo):
                x, y = globs.DIRECTIONS[dbo]
                res = ""

                if y == -1:
                    res += "South"
                elif y == 1:
                    res += "North"

                if x == -1:
                    res += "West"
                elif x == 1:
                    res += "East"
                
                return res
            
            def getType(t):
                if t == 2:
                    return "stem"
                elif t == 3:
                    return "nerve"
                elif t == 4:
                    return "skin"

            with open("lookup_table_outputs_temp.json", 'a+') as outfile:
                out = {
                    "stem_neighbors": neighbors[globs.STEM],
                    "nerve_neighbors": neighbors[globs.NERVE],
                    "skin_neighbors": neighbors[globs.SKIN],
                    "daughter_type": getType(genomeOutput["dT"]),
                    "1": str(getDirectionFromXY(genomeOutput["dbo"][0])),
                    "2": str(getDirectionFromXY(genomeOutput["dbo"][1])),
                    "3": str(getDirectionFromXY(genomeOutput["dbo"][2])),
                    "new_res": str(genomeOutput["nMR"]),
                    "daughter_res": str(genomeOutput["dMR"])
                }
                print(json.dumps(out), file=outfile)
                outfile.close()       

        daughterDirection = (0, 0)
        dbo = genomeOutput["dbo"]
        for d in dbo:
            if globs.DIRECTIONS[d] in freeNeighbors:
                daughterDirection = globs.DIRECTIONS[d] 
                break

        self.metabolicReservoir = genomeOutput["nMR"]
        
        return {
            "daughter": Cell(
                self.genome, 
                genomeOutput["dT"],
                genomeOutput["dMR"], 
                self.x + daughterDirection[0], 
                self.y + daughterDirection[1],
                genomeOutput["thresh"],
                genomeOutput["SA"],
                genomeOutput["rad"]
            ),
        }