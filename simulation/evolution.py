import pylab
import genome
import matplotlib.pyplot as plt
import fitness
import hyperParameters as hp
from statistics import mean
import operator
import math
import copy
import json
import os
import numpy as np


class Evolution:
    def __init__(self, board, filepath):
        self.board = board
        self.reruns = {
            "total": [],
            "cellRatio": [], 
            "aspectRatio": [],
            "size": []
        }
        self.genePool = []
        self.currentAgentNumber = 0
        self.createRandomGenePool()
        self.currentGenome = self.genePool[0]
        self.currentGeneration = 0
        self.simulationDone = False
        self.filepath = filepath
        
        with open(self.filepath, 'a') as outfile:
            print("{\"genomes\": [", file=outfile)
            outfile.close()       
        
    def run(self):
        self.board.reset(self.currentGenome)
        while (not self.simulationDone):
            c = 0
            while (len(self.board.dynamicCells)):
                self.board.step()
                c += 1
                if c == 100:
                    self.board.dynamicCells = set()
            
            self.handleAgentFinished()

    def createRandomGenePool(self):
        for _ in range(hp.populationSize):
            self.genePool.append(genome.Genome(genome={}, initialMetabolicReservoirValues={}, initialInformationalReservoirValues={}))

    def newGeneration(self):
        # sort genepool by fitness in place 
        self.genePool.sort(key=operator.attrgetter("totalFitness"), reverse=True)
        survivors = self.genePool[0 : math.ceil((hp.survivalRate / 10) * hp.populationSize)]
        nextGenePool = [survivors[0]] # top genome from the last generation gets passed in without mutation

        self.reportStats(survivors[0])
        self.storeData(survivors[0])

        if hp.saveAllRunsFromFirstAndLastGenerations:
            if self.currentGeneration == 0 or self.currentGeneration == hp.maxGenerations - 1:
                self.storePopulation(self.genePool)

        for g in survivors:
            g.fillReservoirs()
            for i in range(4):
                g.resHits[i] = {
                    "metFin": 0,
                    "metInf": 0,
                    "infFin": 0,
                    "infInf": 0
                }
        
        if self.currentGeneration == hp.maxGenerations:
            self.simulationDone = True

        # populate next generation with mutated versions of the survivors
        for i in range(hp.populationSize - 1):
            clone = copy.deepcopy(survivors[int(i / hp.survivalRate)])
            clone.mutate()
            nextGenePool.append(clone)

        self.genePool = nextGenePool
        self.currentAgentNumber = 0
        self.currentGeneration += 1
    
    def handleAgentFinished(self):
        if hp.printBoard:
            line = pylab.plot(0, 1, "ro", markersize=6)
            types_grid = [[0 if c is None else c.type for c in row] for row in self.board.grid]
            data = np.array(types_grid)
            rows,cols = data.shape
            plt.imshow(data, interpolation='none', 
                            extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                            aspect="equal")
            plt.show()

        f = fitness.Fitness(self.board)
        f.calculate()
        self.reruns["total"].append(f.getTotalScore())
        self.reruns["cellRatio"].append(f.cellRatioScore)
        self.reruns["aspectRatio"].append(f.aspectRatioScore)
        self.reruns["size"].append(f.sizeScore)
        self.currentGenome.fillReservoirs() 
        if len(self.reruns["total"]) == hp.rerunsPerAgent:
            m = mean(self.reruns["total"])
            mCell = mean(self.reruns["cellRatio"])
            mAspect = mean(self.reruns["aspectRatio"])
            mSize = mean(self.reruns["size"])
            self.currentGenome.setTotalFitness(m)
            if hp.useConsistency:
                self.currentGenome.setTotalFitness(m - np.std(self.reruns["total"])) 
            else:
                self.currentGenome.setTotalFitness(m)
            self.currentGenome.setCellRatioFitness(mCell)
            self.currentGenome.setAspectRatioFitness(mAspect)
            self.currentGenome.setSizeFitness(mSize)
            self.nextGenome()
        self.board.reset(self.currentGenome)

    def nextGenome(self):
        self.reruns = {
            "total": [],
            "cellRatio": [], 
            "aspectRatio": [],
            "size": []
        }
        self.currentAgentNumber += 1
        if self.currentAgentNumber == hp.populationSize:
            self.newGeneration()
        self.currentGenome = self.genePool[self.currentAgentNumber]
    
    def reportStats(self, ts):
        print("Current Generation #: ", self.currentGeneration)
        print("Top Fitness Score: ", ts.totalFitness)
        print("------------------")

    def storeData(self, ts):
        for phase in ts.resHits:
            for k, v in phase.items():
                phase[k] = v / hp.rerunsPerAgent

        data = {
            "generation": self.currentGeneration,
            "fitness": ts.totalFitness,
            "cellRatioFitness": ts.cellRatioFitness,
            "aspectRatioFitness": ts.aspectRatioFitness,
            "sizeFitness": ts.sizeFitness,
            "resHits": ts.resHits,
            "metabolicReservoirValues": ts.initialMetabolicReservoirValues,
            "genome": ts.genome,
        }

        with open(self.filepath, 'a') as outfile:
            json.dump(data, outfile)
            if self.currentGeneration == hp.maxGenerations:
                print("\n]}", file=outfile)
                outfile.close()       
            else:
                print(",", file=outfile)
            outfile.close()

    def storePopulation(self, genePool):
        path = self.filepath[:-5] + "_" + str(self.currentGeneration) + ".json"
        print(path)

        with open(path, 'a') as outfile:
            print("{\"genomes\": [", file=outfile)
            outfile.close()       

        for g in genePool:
            for phase in g.resHits:
                for k, v in phase.items():
                    phase[k] = v / hp.rerunsPerAgent

            data = {
                "generation": self.currentGeneration,
                "fitness": g.totalFitness,
                "cellRatioFitness": g.cellRatioFitness,
                "aspectRatioFitness": g.aspectRatioFitness,
                "sizeFitness": g.sizeFitness,
                "resHits": g.resHits,
                "metabolicReservoirValues": g.initialMetabolicReservoirValues,
                "genome": g.genome,
            }

            with open(path, 'a') as outfile:
                json.dump(data, outfile)
                print(",", file=outfile)
                outfile.close()
        
        with open(path, 'a') as outfile:
            json.dump(data, outfile)
            print("\n]}", file=outfile)
            outfile.close()   

