import genome
import fitness
import hyperParameters as hp
from statistics import mean
import operator
import math
import copy
import json
import numpy as np 
import os
from copy import deepcopy

class Evolution:
    def __init__(self, board, filepath):
        def convertStringKeysToIntKeys(d):
            keys = list(d.keys()) 
            for k in keys:
                d[int(k)] = d[k]
                d.pop(k, None)
            return d

        self.board = board
        self.reruns = {
            "total": [],
            "cellRatio": [], 
            "aspectRatio": [],
            "size": []
        }
        self.genePool = []
        self.currentAgentNumber = 0
        self.simulationDone = False
        self.filepath = filepath

        
        with open(filepath, "r") as f:
            for line in f:
                pass
            last_line = line
        
        # simulations that are done with throw an error here and quit
        genome_data = json.loads(last_line.strip()[:-1].strip())
        
        self.currentGeneration = genome_data["generation"] + 1

        g = genome.Genome(genome_data["genome"], 
            convertStringKeysToIntKeys(genome_data["metabolicReservoirValues"]), 
            genome_data["fitness"],  
        )

        for _ in range(hp.populationSize):
            self.genePool.append(deepcopy(g))

        self.currentGenome = self.genePool[0]
    
    def run(self):
        self.board.reset(self.currentGenome)
        while (not self.simulationDone):
            while (len(self.board.dynamicCells)):
                self.board.step()
            self.handleAgentFinished()

    def newGeneration(self):
        # sort genepool by fitness in place 
        self.genePool.sort(key=operator.attrgetter("totalFitness"), reverse=True)
        survivors = self.genePool[0 : math.ceil((hp.survivalRate / 10) * hp.populationSize)]
        nextGenePool = [survivors[0]] # top genome from the last generation gets passed in without mutation

        self.reportStats(survivors[0])
        self.storeData(survivors[0])

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
