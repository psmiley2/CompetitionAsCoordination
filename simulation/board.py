from glob import glob
from pickle import BYTEARRAY8
import numpy as np
import globalVars as globs
import cell
import hyperParameters as hp
import matplotlib.pyplot as plt
import pylab
import random
import os


class Board:
    def __init__(self, w, h):
        self.height = h
        self.width = w
        # self.grid = [[0 for i in range(self.width)] for j in range(self.height)]
        self.grid = [[None for i in range(self.width)] for j in range(self.height)]
        self.cells = []
        self.avoidSpots = 0
        self.targetSpots = 0 
        self.dynamicCells = set()

    def reset(self, genome):
        self.avoidSpots = 0
        self.targetSpots = 0
        self.grid = [[None for i in range(self.width)] for j in range(self.height)]
        # for s in hp.targetShapeSpecification: TARGET SHAPE STUFF
        #     for i in range(s["E"], s["W"] + 1):
        #         for j in range(s["N"], s["S"] + 1):
        #             self.grid[j][i] = 1
        # for row in self.grid:
        #     for c in row:
        #         if c == 0:
        #             self.avoidSpots += 1
        #         elif c == 1:
        #             self.targetSpots += 1

        if os.path.exists("lookup_table_outputs_temp.txt"):
           os.remove("lookup_table_outputs_temp.txt")

        cellInfo = genome.queryGenome(0, 0, 0)

        rootCell = cell.Cell(genome, globs.STEM, cellInfo["nMR"], hp.startY, hp.startX, abs(round(np.random.normal(5, 5))), random.choice([0, 1, 2, 3]), abs(round(np.random.normal(0, 2))))
        self.addCell(rootCell)
    
    def sendSignals(self, x, y, r):
        for i in range(x - r, x + r):
            for j in range(y - r, y + r):
                if i == x and j == y:
                    # don't send signal to self
                    continue

                try:
                    self.grid[i][j].signalCount += 1
                    if self.grid[i][j].signalCount >= self.grid[i][j].threshold:
                        self.grid[i][j].signalActionTriggered()
                except:
                    pass

    
    def addCell(self, cell):
        self.grid[cell.x][cell.y] = cell
        if cell.type == globs.STEM:
            self.dynamicCells.add(cell)
    
    def step(self):
        cellList = list(self.dynamicCells)
        random.shuffle(cellList)
        for cell in cellList:
            # cells divide and send signals
            neighbors = self.getNeighbors(cell.x, cell.y)
            if len(neighbors["empty"]) > 0:
                new_cell = cell.divide(neighbors)
                self.addCell(new_cell["daughter"])
            if hp.useSignals:
                self.sendSignals(cell.x, cell.y, cell.signalRadius)

        if hp.useSignals:    
            for cell in cellList:
                # cells with signal counts above threshold act 
                if cell.signalCount > cell.threshold:
                    cell.signalCount = 0
                    if cell.signalAction == 0:
                        # divide
                        neighbors = self.getNeighbors(cell.x, cell.y)
                        if len(neighbors["empty"]) > 0:
                            new_cell = cell.divide(neighbors)
                            self.addCell(new_cell["daughter"])
                    elif cell.signalAction == 1:
                        # stop growing
                        try:
                            self.dynamicCells.remove(cell)
                        except:
                            pass
                    elif cell.signalAction == 2:
                        # change reservoir
                        if hp.onlyFinite or hp.onlyInfinite:
                            continue

                        metabolic_reservoirs = {0, 1}
                        metabolic_reservoirs.remove(cell.metabolicReservoir)
                        cell.metabolicReservoir = list(metabolic_reservoirs).pop()
                    elif cell.signalAction == 3:
                        # change type
                        if cell.type == globs.STEM:
                            self.dynamicCells.remove(cell)

                        cell_types = {globs.STEM, globs.SKIN, globs.NERVE}
                        cell_types.remove(cell.type)
                        cell.type = random.choice(list(cell_types))
                        
                        if cell.type == globs.STEM:
                            self.dynamicCells.add(cell)

                        cell.signalAction = random.choice([0, 1, 2, 3])
                        
        # if hp.printBoard:
        #     # pylab.ion()
        #     line = pylab.plot(0, 1, "ro", markersize=6)
        #     types_grid = [[0 if c is None else c.type for c in row] for row in self.grid]
        #     data = np.array(types_grid)
        #     rows,cols = data.shape
        #     plt.imshow(data, interpolation='none', 
        #                     extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
        #                     aspect="equal")
        #     plt.show()

    
    def getNeighbors(self, x, y):
        previous_neighbor_type = random.choice([None, globs.STEM, globs.NERVE, globs.SKIN])
        neighbors = {
            globs.STEM: 0, 
            globs.NERVE: 0,
            globs.SKIN: 0,
            "empty": []
        }
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue

                try:
                    # bad hack which throws error when doesn't exist - edges of grid
                    self.grid[x + i][y + j]
                except:
                    continue

                if random.random() < hp.neighborAssessmentNoise:
                    if previous_neighbor_type == None:
                        neighbors["empty"].append((i, j))
                    else:
                        neighbors[previous_neighbor_type] += 1
                else:
                    if self.grid[x + i][y + j] is None:
                        neighbors["empty"].append((i, j))
                    else:
                        neighbors[self.grid[x + i][y + j].type] += 1

                    if self.grid[x + i][y + j] is None:
                        previous_neighbor_type = None
                    else:
                        previous_neighbor_type = self.grid[x + i][y + j].type

        return neighbors