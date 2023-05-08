import os
from PIL.Image import SAVE
import cell 
import globalVars as globs
import fitness as fit
import evolution as ev
import board as brd
import hyperParameters as hp
import genome
import json
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from matplotlib.colors import ListedColormap
import fitness
from scipy.stats import ttest_ind

# construct cmap
flatui = [ "#FFFFFF", "#FFFFFF", "#000000", "#FF0000", "#00FF00"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

def main():
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    genomesInfo = []

    def convertStringKeysToIntKeys(d):
        keys = list(d.keys()) 
        for k in keys:
            d[int(k)] = d[k]
            d.pop(k, None)
        return d

    xs = []
    top_fitness_scores = []

    for s_idx in range(1):
        FP = "data/test_noise/r" + str(s_idx) + ".json"
        with open(FP, "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        top_fitness_scores.append(genomesInfo[-1]["fitness"])

    max_fitness_score = max(top_fitness_scores)

    for i, score in enumerate(top_fitness_scores):
        if score == max_fitness_score:
            FP = "data/base/r" + str(i) + ".json"
            print("Best agent from simulation: ", FP)
            with open(FP, "r") as json_file:
                data = json.load(json_file)
                genomesInfo = data["genomes"]

                g = genome.Genome(genomesInfo[-1]["genome"], 
                    convertStringKeysToIntKeys(genomesInfo[-1]["metabolicReservoirValues"]), 
                    genomesInfo[-1]["fitness"],  
                )

                received_score = 0 
                while received_score < max_fitness_score: 
                    board.reset(g)
                    g.fillReservoirs()
                    while (len(board.dynamicCells)):
                        board.step()

                    f = fitness.Fitness(board=board)
                    f.calculate()
                    print("total score = ", f.totalScore)
                    received_score = f.totalScore
                
            data = np.array(board.grid)

            rows,cols = data.shape

            plt.imshow(data, interpolation='none', 
                            extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                            aspect="equal",
                            cmap=my_cmap)

            plt.show()

if __name__ == "__main__":
    main()