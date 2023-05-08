from copy import deepcopy
import os
from types import TracebackType
import numpy as np
from PIL.Image import SAVE
from numpy.core.fromnumeric import std
from numpy.lib.index_tricks import s_
from scipy.fft import ihfft
from scipy.stats.stats import trim_mean
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
import os
from scipy.stats import ttest_ind
import pandas as pd 
from math import ceil
from mlxtend.evaluate import permutation_test

flatui = [ "#FFFFFF", "#FFFFFF", "#000000", "#FF0000", "#00FF00"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

def main():

    def updateHyperParametersF1():
        hp.onlyInfinite = False
        hp.targetAspectRatio = 5
        hp.targetSize = 400
        hp.useConsistency = False
        hp.onlyFinite = False
        hp.useReservoirsAsInputs = False
        hp.onlyUseSizeAsFitness = False
        hp.useMidpointsForAspectRatio = True

    FP1 = "inf_or_fin"

    updateHyperParametersF1()

    FP = "data/" + FP1 + "/r" + str(0) + ".json"
    with open(FP, "r") as json_file:
        data = json.load(json_file)
        genomesInfo1 = data["genomes"]

    def convertStringKeysToIntKeys(d):
        keys = list(d.keys()) 
        for k in keys:
            d[int(k)] = d[k]
            d.pop(k, None)
        return d

    g = genome.Genome(genomesInfo1[-1]["genome"], 
        convertStringKeysToIntKeys(genomesInfo1[-1]["metabolicReservoirValues"]), 
        genomesInfo1[-1]["fitness"],  
    )

    board = brd.Board(hp.boardWidth, hp.boardHeight)

    board.reset(g)
    g.fillReservoirs()
    while (len(board.dynamicCells)):
        board.step()

    f = fitness.Fitness(board=board)
    f.calculate()
    score = f.totalScore

    data = np.array(board.grid)

    rows,cols = data.shape

    plt.imshow(data, interpolation='none', 
                    extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                    aspect="equal",
                    cmap=my_cmap)
    
    plt.title("test")

    plt.show()
    plt.clf()
    

if __name__ == "__main__":
    main()