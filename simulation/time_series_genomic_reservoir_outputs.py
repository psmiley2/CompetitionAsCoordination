from typing import final
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
from math import ceil
import pandas as pd
from mlxtend.evaluate import permutation_test


# construct cmap
flatui = [ "#FFFFFF", "#FFFFFF", "#000000", "#FF0000", "#00FF00"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

hp.onlyInfinite = False
hp.targetAspectRatio = 5
hp.targetSize = 400
hp.useConsistency = False
hp.onlyFinite = False
hp.useReservoirsAsInputs = False
hp.onlyUseSizeAsFitness = False
hp.useMidpointsForAspectRatio = True

num_simulations = 50
num_gens = 1501

infinite_list = [] 
finite_list = []

def main():
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    genomesInfo = []


    first_gens = []
    final_gens = []


    for i in range(num_simulations):
        # if i == 30:
        #     continue
        with open("data/start_5_mutate_5_take_2/r" + str(i) +".json", "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        
        for g_idx in range(num_gens):
            total_fins_daughter = 0
            total_infs_daughter = 0 
            total_fins_new = 0
            total_infs_new = 0
        
            for v in genomesInfo[g_idx]["genome"].values():
                if v["dMR"] == 0:
                    total_fins_daughter += 1
                else:
                    total_infs_daughter += 1
                
                if v["nMR"] == 0:
                    total_fins_new += 1
                else:
                    total_infs_new += 1


            try:
                infinite_list[g_idx].append(total_infs_new + total_infs_daughter)
            except:
                infinite_list.append([total_infs_new + total_infs_daughter])

            try:   
                finite_list[g_idx].append(total_fins_new + total_fins_daughter) 
            except:
                finite_list.append([total_fins_new + total_fins_daughter])


        # first_generation = 100 * (finite_first_generation / (finite_first_generation + infinite_first_generation))
        # first_gens.append(first_generation)


    x = list(range(num_gens))
    ax = plt.axes()
    ax.plot(x, [np.mean(arr) for arr in finite_list])
    ax.plot(x, [np.mean(arr) for arr in infinite_list])
        
    plt.title("title")
    plt.ylabel("Number Of Genome Outputs")
    ax.legend(["Finite", "Infinite"])
    # plt.savefig(SAVE_TO + "/" + "threshold_" + str(threshold) + ".png")
    plt.show()
    plt.clf()
    

if __name__ == "__main__":
    main()