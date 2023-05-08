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

genome_output_finite_counts = {}

def main():
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    genomesInfo = []


    first_gens = []
    final_gens = []


    for i in range(num_simulations):
        # if i == 30:
        #     continue
        with open("data/fixed_finite_5/r" + str(i) +".json", "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        def convertStringKeysToIntKeys(d):
            keys = list(d.keys()) 
            for k in keys:
                d[int(k)] = d[k]
                d.pop(k, None)
            return d

        gen = -1
        total_fins_daughter = 0
        total_infs_daughter = 0 
        total_fins_new = 0
        total_infs_new = 0

        for k, v in genomesInfo[gen]["genome"].items():
            if v["dMR"] == 0 or v["nMR"] == 0:
                try:
                    genome_output_finite_counts[k] += 1
                except:
                    genome_output_finite_counts[k] = 1


    d = dict(sorted(genome_output_finite_counts.items(), key=lambda item: item[1]))
    for k, v in d.items():
        print(k,v)

    

    print("mean: ", np.mean(list(d.values())))
    print("median: ", np.median(list(d.values())))

    # x = ["First Generation"] * num_simulations + ["Final Generation"] * num_simulations
    # y = first_gens + final_gens
    # sns.set(style='ticks', context='talk')
    # df= pd.DataFrame({'x': x, 'y': y})

    # sns.swarmplot('x', 'y', data=df)
    # sns.despine()
        
    # plt.axhline(np.mean(y[:num_simulations]), color='blue', linewidth=2)
    # plt.axhline(np.mean(y[num_simulations:]), color='orange', linewidth=2)
    # # plt.title("Simulations Using Only Infinite Reservoirs Take Longer to Reach " + str(threshold) + " Percent of Max Fitness")
    # plt.title("title")
    # plt.ylabel("Percent Finite Fuel Used In Genome")
    # # plt.savefig(SAVE_TO + "/" + "threshold_" + str(threshold) + ".png")
    # plt.show()
    # plt.clf()

if __name__ == "__main__":
    main()