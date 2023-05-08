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

num_simulations = 1

def main():
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    genomesInfo = []


    first_gens = []
    final_gens = []


    for i in range(num_simulations):
        # if i == 30:
        #     continue
        with open("data/test_noise/r" + str(i) +".json", "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        def convertStringKeysToIntKeys(d):
            keys = list(d.keys()) 
            for k in keys:
                d[int(k)] = d[k]
                d.pop(k, None)
            return d

        gen = 0
        total_fins_daughter = 0
        total_infs_daughter = 0 
        total_fins_new = 0
        total_infs_new = 0
    
        for v in genomesInfo[gen]["genome"].values():
            if v["dMR"] == 0:
                total_fins_daughter += 1
            else:
                total_infs_daughter += 1
            
            if v["nMR"] == 0:
                total_fins_new += 1
            else:
                total_infs_new += 1

        print("first gen")
        print(total_fins_daughter + total_fins_new)
        print(total_infs_daughter + total_infs_new)
        print("--------")

        infinite_first_generation = total_infs_new + total_infs_daughter
        finite_first_generation = total_fins_new + total_fins_daughter

        first_generation = 100 * (finite_first_generation / (finite_first_generation + infinite_first_generation))
        first_gens.append(first_generation)


        gen = -1
        total_fins_daughter = 0
        total_infs_daughter = 0 
        total_fins_new = 0
        total_infs_new = 0
    
        for v in genomesInfo[gen]["genome"].values():
            if v["dMR"] == 0:
                total_fins_daughter += 1
            else:
                total_infs_daughter += 1
            
            if v["nMR"] == 0:
                total_fins_new += 1
            else:
                total_infs_new += 1

        print("final gen")
        print(total_fins_daughter + total_fins_new)
        print(total_infs_daughter + total_infs_new)
        print("--------")

        infinite_final_generation = total_infs_new + total_infs_daughter
        finite_final_generation = total_fins_new + total_fins_daughter

        

        final_generation = 100 * (finite_final_generation / (finite_final_generation + infinite_final_generation))
        final_gens.append(final_generation)

    p_value = permutation_test(first_gens, final_gens,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)

    print(p_value)

    x = ["First Generation"] * num_simulations + ["Final Generation"] * num_simulations
    y = first_gens + final_gens
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'x': x, 'y': y})

    sns.swarmplot('x', 'y', data=df)
    sns.despine()
        
    plt.axhline(np.mean(y[:num_simulations]), color='blue', linewidth=2)
    plt.axhline(np.mean(y[num_simulations:]), color='orange', linewidth=2)
    # plt.title("Simulations Using Only Infinite Reservoirs Take Longer to Reach " + str(threshold) + " Percent of Max Fitness")
    plt.title("title")
    plt.ylabel("Percent Finite Fuel Used In Genome")
    # plt.savefig(SAVE_TO + "/" + "threshold_" + str(threshold) + ".png")
    plt.show()
    plt.clf()
        
    # xs = list(range(len(total_fins)))
    
    # fig = plt.figure()
    # ax = plt.axes()
    # ax.plot(xs, total_fins)
    # ax.plot(xs, total_infs)
    # ax.set(xlabel='Development Step', ylabel='Usage Per Step Of Development', title="title")
    # ax.legend(["Finite Usage", "Infinite Usage"])
    # ax.grid(True)
    # plt.savefig("res_usage_in_development/r" + str(i) + ".png")
    # plt.clf()



        # data = np.array(board.grid)

        # rows,cols = data.shape

        # plt.annotate('Finite Reservoir Value: ' + str(ceil(g.currentMetabolicReservoirValues[0])), 
        #         (40, 8), # these are the coordinates to position the label
        #         color='red')

        # plt.imshow(data, interpolation='none', 
        #                 extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
        #                 aspect="equal",
        #                 cmap=my_cmap)
        # plt.show()
        # # plt.savefig(FP + str(i) + '.png')
        # plt.clf()

    # f = fitness.Fitness(board=board)
    # f.calculate()
    # print("total score = ", f.totalScore)

if __name__ == "__main__":
    main()