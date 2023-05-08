from copy import deepcopy
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
import pandas as pd 
from mlxtend.evaluate import permutation_test

# construct cmap
flatui = [ "#FFFFFF", "#FFFFFF", "#000000", "#FF0000", "#00FF00"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())


NUMBER_OF_SIMULATIONS = 100

hp.onlyInfinite = False
hp.targetAspectRatio = 5
hp.targetSize = 400
hp.useConsistency = False
hp.onlyFinite = False
hp.useReservoirsAsInputs = False
hp.onlyUseSizeAsFitness = False
hp.useMidpointsForAspectRatio = True

aspect_ratio_before = []
aspect_ratio_after = []


def main():
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    genomesInfo = []

    hp.punishForHittingBorder = True

    def convertStringKeysToIntKeys(d):
        keys = list(d.keys()) 
        for k in keys:
            d[int(k)] = d[k]
            d.pop(k, None)
        return d

    xs = []
    ys = []
    xs_f = []
    ys_f = []

    ratios = []


    for s_idx in range(NUMBER_OF_SIMULATIONS):
        # os.mkdir("before_after/" + str(s_idx))
    # for s_idx in [11, 47, 61, 77, 78, 80, 95, 98]:
    # for s_idx in [47]: #, 47, 61, 77, 78, 80, 95, 98]:
        FP = "data/inf_or_fin/r" + str(s_idx) + ".json"
        with open(FP, "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        hp.onlyInfinite = False

        g = genome.Genome(genomesInfo[-1]["genome"], 
            convertStringKeysToIntKeys(genomesInfo[-1]["metabolicReservoirValues"]), 
            genomesInfo[-1]["fitness"],  
        )


        avg_scores_before = []
        avg_aspect_ratio_before = []
        for i in range(10):
            board.reset(g)
            g.fillReservoirs()
            while (len(board.dynamicCells)):
                board.step()

            f = fitness.Fitness(board=board)
            f.calculate()
        #     print(str(i),  f.totalScore)
            avg_scores_before.append(f.totalScore)
            avg_aspect_ratio_before.append(f.aspectRatioScore)




        xs.append(np.mean(avg_scores_before))
        aspect_ratio_before.append(np.mean(avg_aspect_ratio_before))

        # b1 = deepcopy(board)

        # -------
        # g = genome.Genome(genomesInfo[0]["genome"], 
        #     convertStringKeysToIntKeys(genomesInfo[0]["metabolicReservoirValues"]), 
        #     genomesInfo[0]["fitness"],  
        # )

        # board.reset(g)
        # c = 0
        # while (len(board.dynamicCells)) and c < 2000:
        #     board.step()
        #     c += 1

        # data = np.array(board.grid)

        # rows,cols = data.shape

        # f = fitness.Fitness(board)
        # score = f.totalScore
        

        # plt.imshow(data, interpolation='none', 
        #                 extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
        #                 aspect="equal",
        #                 cmap=my_cmap)

        # plt.savefig("before_after/" + str(s_idx) + "/" + "original.jpg")
        # plt.show()

        # f = fitness.Fitness(board=board)
        # print("before right: ", f.rightCell)
        # f.calculate()
        # xs_f.append(f.totalScore)

        # -------------------------


        hp.onlyInfinite = True

        g = genome.Genome(genomesInfo[-1]["genome"], 
            convertStringKeysToIntKeys(genomesInfo[-1]["metabolicReservoirValues"]), 
            genomesInfo[-1]["fitness"],  
        )


        avg_scores_after = []
        avg_aspect_ratio_after = []
        for i in range(10):
            board.reset(g)
            g.fillReservoirs()

            c = 0 
            while (len(board.dynamicCells)) and c < 2000:
                board.step()
                c += 1

            f = fitness.Fitness(board=board)
            f.calculate()
            avg_scores_after.append(f.totalScore)
            avg_aspect_ratio_after.append(f.aspectRatioScore)

        print(np.mean(avg_scores_after))
        ys.append(np.mean(avg_scores_after))
        aspect_ratio_after.append(np.mean(avg_aspect_ratio_after))


        

        # b2 = deepcopy(board)

        # -------------


        # g = genome.Genome(genomesInfo[0]["genome"], 
        #     convertStringKeysToIntKeys(genomesInfo[0]["metabolicReservoirValues"]), 
        #     genomesInfo[0]["fitness"],  
        # )

        # board.reset(g)
        # while (len(board.dynamicCells)):
        #     board.step()

        # data = np.array(board.grid)

        # rows,cols = data.shape

        # f = fitness.Fitness(board)
        # score = f.totalScore

        # plt.imshow(data, interpolation='none', 
        #                 extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
        #                 aspect="equal",
        #                 cmap=my_cmap)

        # plt.savefig("before_after/" + str(s_idx) + "/" + "infinite.jpg")
        # plt.show()

        # f = fitness.Fitness(board=board)
        # f.calculate()
        # # print("total score = ", f.totalScore)
        # print("after right: ", f.rightCell)
        # print(f.totalScore)
        # ys_f.append(f.totalScore)
        
        # pixel_similarity(b1, b2)
    
    
    # for i in range(len(xs)):
    #     if xs[i] - ys[i] < 30: 
    #         print (i)

    print("with punishment")
    compute_permutation_test(xs, ys)
    # plot_scatter_2D(xs, ys)
    # plot_relative_drop_1D(xs, ys)
    # plot_2_columns_before_after_drop(xs, ys)
    # plot_relative_drop_1D_first_and_last_generation(xs_f, ys_f, xs, ys)
    # ebryonic_lethal(ys)

    p_value = permutation_test(xs, ys,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)

    print("before after switch = ", p_value)

def ebryonic_lethal(ys):
    dead = []
    survived = []

    for i in range(NUMBER_OF_SIMULATIONS):
        if ys[i] < 100:
            dead.append(aspect_ratio_before[i] - aspect_ratio_after[i])
        else:
            survived.append(aspect_ratio_before[i] - aspect_ratio_after[i])

    print(survived)
    print(dead)
    print(aspect_ratio_before)
    print(aspect_ratio_after)

    p_value = permutation_test(survived, dead,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)

    print("embryonic lethal = ", p_value)

    # plt.rcParams["figure.figsize"] = [2.00, 3.50]
    x_f = ["survived"] * len(survived)
    x_l = ["dead"] * len(dead)
    # x_f = [0] * NUMBER_OF_SIMULATIONS
    # x_l = [1] * NUMBER_OF_SIMULATIONS
    y_f = survived
    y_l = dead

    # for a, b in list(zip(y_f, y_l)):
    #     plt.plot(["Normal", "Infinite Only"], [a, b], 'k-', lw=.1)

    x = x_f + x_l
    y = y_f + y_l
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'x': x, 'y': y})

    sns.swarmplot('x', 'y', data=df)
    sns.despine()
        
    # plt.scatter(x_f, y_f)
    # plt.scatter(x_l, y_l)
    ax = plt.gca()
    ax.set_ylim([0, 52])
    plt.axhline(np.mean(y_f), color='blue', linewidth=2)
    plt.axhline(np.mean(y_l), color='orange', linewidth=2)
    plt.title("Taking Away Finite Reservoirs from Agents Causes a Decrease in Fitness")
    plt.ylabel("drop in aspect ratio fitness")
    plt.show()

def plot_2_columns_before_after_drop(xs, ys):
    # plt.rcParams["figure.figsize"] = [2.00, 3.50]
    x_f = ["Infinite Or Finite"] * NUMBER_OF_SIMULATIONS
    x_l = ["Infinite Only"] * NUMBER_OF_SIMULATIONS
    # x_f = [0] * NUMBER_OF_SIMULATIONS
    # x_l = [1] * NUMBER_OF_SIMULATIONS
    y_f = xs
    y_l = ys

    for a, b in list(zip(y_f, y_l)):
        plt.plot(["Normal", "Infinite Only"], [a, b], 'k-', lw=.1)

    x = x_f + x_l
    y = y_f + y_l
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'x': x, 'y': y})

    sns.swarmplot('x', 'y', data=df)
    sns.despine()
        
    # plt.scatter(x_f, y_f)
    # plt.scatter(x_l, y_l)
    ax = plt.gca()
    ax.set_ylim([0, 150])
    plt.axhline(np.mean(y_f), color='blue', linewidth=2)
    plt.axhline(np.mean(y_l), color='orange', linewidth=2)
    plt.title("Taking Away Finite Reservoirs from Agents Causes a Decrease in Fitness")
    plt.ylabel("Fitness")
    plt.show()

def compute_permutation_test(xs, ys):
    p_value = permutation_test(xs, ys,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)

    print("permutation test results: ", p_value)


def plot_relative_drop_1D(xs, ys):
    x = [""] * NUMBER_OF_SIMULATIONS
    y = [((x - y) / x) * 100 for x, y in list(zip(xs, ys)) if x > 0]
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'': x, 'Percent Drop In Fitness': y})

    sns.swarmplot('', 'Percent Drop In Fitness', data=df)
    sns.despine()

    ax = plt.gca()
    ax.set_ylim([0, 100])
    
    plt.axhline(np.mean(y), color='blue', linewidth=2)
    plt.title("Relative Percent Drop In Fitness After Switching To Only Infinite Reservoirs")
    # plt.ylabel("Percent Drop In Fitness")
    plt.show()

def plot_relative_drop_1D_first_and_last_generation(xs_f, ys_f, xs_l, ys_l):
    y1 = [((x - y) / x) * 100 for x, y in list(zip(xs_f, ys_f)) if x > 0]
    y2 = [((x - y) / x) * 100 for x, y in list(zip(xs_l, ys_l)) if x > 0]
    y = y1 + y2

    x = ["First"] * len(y1) + ["Last"] * len(y2)


    # x = ["First"] * len(xs_f) + ["Last"] * len(xs_l)
    # y = [((x - y) / x) * 100 for x, y in list(zip(xs_f, ys_f)) if x > 0] + [((x - y) / x) * 100 for x, y in list(zip(xs_l, ys_l)) if x > 0]
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'Generation': x, 'Percent Drop In Fitness': y})

    sns.swarmplot('Generation', 'Percent Drop In Fitness', data=df)
    sns.despine()


    ax = plt.gca()
    ax.set_ylim([0, 100])
    plt.axhline(np.mean(y[len(xs_f):]), color='orange', linewidth=2)
    plt.axhline(np.mean(y[:len(xs_l)]), color='blue', linewidth=2)
    plt.title("Relative Percent Drop In Fitness After Switching To Only Infinite Reservoirs")
    plt.ylabel("Percent Drop In Fitness")
    plt.show()

def plot_scatter_2D(xs, ys):
    plt.scatter(xs, ys)
    ax = plt.gca()
    ax.set_xlim([0, 150])
    ax.set_ylim([0, 150])
    plt.title("Fitness Drops When Agents Are Forced To Only Use Infinite Reservoirs")
    plt.xlabel("Fitness When Using Finite and Infinite Reservoirs")
    plt.ylabel("Fitness After Forcing Agent to Use Only Infinite Reservoirs")
    plt.grid(True)
    plt.show()

def pixel_similarity(b1, b2):
    diff_count = 0


    for row in range(len(b1.grid)):
        for col in range(len(b1.grid[0])):
            if b1.grid[row][col] != b2.grid[row][col]:
                diff_count += 1
    
    print(diff_count)

if __name__ == "__main__":
    main()