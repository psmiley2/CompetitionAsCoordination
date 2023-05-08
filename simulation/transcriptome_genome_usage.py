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

num_simulations = 50

hp.onlyInfinite = False
hp.targetAspectRatio = 5
hp.targetSize = 400
hp.useConsistency = False
hp.onlyFinite = False
hp.useReservoirsAsInputs = False
hp.onlyUseSizeAsFitness = False
hp.useMidpointsForAspectRatio = True

total_infs = []
total_fins = []

first_gen = []
last_gen = []
starting_finite_fuel_list = []
remaining_finite_fuel_list = []

def main():
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    genomesInfo = []

    ending_finite_fuel_conc = []
    for i in range(num_simulations):
        with open("data/start_5_mutate_5_5000_gens/r" + str(i) +".json", "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        def convertStringKeysToIntKeys(d):
            keys = list(d.keys()) 
            for k in keys:
                d[int(k)] = d[k]
                d.pop(k, None)
            return d

        # for i in range(len(genomesInfo)):
        g = genome.Genome(genomesInfo[0]["genome"], 
            convertStringKeysToIntKeys(genomesInfo[0]["metabolicReservoirValues"]), 
            genomesInfo[0]["fitness"],  
        )
        

        board.reset(g)
        infs_conc = []
        fins_conc = []
        starting_finite_fuel = g.initialMetabolicReservoirValues[0]
        rff = []
        for i in range(5):
            last_step_finite_fuel = g.initialMetabolicReservoirValues[0]
            previous_cell_count = 1
            infs = []
            fins = []
            while (len(board.dynamicCells)):
                board.step()
                
                f = fitness.Fitness(board=board)
                current_cell_count = f.skinCellCount + f.nerveCellCount + f.stemCellCount
                remaining_finite_fuel = ceil(g.currentMetabolicReservoirValues[0])

                change_cell_count = current_cell_count - previous_cell_count
                previous_cell_count = current_cell_count
                change_finite_fuel = last_step_finite_fuel - remaining_finite_fuel
                last_step_finite_fuel = remaining_finite_fuel
                change_infinite = change_cell_count - change_finite_fuel

                infs.append(change_infinite)
                fins.append(change_finite_fuel)

            rff.append(remaining_finite_fuel)


            # print(starting_finite_fuel, remaining_finite_fuel)
            
            infs_conc.append(sum(infs))
            fins_conc.append(sum(fins))
        
        starting_finite_fuel_list.append(starting_finite_fuel)
        
        ending_finite_fuel_conc.append(np.mean(rff) / starting_finite_fuel)
        remaining_finite_fuel_list.append(np.mean(rff))

        first_gen.append(100 * (np.mean(fins_conc) / (np.mean(infs_conc) + np.mean(fins_conc))))

    # print(ending_finite_fuel_conc)

    # print("mean percent finite fuel remaining: ", np.mean(ending_finite_fuel_conc))


    # x = ["Start Of Development"] * num_simulations + ["End Of Development"] * num_simulations
    # y = starting_finite_fuel_list + remaining_finite_fuel_list
    # print(x)
    # print(y)
    # sns.set(style='ticks', context='talk')
    # df= pd.DataFrame({'x': x, 'y': y})

    # sns.swarmplot('x', 'y', data=df)
    # sns.despine()
        
    # plt.axhline(np.mean(y[:num_simulations]), color='blue', linewidth=2)
    # plt.axhline(np.mean(y[num_simulations:]), color='orange', linewidth=2)
    # # plt.title("Simulations Using Only Infinite Reservoirs Take Longer to Reach " + str(threshold) + " Percent of Max Fitness")
    # plt.title("title")
    # plt.ylabel("Remaining Finite Fuel")
    # # plt.savefig(SAVE_TO + "/" + "threshold_" + str(threshold) + ".png")
    # plt.show()
    # plt.clf()

        g = genome.Genome(genomesInfo[-1]["genome"], 
            convertStringKeysToIntKeys(genomesInfo[-1]["metabolicReservoirValues"]), 
            genomesInfo[-1]["fitness"],  
        )
        

        board.reset(g)
        infs_conc = []
        fins_conc = []
        for i in range(5):
            last_step_finite_fuel = g.initialMetabolicReservoirValues[0]
            previous_cell_count = 1
            infs = []
            fins = []
            while (len(board.dynamicCells)):
                board.step()
                
                f = fitness.Fitness(board=board)
                current_cell_count = f.skinCellCount + f.nerveCellCount + f.stemCellCount
                remaining_finite_fuel = ceil(g.currentMetabolicReservoirValues[0])

                change_cell_count = current_cell_count - previous_cell_count
                previous_cell_count = current_cell_count
                change_finite_fuel = last_step_finite_fuel - remaining_finite_fuel
                last_step_finite_fuel = remaining_finite_fuel
                change_infinite = change_cell_count - change_finite_fuel

                infs.append(change_infinite)
                fins.append(change_finite_fuel)
            
            infs_conc.append(sum(infs))
            fins_conc.append(sum(fins))

        last_gen.append(100 * (np.mean(fins_conc) / (np.mean(infs_conc) + np.mean(fins_conc))))


    p_value = permutation_test(first_gen, last_gen,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)
    print("p-value = ", p_value)

    # # print(sum(total_fins))
    # # print(sum(total_infs))


    # p_value = permutation_test(first_gen, last_gen,
    #                        method='approximate',
    #                        num_rounds=100000,
    #                        seed=0)
    # print("p_value: ", p_value)

    x = ["First Generation"] * num_simulations + ["Final Generation"] * num_simulations
    y = first_gen + last_gen
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'x': x, 'y': y})

    sns.swarmplot('x', 'y', data=df)
    sns.despine()
        
    plt.axhline(np.mean(y[:num_simulations]), color='blue', linewidth=2)
    plt.axhline(np.mean(y[num_simulations:]), color='orange', linewidth=2)
    # plt.title("Simulations Using Only Infinite Reservoirs Take Longer to Reach " + str(threshold) + " Percent of Max Fitness")
    plt.title("title")
    plt.ylabel("Percent Finite Fuel Used")
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