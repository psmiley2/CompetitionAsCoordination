from ast import Num
from cProfile import run
from numbers import Rational
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
import os 
from mlxtend.evaluate import permutation_test

flatui = [ "#FFFFFF", "#FFFFFF", "#000000", "#FF0000", "#00FF00"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())


NUMBER_OF_SIMULATIONS = 1
unique_outputs_simulations = []
genome_lengths = []
total_sizes = []

repeat_5s = []

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


    updateHyperParametersF1()
    hp.printLookupTableOutputs = True

    def convertStringKeysToIntKeys(d):
        keys = list(d.keys()) 
        for k in keys:
            d[int(k)] = d[k]
            d.pop(k, None)
        return d

    first_board = brd.Board(hp.boardWidth, hp.boardHeight)
    last_board = brd.Board(hp.boardWidth, hp.boardHeight)
    genomesInfo = []

    for s_idx in range(NUMBER_OF_SIMULATIONS):
        FP = "data/test_noise/r" + str(s_idx) + ".json"
        with open(FP, "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]
            json_file.close() 

        with open("lookup_table_outputs_temp.json", 'w') as outfile:
            print("[", file=outfile)
            outfile.close()  

        # first_g = genome.Genome(genomesInfo[0]["genome"], 
        #     convertStringKeysToIntKeys(genomesInfo[0]["metabolicReservoirValues"]), 
        #     genomesInfo[0]["fitness"],  
        # )

        last_g = genome.Genome(genomesInfo[-1]["genome"], 
            convertStringKeysToIntKeys(genomesInfo[-1]["metabolicReservoirValues"]), 
            genomesInfo[-1]["fitness"],  
        )
        # first_g.fillReservoirs()
        last_g.fillReservoirs()
        
        # first_board.reset(first_g)
        # while (len(first_board.dynamicCells)):
        #     first_board.step()
        
        last_board.reset(last_g)
        while (len(last_board.dynamicCells)):
            last_board.step()

        f = fit.Fitness(last_board)
        f.calculate()
        total_sizes.append(f.stemCellCount + f.skinCellCount + f.nerveCellCount)
            # data = np.array(last_board.grid)

            # rows,cols = data.shape

            # plt.imshow(data, interpolation='none', 
            #                 extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
            #                 aspect="equal",
            #                 cmap=my_cmap)
            
            # plt.title("test")

            # plt.show()
            # plt.clf()


        
        pie_chart(last_g, s_idx)
        genome_lengths.append(len(genomesInfo[-1]["genome"]))
    
        
    # print(unique_outputs_simulations)
    # print(total_sizes)

    # print(np.mean(total_sizes))
    # print(np.mean(unique_outputs_simulations))
    # print(genome_lengths)
    # print(np.mean(total_sizes) / np.mean(unique_outputs_simulations))



    # ratios = []
    # for i in range(NUMBER_OF_SIMULATIONS):
    #     ratios.append(total_sizes[i] / unique_outputs_simulations[i])
    
    # print(ratios)
    # print(np.mean(ratios))

    print("avg: ", np.mean(repeat_5s))
    repeat_5s.sort()
    print(repeat_5s)

    hp.printLookupTableOutputs = False


def pie_chart(g, s_idx):
    with open("lookup_table_outputs_temp.json", 'a+') as outfile:
        print("]", file=outfile)
        outfile.close()  
    
    file=open('lookup_table_outputs_temp.json','r')
    states = [line for line in file.readlines()]
    file.close()

    target=open('lookup_table_outputs_temp.json','w')
    for i in range(len(states)):
        if i < len(states) - 2 and i != 0:
            target.write(states[i] + ",")
        else:
            target.write(states[i])
    target.close()
    

    run_data = []
    with open("lookup_table_outputs_temp.json", "r") as json_file:
        run_data = json.load(json_file)
        json_file.close() 

    d = {}



    # print(len(run_data))
    # print(run_data)

    for i in range(len(run_data) - 5):
        nc = []
        nc.append(str(run_data[i]))
        nc.append(str(run_data[i + 1]))
        nc.append(str(run_data[i + 2]))
        nc.append(str(run_data[i + 3]))
        nc.append(str(run_data[i + 4]))
        nc.sort()
        nc = str(nc)
        try:
            d[nc] = d[nc] + 1
        except:
            d[nc] = 1

    l = sorted(d, key=d.get, reverse=True)
    # for el in l:
    #     print(el)


    vals = []
    for v in d.values():
        vals.append(v)

    vals.sort()
    print(sum(vals) - len(vals))
    repeat_5s.append(sum(vals) - len(vals))
    

        

    # for el in run_data:
    #     try:
    #         d[str(el)] += 1
    #     except:
    #         d[str(el)] = 1
    
    # # FP2 = "data/r" + str(s_idx) + ".png" 
    # run_data = sorted(d, key=d.get, reverse=True)
    # # outputs = []
    # number_of_times_used = []

    # for el in run_data:
    #     number_of_times_used.append(d[el])
    # #     outputs.append(el)

    # print(number_of_times_used)

    
    
    # s = sum(number_of_times_used)
    # threshold = .75
    # unique_outputs = 0
    # counter = 0
    # limit = s * threshold
    # # print("limit", limit)
    # for hits in number_of_times_used:
    #     # print(counter)
    #     if int(hits) + counter >= limit:
    #         unique_outputs += 1
    #         break
    #     else:
    #         counter += int(hits)
    #         unique_outputs += 1
    
    # unique_outputs_simulations.append(unique_outputs)
    
    





    # fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    # data = number_of_times_used

    # def func(pct, allvals):
    #     absolute = int(np.round(pct/100.*np.sum(allvals)))
    #     return "{:.1f}%\n({:d})".format(pct, absolute)

    
    # wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
    #                                 textprops=dict(color="w"))

    # print(wedges)
    # print(texts)
    # print(autotexts)

    # ax.legend(wedges, outputs,
    #         title="Outputs",
    #         loc="lower center",
    #         bbox_to_anchor=(0.5, -0.1))

    # plt.setp(autotexts, size=8, weight="bold")

    # ax.set_title("Top Lookup Table Outputs/Inputs for r" + str(s_idx))

    # plt.show()


    # print("lookup_table_length:", len(genomesInfo[-1]["genome"].keys()), ",", file=outfile)
    # print("number_of_outputs_used:", len(run_data), ",",file=outfile)
    # print("unique_outputs_used:", len(d), ",", file=outfile)

if __name__ == "__main__":
    main()