from itertools import count
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

num_simulations = 1

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

total_counts = {}

def main():
    genomesInfo = []

    for s_idx in range(num_simulations):
        with open("data/inf_or_fin/r" + str(s_idx) +".json", "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        def convertStringKeysToIntKeys(d):
            keys = list(d.keys()) 
            for k in keys:
                d[int(k)] = d[k]
                d.pop(k, None)
            return d

        
        g = genome.Genome(genomesInfo[-1]["genome"], 
            convertStringKeysToIntKeys(genomesInfo[-1]["metabolicReservoirValues"]), 
            genomesInfo[-1]["fitness"],  
        )

        unique_genes = list(g.genome.keys())
        # print(unique_genes)

        input_counts = {}

        for uq in unique_genes:
            d = {}
            for i in range(0, len(genomesInfo)):
            # for i in range(7000, len(genomesInfo)):
            # for i in range(0, 500):
                g = genome.Genome(genomesInfo[i]["genome"], 
                    convertStringKeysToIntKeys(genomesInfo[i]["metabolicReservoirValues"]), 
                    genomesInfo[i]["fitness"],  
                )

                try:
                    d[str(g.genome[uq])] += 1
                except:
                    try:
                        d[str(g.genome[uq])] = 1
                    except: pass

            input_counts[uq] = d


        for k, v in input_counts.items():
            try:
                total_counts[k].append(len(v))
            except:
                total_counts[k] = [len(v)]
            
    


    d = dict(sorted(total_counts.items(), key=lambda item: item[1]))
    keys_to_remove = []
    for k, v in d.items():
        print(len(v))
        if len(v) < 85:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        d.pop(k)


    output_dict = {}
    d = dict(sorted(d.items(), key=lambda item: item[1]))
    c = 0
    for k, v in d.items():
        c += 1
        output_dict[k] = v
        if c == 5:
            break
    
    c = 0
    for k, v in d.items():
        c += 1
        if c >= len(d) - 4:
            output_dict[k] = v

    print(output_dict)




    x = []
    y = []
    for k, v in output_dict.items():
        y += v
        x += [str(k)] * len(v)


    means = []
    for a in output_dict.values():
        means.append(np.mean(a))
    
    print(means)

    # print(x)
    # print(y)

    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'x': x, 'y': y})

    sns.swarmplot('x', 'y', data=df)
    sns.despine()
        
    # plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
    # plt.axhline(np.mean(y[numberOfSimulations:]), color='orange', linewidth=2)
    # plt.title("Simulations Using Only Infinite Reservoirs Take Longer to Reach " + str(threshold) + " Percent of Max Fitness")
    plt.title("title")
    plt.ylabel("Remaining Finite Fuel")
    # plt.savefig(SAVE_TO + "/" + "threshold_" + str(threshold) + ".png")
    plt.show()
    plt.clf()


    

if __name__ == "__main__":
    main()