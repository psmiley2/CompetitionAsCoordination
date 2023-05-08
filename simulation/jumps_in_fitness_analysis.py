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

num_simulations = 100

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

start_but_not_end_counts = {}
end_but_not_start_counts = {}


# examples = [11, 23, 70, 71, 74, 78, 82, 88, 99]
# examples = [23,78]
examples = [23]
removed = []
changed_or_added = []

def main():
    genomesInfo = []

    # for ex in range(100):
    for ex in range(num_simulations):
        with open("data/inf_or_fin/r" + str(ex) +".json", "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        def convertStringKeysToIntKeys(d):
            keys = list(d.keys()) 
            for k in keys:
                d[int(k)] = d[k]
                d.pop(k, None)
            return d

        fitness_scores = []

        for i in range(len(genomesInfo)):
            g = genome.Genome(genomesInfo[i]["genome"], 
                convertStringKeysToIntKeys(genomesInfo[i]["metabolicReservoirValues"]), 
                genomesInfo[i]["fitness"],  
            )

            fitness_scores.append(genomesInfo[i]["fitness"])
        

        buffer = 200

        start = 0
        end = 0

        for i in range(250, len(fitness_scores) - 500):
            if np.mean(fitness_scores[i+490:i+500]) - np.mean(fitness_scores[i:i+10]) > 5:
                buffer -= 1
                if buffer == 0:
                    print(i, i+500)
                    start = i
                    end = i+500
        

        locked_in_genes_start = []
        locked_in_genes_end = []

        unique_genes = list(genomesInfo[end]["genome"].keys())
        input_counts = {}

        for uq in unique_genes:
            d = {}
            for i in range(start, start + 100):
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
            total_counts[k] = len(v)

        # print(total_counts)

        d = dict(sorted(total_counts.items(), key=lambda item: item[1]))
        # print("start")
        for k, v in d.items():
            # print(k,v)
            if v <= 3:
                locked_in_genes_start.append(k)

        
        for uq in unique_genes:
            d = {}
            for i in range(start+400, start + 500):
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
            total_counts[k] = len(v)

        # print(total_counts)

        d = dict(sorted(total_counts.items(), key=lambda item: item[1]))
        # print("End:")
        for k, v in d.items():
            # print(k,v)
            if v <= 3:
                locked_in_genes_end.append(k)



        print(locked_in_genes_start)
        print(locked_in_genes_end)
        print("in start, but not end: ", list(set(locked_in_genes_start) - set(locked_in_genes_end)))
        print("in end, but not start: ", list(set(locked_in_genes_end) - set(locked_in_genes_start)))

        removed.append(len(list(set(locked_in_genes_start) - set(locked_in_genes_end))))
        changed_or_added.append(len(list(set(locked_in_genes_end) - set(locked_in_genes_start))))

        for el in list(set(locked_in_genes_start) - set(locked_in_genes_end)):
            print(el, g.genome[el])
        

        for el in list(set(locked_in_genes_start) - set(locked_in_genes_end)):
            try:
                start_but_not_end_counts[el] += 1
            except:
                start_but_not_end_counts[el] = 1

        for el in list(set(locked_in_genes_end) - set(locked_in_genes_start)):
            try:
                end_but_not_start_counts[el] += 1
            except:
                end_but_not_start_counts[el] = 1

    print("removed: ", np.mean(removed))
    print("changed or added: ", np.mean(changed_or_added))
    # print("start")
    # d = dict(sorted(start_but_not_end_counts.items(), key=lambda item: item[1]))
    # for k, v in d.items():
    #     print(k,v)
        
    # print("end")
    # d = dict(sorted(end_but_not_start_counts.items(), key=lambda item: item[1]))
    # for k, v in d.items():
    #     print(k,v)

    # print("mean: ", np.mean(list(d.values())))
    # print("median: ", np.median(list(d.values())))

        
        # hp.onlyInfinite = False
        # hp.targetAspectRatio = 5
        # hp.targetSize = 400
        # hp.useConsistency = False
        # hp.onlyFinite = False
        # hp.useReservoirsAsInputs = False
        # hp.onlyUseSizeAsFitness = False
        # hp.useMidpointsForAspectRatio = True


        # board = brd.Board(hp.boardWidth, hp.boardHeight)
        
        # genomes = []
        # for gi in genomesInfo:
        #     genomes.append( 
        #         genome.Genome(gi["genome"], 
        #             convertStringKeysToIntKeys(gi["metabolicReservoirValues"]), 
        #             gi["fitness"],  
        #         )
        #     )

        # genomeToLookAt = genomes[start]
        # genomeToLookAt.fillReservoirs()
        # board.reset(genomeToLookAt)
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


        # fitness_breakdown_string = "total fitness: " + str(score) + "/ 150\nhead vs. tail fitness: " + str(f.cellRatioScore) + " / 50 \naspect ratio fitness: " + str(f.aspectRatioScore) + " / 50 \nsize fitness: " + str(f.sizeScore) + " / 50"

        # plt.title("Before r" + str(ex))

        # # plt.show()
        # plt.savefig("before_after_jump/r" + str(ex) + "_" + "start" + '.png')
        # plt.clf()






        # board = brd.Board(hp.boardWidth + 10, hp.boardHeight + 10)
        
        # genomes = []
        # for gi in genomesInfo:
        #     genomes.append( 
        #         genome.Genome(gi["genome"], 
        #             convertStringKeysToIntKeys(gi["metabolicReservoirValues"]), 
        #             gi["fitness"],  
        #         )
        #     )

        # genomeToLookAt = genomes[end]
        # genomeToLookAt.fillReservoirs()
        # board.reset(genomeToLookAt)
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


        # fitness_breakdown_string = "total fitness: " + str(score) + "/ 150\nhead vs. tail fitness: " + str(f.cellRatioScore) + " / 50 \naspect ratio fitness: " + str(f.aspectRatioScore) + " / 50 \nsize fitness: " + str(f.sizeScore) + " / 50"

        # plt.title("End r" + str(ex))

        # # plt.show()
        # plt.savefig("before_after_jump/r" + str(ex) + "_" + "end" + '.png')
        # plt.clf()



    # x = []
    # y = []
    # for k, v in output_dict.items():
    #     y += v
    #     x += [str(k)] * len(v)

    # print(x)
    # print(y)

    # sns.set(style='ticks', context='talk')
    # df= pd.DataFrame({'x': x, 'y': y})

    # sns.swarmplot('x', 'y', data=df)
    # sns.despine()
        
    # # plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
    # # plt.axhline(np.mean(y[numberOfSimulations:]), color='orange', linewidth=2)
    # # plt.title("Simulations Using Only Infinite Reservoirs Take Longer to Reach " + str(threshold) + " Percent of Max Fitness")
    # plt.title("title")
    # plt.ylabel("Remaining Finite Fuel")
    # # plt.savefig(SAVE_TO + "/" + "threshold_" + str(threshold) + ".png")
    # plt.show()
    # plt.clf()


    

if __name__ == "__main__":
    main()