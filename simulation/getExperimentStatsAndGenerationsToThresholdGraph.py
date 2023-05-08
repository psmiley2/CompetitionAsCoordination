import os
import numpy as np
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
import pandas as pd 

# construct cmap
flatui = [ "#FFFFFF", "#FFFFFF", "#000000", "#FF0000", "#00FF00"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

def main():
    genomesInfo = []
    totalFitness = 150

    gotToFiftyPercent = []
    gotToSixtyPercent = []
    gotToSeventyPercent = []
    gotToEightyPercent = []
    gotToNinetyPercent = []
    gotToNinetyFivePercent = []
    gotToNinetyEightPercent = []
    finiteConsumption = []
    infiniteConsumption = []

    for s_idx in range(20):
        FP = "data/base_res_as_input_consistency/r" + str(s_idx) + ".json"
        with open(FP, "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]
        
        rh = genomesInfo[s_idx]["resHits"]
        finiteConsumption.append(rh[0]["metFin"] + rh[1]["metFin"] + rh[2]["metFin"] + rh[3]["metFin"])
        infiniteConsumption.append(rh[0]["metInf"] + rh[1]["metInf"] + rh[2]["metInf"] + rh[3]["metInf"])

        for i, gi in enumerate(genomesInfo):
            if len(gotToFiftyPercent) == s_idx:
                if gi["fitness"] > totalFitness * .5:
                    gotToFiftyPercent.append(i)
            
            if len(gotToSixtyPercent) == s_idx:
                if gi["fitness"] > totalFitness * .6:
                    gotToSixtyPercent.append(i)

            if len(gotToSeventyPercent) == s_idx:
                if gi["fitness"] > totalFitness * .7:
                    gotToSeventyPercent.append(i)
            
            if len(gotToEightyPercent) == s_idx:
                if gi["fitness"] > totalFitness * .8:
                    gotToEightyPercent.append(i)
            
            if len(gotToNinetyPercent) == s_idx:
                if gi["fitness"] > totalFitness * .9:
                    gotToNinetyPercent.append(i)

            if len(gotToNinetyFivePercent) == s_idx:
                if gi["fitness"] > totalFitness * .95:
                    # print(s_idx)
                    gotToNinetyFivePercent.append(i)

            if len(gotToNinetyEightPercent) == s_idx:
                if gi["fitness"] > totalFitness * .98:
                    gotToNinetyEightPercent.append(i)
        
        if len(gotToFiftyPercent) < s_idx + 1:
            gotToFiftyPercent.append(" ")
        if len(gotToSixtyPercent) < s_idx + 1:
            gotToSixtyPercent.append(" ")
        if len(gotToSeventyPercent) < s_idx + 1:
            gotToSeventyPercent.append(" ")
        if len(gotToEightyPercent) < s_idx + 1:
            gotToEightyPercent.append(" ")
        if len(gotToNinetyPercent) < s_idx + 1:
            gotToNinetyPercent.append(" ")
        if len(gotToNinetyFivePercent) < s_idx + 1:
            gotToNinetyFivePercent.append(" ")
        if len(gotToNinetyEightPercent) < s_idx + 1:
            gotToNinetyEightPercent.append(" ")


    gotToFiftyPercent = list(filter(lambda i: i != " ", gotToFiftyPercent))
    gotToSixtyPercent = list(filter(lambda i: i != " ", gotToSixtyPercent))
    gotToSeventyPercent = list(filter(lambda i: i != " ", gotToSeventyPercent))
    gotToEightyPercent = list(filter(lambda i: i != " ", gotToEightyPercent))
    gotToNinetyPercent = list(filter(lambda i: i != " ", gotToNinetyPercent))
    gotToNinetyFivePercent = list(filter(lambda i: i != " ", gotToNinetyFivePercent))
    gotToNinetyEightPercent = list(filter(lambda i: i != " ", gotToNinetyEightPercent))

    print("50:")
    print(len(gotToFiftyPercent) , " / 20 reached this fitness")
    print("mean: ", np.mean(gotToFiftyPercent))
    print("median: ", np.median(gotToFiftyPercent))
    print(gotToFiftyPercent)
    print("-----")
    print("60:")
    print(len(gotToSixtyPercent) , " / 20 reached this fitness")
    print("mean: ", np.mean(gotToSixtyPercent))
    print("median: ", np.median(gotToSixtyPercent))
    print(gotToSixtyPercent)
    print("-----")
    print("70:")
    print(len(gotToSeventyPercent) , " / 20 reached this fitness")
    print("mean: ", np.mean(gotToSeventyPercent))
    print("median: ", np.median(gotToSeventyPercent))
    print(gotToSeventyPercent)
    print("-----")
    print("80:")
    print(len(gotToEightyPercent) , " / 20 reached this fitness")
    print("mean: ", np.mean(gotToEightyPercent))
    print("median: ", np.median(gotToEightyPercent))
    print(gotToEightyPercent)
    print("-----")
    print("90:")
    print(len(gotToNinetyPercent) , " / 20 reached this fitness")
    print("mean: ", np.mean(gotToNinetyPercent))
    print("median: ", np.median(gotToNinetyPercent))
    print(gotToNinetyPercent)
    print("-----")
    print("95:")
    print(len(gotToNinetyFivePercent) , " / 20 reached this fitness")
    print("mean: ", np.mean(gotToNinetyFivePercent))
    print("median: ", np.median(gotToNinetyFivePercent))
    print(gotToNinetyFivePercent)
    print("-----")
    print("98:")
    print(len(gotToNinetyEightPercent) , " / 20 reached this fitness")
    print("mean: ", np.mean(gotToNinetyEightPercent))
    print("median: ", np.median(gotToNinetyEightPercent))
    print(gotToNinetyEightPercent)
    print("-----")

    print("mean finite consumption: ", np.mean(finiteConsumption))
    print("median finite consumption: ", np.median(finiteConsumption))
    print("mean infinite consumption: ", np.mean(infiniteConsumption))
    print("median infinite consumption: ", np.median(infiniteConsumption))


if __name__ == "__main__":
    main()