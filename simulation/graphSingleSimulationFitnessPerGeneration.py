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
import fitness

def main():
    genomesInfo = []
    totalFitness = 150

    fitnesses = []

    max_fitness = 150 



    FPs = [
        # "data/fin_only/r54.json",
        "data/test_noise/.json",
    ]

    # for i in range(30): 
    #     FPs.append("data/fin_only/r" + str(i) + ".json")

    for FP in FPs:
        with open(FP, "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]
        

        xs = list(range(len(genomesInfo)))
        ys = [genomesInfo[i]["fitness"] for i in range(len(genomesInfo))]

        fig = plt.figure()
        ax = plt.axes()
        ax.set_ylim([80, 150])
        ax.plot(xs, ys)
        ax.set(xlabel='Generation Number', ylabel='Fitness', title="Infinite And Finite Throughout The Course Of Evolution")
        ax.grid(True)
        plt.show()
        # print(FP)
        # print(str(FP[-7: -5]))
        # plt.savefig("fin_only_single_graphs/" + str(FP[-7: -5]) + ".png")
        plt.clf()

if __name__ == "__main__":
    main()