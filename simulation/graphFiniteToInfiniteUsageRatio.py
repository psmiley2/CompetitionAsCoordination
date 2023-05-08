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

NUMBER_OF_SIMULATIONS = 20 

def main():
    def convertStringKeysToIntKeys(d):
        keys = list(d.keys()) 
        for k in keys:
            d[int(k)] = d[k]
            d.pop(k, None)
        return d

    # first_board = brd.Board(hp.boardWidth, hp.boardHeight)
    # last_board = brd.Board(hp.boardWidth, hp.boardHeight)
    genomesInfo = []

    first_gen_ys = []
    last_gen_ys = []

    for s_idx in range(NUMBER_OF_SIMULATIONS):
        FP = "data/inf_or_fin/r" + str(s_idx) + ".json"
        with open(FP, "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]
            json_file.close() 

        # with open("lookup_table_outputs_temp.json", 'w') as outfile:
        #     print("[", file=outfile)
        #     outfile.close()  

        first_g = genome.Genome(genomesInfo[0]["genome"], 
            convertStringKeysToIntKeys(genomesInfo[0]["metabolicReservoirValues"]), 
            genomesInfo[0]["fitness"],  
        )

        last_g = genome.Genome(genomesInfo[-1]["genome"], 
            convertStringKeysToIntKeys(genomesInfo[-1]["metabolicReservoirValues"]), 
            genomesInfo[-1]["fitness"],  
        )

        rh_fg = genomesInfo[0]["resHits"]
        rh_lg = genomesInfo[-1]["resHits"]

        sFinF = sum([rh_fg[0]["metFin"], rh_fg[1]["metFin"], rh_fg[2]["metFin"], rh_fg[3]["metFin"]])
        sInfF = sum([rh_fg[0]["metInf"], rh_fg[1]["metInf"], rh_fg[2]["metInf"], rh_fg[3]["metInf"]])

        sFinL = sum([rh_lg[0]["metFin"], rh_lg[1]["metFin"], rh_lg[2]["metFin"], rh_lg[3]["metFin"]])
        sInfL = sum([rh_lg[0]["metInf"], rh_lg[1]["metInf"], rh_lg[2]["metInf"], rh_lg[3]["metInf"]])

        first_gen_ys.append(sFinF / sInfF)
        if sInfL == 0:
            sInfL = 1
        last_gen_ys.append(sFinL / sInfL)
        

    plt.rcParams["figure.figsize"] = [2.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    x_f = ["first"] * NUMBER_OF_SIMULATIONS
    x_l = ["last"] * NUMBER_OF_SIMULATIONS
    y_f = first_gen_ys
    y_l = last_gen_ys
    plt.scatter(x_f, y_f)
    plt.scatter(x_l, y_l)
    plt.axhline(np.mean(y_f), color='blue', linewidth=2)
    plt.axhline(np.mean(y_l), color='orange', linewidth=2)
    plt.title("How Finite vs. Infinite Reservoirs Usages Evolves")
    plt.xlabel("Generation")
    plt.ylabel("Ratio of Finite to Infnite Reservoir Usage")
    plt.show()
        

if __name__ == "__main__":
    main()