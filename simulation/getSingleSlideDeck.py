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

# construct cmap
flatui = [ "#00FF00", "#FFFFFF", "#000000", "#FF0000", "#0000FF"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

def main():
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    genomesInfo = []

    with open("data/test_noise/r7.json", "r") as json_file:
        data = json.load(json_file)
        genomesInfo = data["genomes"]
    def convertStringKeysToIntKeys(d):
        keys = list(d.keys()) 
        for k in keys:
            d[int(k)] = d[k]
            d.pop(k, None)
        return d
    genomes = []
    for gi in genomesInfo:
        genomes.append( 
            genome.Genome(gi["genome"], 
                convertStringKeysToIntKeys(gi["metabolicReservoirValues"]), 
                gi["fitness"],  
            )
        )

    print(len(genomes))
    
    gens_to_run = list(range(0, len(genomes), 500))
    gens_to_run.append(len(genomes) - 1)
    print(gens_to_run)

    FP = "data/test_noise/"

    for k in gens_to_run:
        os.mkdir(FP + "/" + str(k))
        for i in range(5):
            for g in genomes:
                g.fillReservoirs()

            genomeToLookAt = genomes[k]
            genomeToLookAt.fillReservoirs()
            board.reset(genomeToLookAt)
            j = 0
            while (len(board.dynamicCells)):
                board.step()

            data = np.array(board.grid)

            rows,cols = data.shape

            plt.imshow(data, interpolation='none', 
                            extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                            aspect="equal", 
                            cmap=my_cmap)


            
            plt.savefig(FP + str(k) + "/" + str(i) + '.png')


if __name__ == "__main__":
    main()









