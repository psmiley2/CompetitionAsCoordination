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

# construct cmap
flatui = [ "#FFFFFF", "#FFFFFF", "#000000", "#FF0000", "#00FF00"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

def main():
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    genomesInfo = []

    with open("data/infinite_or_finite/r0.json", "r") as json_file:
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


    board.reset(g)
    while (len(board.dynamicCells)):
        board.step()

    data = np.array(board.grid)

    rows,cols = data.shape

    plt.imshow(data, interpolation='none', 
                    extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                    aspect="equal",
                    cmap=my_cmap)
    plt.show()

    if hp.deleteCells:
        for i in range(board.height):
            for j in range(board.width):
                if j < hp.deleteCellSpecification["E"] and j > hp.deleteCellSpecification["W"] and i > hp.deleteCellSpecification["N"] and i < hp.deleteCellSpecification["S"]:
                    board.grid[i][j] = 0
    data = np.array(board.grid)

    rows,cols = data.shape

    plt.imshow(data, interpolation='none', 
                    extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                    aspect="equal",
                    cmap=my_cmap)
    plt.show()

    for c in board.staticCells:
        board.dynamicCells.add(c)
    
    while (len(board.dynamicCells)):
        board.step()
   
    data = np.array(board.grid)

    rows,cols = data.shape

    plt.imshow(data, interpolation='none', 
                    extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                    aspect="equal",
                    cmap=my_cmap)
    plt.show()



if __name__ == "__main__":
    main()