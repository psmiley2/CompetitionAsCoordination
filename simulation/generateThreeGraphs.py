import os
from types import TracebackType
import numpy as np
from PIL.Image import SAVE
from numpy.core.fromnumeric import std
from numpy.lib.index_tricks import s_
from scipy.stats.stats import trim_mean
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
import os
from scipy.stats import ttest_ind
import pandas as pd 
from mlxtend.evaluate import permutation_test



flatui = [ "#FFFFFF", "#FFFFFF", "#000000", "#FF0000", "#00FF00"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())


totalFitness = 150
numberOfSimulations = 100
scaleDownFitnessScore = 80
graph_y_min = 0
graph_y_max = 70 + 2
numGenerations = 7500
minThresholdForImages = 0

top_fitness_scores_1_first = []
top_fitness_scores_1_final = []
top_fitness_scores_2_first = []
top_fitness_scores_2_final = []
top_fitness_scores_3_first = []
top_fitness_scores_3_final = []

FP1 = "inf_or_fin"
FP2 = "fin_only"
FP3 = "inf_only"
SAVE_TO = "eh"

std_devs_f1 = []
std_devs_f2 = []
std_devs_f3 = []


def updateHyperParametersF1():
    hp.onlyInfinite = False
    hp.targetAspectRatio = 5
    hp.targetSize = 400
    hp.useConsistency = False
    hp.onlyFinite = False
    hp.useReservoirsAsInputs = False
    hp.onlyUseSizeAsFitness = False
    hp.useMidpointsForAspectRatio = True

def updateHyperParametersF2():
    hp.onlyInfinite = True
    hp.targetAspectRatio = 5
    hp.targetSize = 400
    hp.useConsistency = False
    hp.onlyFinite = False
    hp.useReservoirsAsInputs = False
    hp.onlyUseSizeAsFitness = False
    hp.useMidpointsForAspectRatio = True

def updateHyperParametersF3():
    hp.onlyInfinite = False
    hp.targetAspectRatio = 5
    hp.targetSize = 400
    hp.useConsistency = False
    hp.onlyFinite = False
    hp.useReservoirsAsInputs = False
    hp.onlyUseSizeAsFitness = False
    hp.useMidpointsForAspectRatio = True

def convertStringKeysToIntKeys(d):
    keys = list(d.keys()) 
    for k in keys:
        d[int(k)] = d[k]
        d.pop(k, None)
    return d


def main():
    
    genomesInfo1 = []
    genomesInfo2 = []
    genomesInfo3 = []

    first_gen_1 = []
    final_gen_1 = []
    first_gen_2 = []
    final_gen_2 = []
    first_gen_3 = []
    final_gen_3 = []

    # os.mkdir(SAVE_TO)
    # os.mkdir(SAVE_TO + "/first_gen_pics_" + FP1)
    # os.mkdir(SAVE_TO + "/first_gen_pics_" + FP2)
    # os.mkdir(SAVE_TO + "/final_gen_pics_" + FP1)
    # os.mkdir(SAVE_TO + "/final_gen_pics_" + FP2)
    # os.mkdir(SAVE_TO + "/evolution_pics_" + FP1)
    # os.mkdir(SAVE_TO + "/evolution_pics_" + FP2)

    

    for s_idx in range(numberOfSimulations):
        print(s_idx)
    # for s_idx in [20, 70, 85, 24]:
    # for s_idx in [0, 70, 6, 63]:
    # for s_idx in [95, 11, 29, 65]:
    # for s_idx in [63]:
        
        FP = "data/" + FP1 + "/r" + str(s_idx) + ".json"
        print(FP)
        with open(FP, "r") as json_file:
            data = json.load(json_file)
            genomesInfo1 = data["genomes"]

        updateHyperParametersF1()

        first_gen_1.append(genomesInfo1[0]["fitness"])
        final_gen_1.append(genomesInfo1[-1]["fitness"])

        g_final_1 = genome.Genome(genomesInfo1[-1]["genome"], 
            convertStringKeysToIntKeys(genomesInfo1[-1]["metabolicReservoirValues"]), 
            genomesInfo1[-1]["fitness"],  
        )

        g_first_1 = genome.Genome(genomesInfo1[0]["genome"], 
            convertStringKeysToIntKeys(genomesInfo1[0]["metabolicReservoirValues"]), 
            genomesInfo1[0]["fitness"],  
            )
    
        updateHyperParametersF2()
        FP = "data/" + FP2 + "/r" + str(s_idx) + ".json"
        # print(FP)
        with open(FP, "r") as json_file:
            data = json.load(json_file)
            genomesInfo2 = data["genomes"]

        first_gen_2.append(genomesInfo2[0]["fitness"])
        final_gen_2.append(genomesInfo2[-1]["fitness"])

        g_final_2 = genome.Genome(genomesInfo2[-1]["genome"], 
            convertStringKeysToIntKeys(genomesInfo2[-1]["metabolicReservoirValues"]), 
            genomesInfo2[-1]["fitness"],  
        )

        g_first_2 = genome.Genome(genomesInfo2[0]["genome"], 
            convertStringKeysToIntKeys(genomesInfo2[0]["metabolicReservoirValues"]), 
            genomesInfo2[0]["fitness"],  
        )

        updateHyperParametersF3()
        FP = "data/" + FP3 + "/r" + str(s_idx) + ".json"
        # print(FP)
        with open(FP, "r") as json_file:
            data = json.load(json_file)
            genomesInfo3 = data["genomes"]

        first_gen_3.append(genomesInfo3[0]["fitness"])
        final_gen_3.append(genomesInfo3[-1]["fitness"])

        g_final_3 = genome.Genome(genomesInfo3[-1]["genome"], 
            convertStringKeysToIntKeys(genomesInfo3[-1]["metabolicReservoirValues"]), 
            genomesInfo3[-1]["fitness"],  
        )

        g_first_3 = genome.Genome(genomesInfo3[0]["genome"], 
            convertStringKeysToIntKeys(genomesInfo3[0]["metabolicReservoirValues"]), 
            genomesInfo3[0]["fitness"],  
        )
        

        # ------------------
        # GRAPHS TO GENERATE 
        # ------------------

        # Keep tabbed 
        # generate_first_generation_pictures(g_first_1, FP1, s_idx, updateHyperParametersF1)
        # generate_first_generation_pictures(g_first_2, FP2, s_idx, updateHyperParametersF2)
        # generate_final_generation_pictures(g_final_1, FP1, s_idx, updateHyperParametersF1)
        # generate_final_generation_pictures(g_final_2, FP2, s_idx, updateHyperParametersF2)
        
        consistency_difference(g_final_1, g_final_2, g_final_3, FP1, FP2, FP3, updateHyperParametersF1, updateHyperParametersF2, updateHyperParametersF3, 30, s_idx)
      
        # if s_idx in [2, 4]:
        #     generate_pic_of_each_generation_in_evolution(genomesInfo1, FP1, s_idx, updateHyperParametersF1)
        #     generate_pic_of_each_generation_in_evolution(genomesInfo2, FP2, s_idx, updateHyperParametersF2)

        # top_fitness_scores_1_final.append([g_final_1.totalFitness, s_idx])
    #     top_fitness_scores_2_final.append([g_final_2.totalFitness, s_idx])
    #     top_fitness_scores_1_first.append([g_first_1.totalFitness, s_idx])
    #     top_fitness_scores_2_first.append([g_first_2.totalFitness, s_idx])


    # print_top_agent_numbers(top_fitness_scores_1_first, top_fitness_scores_2_first, "first")
    # print_top_agent_numbers(top_fitness_scores_1_final, top_fitness_scores_2_final, "final")

    # smooshed_avg_fitness_per_generation(FP1, FP2, FP3)
    # avg_fitness_per_generation(genomesInfo1, FP1, "Selecting For Consistency Infinite And Finite")
    # avg_fitness_per_generation(genomesInfo2, FP2, "Selecting For Consistency Only Infinite")
    # avg_fitness_per_generation(genomesInfo3, FP3, "Selecting For Consistency Only Finite")
    # gen_csv_of_first_and_final_gen_scores(first_gen_1, final_gen_1, first_gen_2, final_gen_2)
    # threshold_passing([.8], FP1, FP2, FP3)
    # t_test(FP1, FP2, final_gen_1, final_gen_2)
    consistency_difference_summary()
    # top_fitness_dot_plot(final_gen_1, final_gen_2, final_gen_3)


def top_fitness_dot_plot(final_gen_1, final_gen_2, final_gen_3):
    ys_a = final_gen_1
    ys_b = final_gen_2
    ys_c = final_gen_3

    x = ["2_p_start"] * numberOfSimulations + ["10_p_start"] * numberOfSimulations + ["50_p_start"] * numberOfSimulations 
    y = ys_a + ys_b + ys_c
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'x': x, 'y': y})

    sns.swarmplot('x', 'y', data=df)
    sns.despine()
        
    plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
    plt.axhline(np.mean(y[numberOfSimulations:numberOfSimulations * 2]), color='orange', linewidth=2)
    plt.axhline(np.mean(y[numberOfSimulations * 2:]), color='green', linewidth=2)
    ax = plt.gca()
    ax.set_ylim([80, 150])
    plt.title("Summary Max Fitness Reached By The Simulations")
    plt.ylabel("Max Fitness Of A Single Simulation")
    # plt.savefig(SAVE_TO + "/max_fitness_dot_plot.png")
    # plt.clf()

    plt.show()


def print_top_agent_numbers(top_fitness_scores_1, top_fitness_scores_2, x):
    top_fitness_scores_1.sort()
    # top_fitness_scores_2.sort()
    print(x, top_fitness_scores_1[-4:])
    # print(x, top_fitness_scores_2[-4:])

def smooshed_avg_fitness_per_generation(FP1, FP2, FP3):
    fitnesses1 = []
    fitnesses2 = []
    fitnesses3 = []

    for s_idx in range(numberOfSimulations):
        print(s_idx)
        file = "data/" + FP1 + "/r" + str(s_idx) + ".json"
        with open(file, "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        for i in range(len(genomesInfo)):
            score = genomesInfo[i]["fitness"]
            
            try:
                fitnesses1[i].append(score)
            except:
                fitnesses1.append([score])

    for s_idx in range(numberOfSimulations):
        print(s_idx)
        file = "data/" + FP2 + "/r" + str(s_idx) + ".json"
        with open(file, "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        for i in range(len(genomesInfo)):
            score = genomesInfo[i]["fitness"]
            
            try:
                fitnesses2[i].append(score)
            except:
                fitnesses2.append([score])

    for s_idx in range(numberOfSimulations):
        print(s_idx)
        file = "data/" + FP3 + "/r" + str(s_idx) + ".json"
        with open(file, "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

        for i in range(len(genomesInfo)):
            score = genomesInfo[i]["fitness"]
            
            try:
                fitnesses3[i].append(score)
            except:
                fitnesses3.append([score])
        
    xs = list(range(numGenerations + 1))
    ys1 = []
    ys2 = []
    ys3 = []
    for k in range(numGenerations + 1):
        ys1.append(np.mean(fitnesses1[k]))
        ys2.append(np.mean(fitnesses2[k]))
        ys3.append(np.mean(fitnesses3[k]))

    ax = plt.gca()
    ax.plot(xs, ys1[:numGenerations + 1])
    ax.plot(xs, ys2[:numGenerations + 1])
    ax.plot(xs, ys3[:numGenerations + 1])
    ax.set(xlabel='Generation Number', ylabel='Average of Top Fitnesses of 100 Simulations', title="Average Fitness Progression During Evolution")
    ax.grid(True)
    ax.set_ylim([0, 150])
    ax.legend(["Infinite Only", "Finite Locked At 5%", "Infinite And Finite"])
    # ax.legend(["Finite Fixed 5%", "Infinite Only"])
    # plt.savefig(SAVE_TO  + "/Avg_Fitness_smooshed.png")
    plt.show()
    # plt.clf()

def avg_fitness_per_generation(genomesInfo, FP, graph_title):
    fitnesses = []

    for i in range(len(genomesInfo)):
        score = genomesInfo[i]["fitness"]
        
        try:
            fitnesses[i].append(score)
        except:
            fitnesses.append([score])
        
    xs = list(range(numGenerations + 1))
    ys = []
    for k in range(len(fitnesses)):
        ys.append(np.mean(fitnesses[k]))
        # ys.append(max(fitnesses[k]))

    fig = plt.figure()
    ax = plt.axes()
    ax = plt.gca()
    ax.set_ylim([0, 150])
    ax.plot(xs, ys)
    ax.set(xlabel='Generation Number', ylabel='Average of Top Fitnesses of 100 Simulations', title=graph_title)
    ax.grid(True)
    ax.set_ylim([graph_y_min, graph_y_max])
    # plt.savefig(SAVE_TO  + "/Avg_Fitness__" + FP + ".png")
    plt.show()
    plt.clf()

def gen_csv_of_first_and_final_gen_scores(first_gen_1, final_gen_1, first_gen_2, final_gen_2):
    df = pd.DataFrame({
        FP1 + "_first_generation": first_gen_1,
        FP1 + "final_generation": final_gen_1,
        FP2 + "first_generation": first_gen_2,
        FP2 + "final_generation": final_gen_2,
    })

    df.to_csv(SAVE_TO + "/csv_of_first_and_final_gen_scores.csv", index=False)

def generate_first_generation_pictures(g, FP, s_idx, update_params):
    update_params()
    
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    score = 0

    # while score < minThresholdForImages:
    board.reset(g)
    g.fillReservoirs()
    while (len(board.dynamicCells)):
        board.step()

    f = fitness.Fitness(board=board)
    f.calculate()
    score = f.totalScore

    data = np.array(board.grid)

    rows,cols = data.shape

    plt.imshow(data, interpolation='none', 
                    extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                    aspect="equal",
                    cmap=my_cmap)
    
    plt.title("r" + str(s_idx))
    fitness_breakdown_string = "total fitness: " + str(round(score)) + "/ 150\nhead vs. tail fitness: " + str(round(f.cellRatioScore)) + " / 50 \naspect ratio fitness: " + str(round(f.aspectRatioScore)) + " / 50 \nsize fitness: " + str(round(f.sizeScore)) + " / 50"   
    plt.annotate(fitness_breakdown_string, 
                (20, 8), # these are the coordinates to position the label
                color='blue')
    # plt.savefig(SAVE_TO + "/first_gen_pics_" + FP + "/r" + str(s_idx) + ".png")
    plt.show()
    plt.clf()

def generate_final_generation_pictures(g, FP, s_idx, update_params):
    update_params()
    
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    score = 0

    # while score < minThresholdForImages:
    board.reset(g)
    g.fillReservoirs()
    while (len(board.dynamicCells)):
        board.step()

    f = fitness.Fitness(board=board)
    f.calculate()
    score = f.totalScore

    data = np.array(board.grid)

    rows,cols = data.shape

    plt.imshow(data, interpolation='none', 
                    extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                    aspect="equal",
                    cmap=my_cmap)
    
    plt.title("r" + str(s_idx))
    fitness_breakdown_string = "total fitness: " + str(round(score)) + "/ 150\nhead vs. tail fitness: " + str(round(f.cellRatioScore)) + " / 50 \naspect ratio fitness: " + str(round(f.aspectRatioScore)) + " / 50 \nsize fitness: " + str(round(f.sizeScore)) + " / 50"
    plt.annotate(fitness_breakdown_string, 
                (20, 8), # these are the coordinates to position the label
                color='blue')
    # plt.savefig(SAVE_TO + "/final_gen_pics_" + FP + "/r" + str(s_idx) + ".png")
    plt.show()
    plt.clf()

def generate_pic_of_each_generation_in_evolution(genomesInfo, FP, s_idx, update_params):
    update_params()

    os.mkdir(SAVE_TO + "/evolution_pics_" + FP +"/r" + str(s_idx))

    board = brd.Board(hp.boardWidth, hp.boardHeight)
    
    genomes = []
    for gi in genomesInfo:
        genomes.append( 
            genome.Genome(gi["genome"], 
                convertStringKeysToIntKeys(gi["metabolicReservoirValues"]), 
                gi["fitness"],  
            )
        )

    # for i in range(0, len(genomes)):
    for i in [0, 100, 500, 3500, 5000, 7499]:
        genomeToLookAt = genomes[i]
        genomeToLookAt.fillReservoirs()
        board.reset(genomeToLookAt)
        while (len(board.dynamicCells)):
            board.step()

        data = np.array(board.grid)

        rows,cols = data.shape

        f = fitness.Fitness(board)
        score = f.totalScore

        plt.imshow(data, interpolation='none', 
                        extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                        aspect="equal",
                        cmap=my_cmap)


        fitness_breakdown_string = "total fitness: " + str(score) + "/ 150\nhead vs. tail fitness: " + str(f.cellRatioScore) + " / 50 \naspect ratio fitness: " + str(f.aspectRatioScore) + " / 50 \nsize fitness: " + str(f.sizeScore) + " / 50"
        plt.annotate(fitness_breakdown_string, 
                    (40, 8), # these are the coordinates to position the label
                    color='blue')

        plt.title("Generation " + str(i))

        # plt.savefig(SAVE_TO + "/evolution_pics_" + FP +"/r" + str(s_idx) + "/" + str(i) + '.png')
        plt.show()
        plt.clf()

def threshold_passing(thresholds, FP1, FP2, FP3):
    for threshold in thresholds:
        ys_a = []
        ys_b = []
        ys_c = []

        for s_idx in range(numberOfSimulations):
            FP = "data/" + FP1 + "/r" + str(s_idx) + ".json"
            with open(FP, "r") as json_file:
                data = json.load(json_file)
                genomesInfo = data["genomes"]
            
            for i, gi in enumerate(genomesInfo):
                if len(ys_a) == s_idx:
                    if gi["fitness"] > totalFitness * threshold:
                        ys_a.append(i)
                
                
            if len(ys_a) == s_idx:
                ys_a.append(numGenerations)
        
        for s_idx in range(numberOfSimulations):
            FP = "data/" + FP2 + "/r" + str(s_idx) + ".json"
            with open(FP, "r") as json_file:
                data = json.load(json_file)
                genomesInfo = data["genomes"]
            
            for i, gi in enumerate(genomesInfo):
                if len(ys_b) == s_idx:
                    if gi["fitness"] > totalFitness * threshold:
                        ys_b.append(i)
                
                
            if len(ys_b) == s_idx:
                ys_b.append(numGenerations)
        

        for s_idx in range(numberOfSimulations):
            FP = "data/" + FP3 + "/r" + str(s_idx) + ".json"
            with open(FP, "r") as json_file:
                data = json.load(json_file)
                genomesInfo = data["genomes"]
            
            for i, gi in enumerate(genomesInfo):
                if len(ys_c) == s_idx:
                    if gi["fitness"] > totalFitness * threshold:
                        ys_c.append(i)
                
                
            if len(ys_c) == s_idx:
                ys_c.append(numGenerations)

        # x = [FP1] * numberOfSimulations + [FP2] * numberOfSimulations
        x = ["2_p"] * numberOfSimulations + ["10_p"] * numberOfSimulations + ["50_p"] * numberOfSimulations 
        y = ys_a + ys_b + ys_c
        sns.set(style='ticks', context='talk')
        df= pd.DataFrame({'x': x, 'y': y})

        sns.swarmplot('x', 'y', data=df)
        sns.despine()
            
        print(np.mean(y[:numberOfSimulations]))
        print(np.mean(y[numberOfSimulations:numberOfSimulations*2]))
        print(np.mean(y[numberOfSimulations*2:]))

        plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
        plt.axhline(np.mean(y[numberOfSimulations:numberOfSimulations*2]), color='orange', linewidth=2)
        plt.axhline(np.mean(y[numberOfSimulations*2:]), color='green', linewidth=2)
        # plt.title("Simulations Using Only Infinite Reservoirs Take Longer to Reach " + str(threshold) + " Percent of Max Fitness")
        plt.title("Generations When Simulations Passed " + str(threshold * 100) + "%" + " of Max Fitness")
        plt.ylabel("Generation")
        plt.show()
        # plt.savefig(SAVE_TO + "/" + "threshold_" + str(threshold) + ".png")
        plt.clf()

def consistency_difference(g1, g2, g3, FP1, FP2, FP3, update_params_1, update_params_2, update_params_3, repeats, s_idx):
    
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    score = 0
    
    ys_a = []
    ys_b = []
    ys_c = []

    update_params_1()

    for _ in range(repeats):
        board.reset(g1)
        g1.fillReservoirs()
        while (len(board.dynamicCells)):
            board.step()

        f = fitness.Fitness(board=board)
        f.calculate()
        score = f.totalScore
        ys_a.append(score)

    update_params_2()

    for _ in range(repeats):
        board.reset(g2)
        g2.fillReservoirs()
        while (len(board.dynamicCells)):
            board.step()

        f = fitness.Fitness(board=board)
        f.calculate()
        score = f.totalScore
        ys_b.append(score)

    update_params_3()

    for _ in range(repeats):
        board.reset(g3)
        g3.fillReservoirs()
        while (len(board.dynamicCells)):
            board.step()

        f = fitness.Fitness(board=board)
        f.calculate()
        score = f.totalScore
        ys_c.append(score)

    x = [FP1] * repeats + [FP2] * repeats + [FP3] * repeats
    y = ys_a + ys_b + ys_c
    # sns.set(style='ticks', context='talk')
    # df= pd.DataFrame({'x': x, 'y': y})

    # sns.swarmplot('x', 'y', data=df)
    # sns.despine()

    std_devs_f1.append(np.std(ys_a))
    std_devs_f2.append(np.std(ys_b))
    std_devs_f3.append(np.std(ys_c))

    std_dev_string = FP1 + " std dev: " + str(np.std(ys_a)) + "\n" +  FP2 + " std dev: " + str(np.std(ys_b))
    # plt.annotate(std_dev_string, 
    #             (50, 50), # these are the coordinates to position the label
    #             color='blue')
        
    # plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
    # plt.axhline(np.mean(y[numberOfSimulations:]), color='orange', linewidth=2)
    # plt.title("Repeat for r" + str(s_idx) + "\n" + std_dev_string)

    
    # plt.ylabel("Fitness Score")
    
    # plt.text(3+0.2, 4.5, "An annotation", horizontalalignment='left', size='medium', color='black', weight='semibold')
    # plt.savefig(SAVE_TO + "/" + "repeats_r" + str(s_idx) + ".png")
    # plt.show()
    # plt.clf()



def t_test(FP1, FP2, final_gen_1, final_gen_2):
    f = open(SAVE_TO + "/t_test___" + FP1 + "___" + FP2 + ".txt", "w")
    f.write(str(ttest_ind(final_gen_1, final_gen_2)))
    f.close()

def consistency_difference_summary():
    plt.rcParams["figure.figsize"] = [2.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    y_f = std_devs_f2
    y_l = std_devs_f3
    y_t = std_devs_f1

    x = ["Infinite Only"] * numberOfSimulations +  ["Finite Only"] * numberOfSimulations + ["Infinite And Finite"] * numberOfSimulations
    y = y_f + y_l + y_t
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'x': x, 'y': y})

    sns.swarmplot('x', 'y', data=df)
    sns.despine()
    
    p_value = permutation_test(std_devs_f2, std_devs_f3,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)
    print("1 p-value = ", p_value)

    p_value = permutation_test(std_devs_f1, std_devs_f2,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)
    print("2 p-value = ", p_value)

    p_value = permutation_test(std_devs_f1, std_devs_f3,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)
    print("3 p-value = ", p_value)

    plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
    plt.axhline(np.mean(y[numberOfSimulations:numberOfSimulations * 2]), color='orange', linewidth=2)
    plt.axhline(np.mean(y[numberOfSimulations * 2:]), color='green', linewidth=2)
    plt.title("Finite Reservoirs Enable Consistent Growth")
    plt.ylabel("Std Dev. of Fitness Over 30 Repeats")
    # plt.savefig(SAVE_TO + "/" + "consistency_summary.png")
    plt.show()
    

if __name__ == "__main__":
    main()