from copy import deepcopy
import os
from types import TracebackType
import numpy as np
from PIL.Image import SAVE
from numpy.core.fromnumeric import std
from numpy.lib.index_tricks import s_
from scipy.fft import ihfft
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
from math import ceil
from mlxtend.evaluate import permutation_test



flatui = [ "#FFFFFF", "#FFFFFF", "#000000", "#FF0000", "#00FF00"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())


totalFitness = 150
numberOfSimulations = 100
graph_y_min = 0
graph_y_max = 70 + 2
numGenerations = 7500
minThresholdForImages = 0

top_fitness_scores_1_first = []
top_fitness_scores_1_final = []
top_fitness_scores_2_first = []
top_fitness_scores_2_final = []

FP1 = "oct_10_control"
FP2 = "oct_10_signals"
# SAVE_TO = "comparing_" + FP1 + "__and__" + FP2 + "forcing_infinite"
SAVE_TO = "comparing reservoir caps"
std_devs_f1 = []
std_devs_f2 = []

pixel_similarities_f1 = []
pixel_similarities_f2 = []

total_finite_hits = []
total_infinite_hits = []

target_size_scores = []
aspect_ratio_scores = []
red_green_scores = []

metabolic_reservoir_caps = []
metabolic_reservoir_caps.append([])



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

    first_gen_1 = []
    final_gen_1 = []
    first_gen_2 = []
    final_gen_2 = []


    # os.mkdir(SAVE_TO)
    # os.mkdir(SAVE_TO + "/first_gen_pics_" + FP1)
    # os.mkdir(SAVE_TO + "/first_gen_pics_" + FP2)
    # os.mkdir(SAVE_TO + "/final_gen_pics_" + FP1)
    # os.mkdir(SAVE_TO + "/final_gen_pics_" + FP2)
    # os.mkdir(SAVE_TO + "/evolution_pics_" + FP1)
    # os.mkdir(SAVE_TO + "/evolution_pics_" + FP2)

    updateHyperParametersF1()

    for s_idx in range(numberOfSimulations):
    # for s_idx in [59, 56, 96, 92]:
    # for s_idx in [20, 70, 85, 24]:
    # for s_idx in [0, 70, 6, 63]:
    # for s_idx in [95, 11, 29, 65]:
    # for s_idx in [1, 2, 3, 4, 5]:
    # for s_idx in [70]:
        FP = "data/" + FP1 + "/r" + str(s_idx) + ".json"
        with open(FP, "r") as json_file:
            data = json.load(json_file)
            genomesInfo1 = data["genomes"]

        first_gen_1.append(genomesInfo1[0]["fitness"])
        final_gen_1.append(genomesInfo1[-1]["fitness"])

        finiteHits = []
        infiniteHits = []
        for gen_idx, g in enumerate(genomesInfo1):
            fh = 0
            ih = 0
            for s in g["resHits"]:
                fh += s["metFin"]
                ih += s["infFin"]
            
            try:
                total_finite_hits[gen_idx].append(fh)
            except:
                total_finite_hits.append([fh])

            try:
                total_infinite_hits[gen_idx].append(ih)
            except:
                total_infinite_hits.append([ih])


            try:
                target_size_scores[gen_idx].append(g["sizeFitness"])
            except:
                target_size_scores.append([g["sizeFitness"]])

            try:
                aspect_ratio_scores[gen_idx].append(g["aspectRatioFitness"])
            except:
                aspect_ratio_scores.append([g["aspectRatioFitness"]])

            try:
                red_green_scores[gen_idx].append(g["cellRatioFitness"])
            except:
                red_green_scores.append([g["cellRatioFitness"]])

        for i in range(numGenerations + 1):
            try:
                metabolic_reservoir_caps[i].append(genomesInfo1[i]["metabolicReservoirValues"]["0"])
            except:
                metabolic_reservoir_caps.append([genomesInfo1[i]["metabolicReservoirValues"]["0"]])

        # print(metabolic_reservoir_caps)
            
            # finiteHits.append(fh)
            # infiniteHits.append(ih)
        # ys_f = finiteHits
        # ys_i = infiniteHits

        

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

        # ------------------
        # GRAPHS TO GENERATE 
        # ------------------

        # Keep tabbed 
        # generate_first_generation_pictures(g_first_1, FP1, s_idx, updateHyperParametersF1)
        # generate_first_generation_pictures(g_first_2, FP2, s_idx, updateHyperParametersF2)
        # generate_final_generation_pictures(g_final_1, FP1, s_idx, updateHyperParametersF1)
        # generate_final_generation_pictures(g_final_2, FP2, s_idx, updateHyperParametersF2)
        
        # consistency_difference(g_final_1, g_final_2, FP1, FP2, updateHyperParametersF1, updateHyperParametersF2, 5, s_idx)
        # pixel_similarity_helper(g_final_1, pixel_similarities_f1, updateHyperParametersF1, 30, s_idx)
        # pixel_similarity_helper(g_final_2, pixel_similarities_f2, updateHyperParametersF2, 30, s_idx)

        # if s_idx in [91]:
            # generate_pic_of_each_generation_in_evolution(genomesInfo1, FP1, s_idx, updateHyperParametersF1)
            # generate_pic_of_each_generation_in_evolution(genomesInfo2, FP2, s_idx, updateHyperParametersF2)

        # top_fitness_scores_1_final.append([g_final_1.totalFitness, s_idx])
        # top_fitness_scores_2_final = []
        # top_fitness_scores_2_final.append([g_final_2.totalFitness, s_idx])
        # top_fitness_scores_1_first.append([g_first_1.totalFitness, s_idx])
        # top_fitness_scores_2_first.append([g_first_2.totalFitness, s_idx])
        # individual_fitness_breakdown(s_idx)
        # individual_fitness_graph(s_idx, genomesInfo1)

        # finite_reservoir_capacity_over_time_individual(s_idx, genomesInfo1)
    
    

    # print(np.mean(pixel_similarities_f1))
    # print(np.mean(pixel_similarities_f2))
    # pixel_similarity_graph()
    # print_top_agent_numbers(top_fitness_scores_1_first, top_fitness_scores_2_first, "first")
    # print_top_agent_numbers(top_fitness_scores_1_final, top_fitness_scores_2_final, "final")

    avg_fitness_per_generation(FP1, FP1)
    avg_fitness_per_generation(FP2, FP2)
    # gen_csv_of_first_and_final_gen_scores(first_gen_1, final_gen_1, first_gen_2, final_gen_2)
    # threshold_passing([.8], FP1, FP2)
    # t_test(FP1, FP2, final_gen_1, final_gen_2)
    # compute_permutation_test(FP1, FP2, final_gen_1, final_gen_2)
    # consistency_difference_summary()
    # pixel_vs_fitness_similarity(std_devs_f1, pixel_similarities_f1, "infinite_and_finite")
    # pixel_vs_fitness_similarity(std_devs_f2, pixel_similarities_f2, "infinite only")
    # infinite_and_finite_reservoir_usage()
    # fitness_breakdown()
    # finite_reservoir_capacity_over_time()
    print(min(final_gen_1))
    print(max(final_gen_1))

    print(min(final_gen_2))
    print(max(final_gen_2))


def finite_reservoir_capacity_over_time_individual(s_idx, genomesInfo):

    ys = []
    for gi in genomesInfo:
        try:
            ys.append(gi["metabolicReservoirValues"][0])
        except:
            ys.append(gi["metabolicReservoirValues"]["0"])
        # print(gi["metabolicReservoirValues"][0])

        # ys.append(gi["metabolicReservoirValues"][0])

    xs = list(range(numGenerations + 1))
    fig = plt.figure()
    ax = plt.axes()
    # a_list = [np.mean(x) for x in total_finite_hits]
    # b_list = [np.mean(x) for x in total_infinite_hits]
    # c = [100 * (a / (a + b)) for a, b in zip(a_list, b_list)]
    # ax.plot(xs[:1000], c[:1000])
    
    # print(ys)
    

    # for i in range(numGenerations + 1):
    #     ys.append(gi[i]["metabolicReservoirValues"])

    ax.plot(xs, ys)

    ax.set(xlabel='Generation Number', ylabel="Finite Reservoir Cap", title=str(s_idx))
    ax.grid(True)
    # ax.legend(["Finite", "Infinite"])
    # ax.set_ylim([0, 100])
    plt.savefig("individual_reservoir_caps/" + str(s_idx) + ".png")
    # plt.show()
    plt.clf()


def finite_reservoir_capacity_over_time():
    xs = list(range(numGenerations + 1))
    fig = plt.figure()
    ax = plt.axes()
    # a_list = [np.mean(x) for x in total_finite_hits]
    # b_list = [np.mean(x) for x in total_infinite_hits]
    # c = [100 * (a / (a + b)) for a, b in zip(a_list, b_list)]
    # ax.plot(xs[:1000], c[:1000])
    ys = [np.mean(arr) for arr in metabolic_reservoir_caps[:numGenerations + 1]]
    ax.plot(xs, ys)

    ax.set(xlabel='Generation Number', ylabel="Finite Reservoir Cap", title="fixed finite 5")
    ax.grid(True)
    # ax.legend(["Finite", "Infinite"])
    # ax.set_ylim([0, 100])
    # plt.savefig(SAVE_TO  + "/Avg_Fitness__" + FP + ".png")
    plt.show()
    plt.clf()

def individual_fitness_graph(s_idx, gi):
    xs = list(range(numGenerations + 1))
    ys = []
    for i in range(len(gi)):
        ys.append(gi[i]["fitness"])
    
    ax = plt.gca()
    ax.plot(xs, ys)
    ax.set(xlabel='Generation Number', ylabel='Fitness Score', title=str(s_idx))
    ax.grid(True)
    ax.set_ylim([0, 152])
    plt.savefig("individual_fitnesses/r" + str(s_idx) + ".png")
    plt.clf()
    # plt.show()

def individual_fitness_breakdown(s_idx):
    xs = list(range(numGenerations + 1))
    ys1 = []
    ys2 = []
    ys3 = []
    for k in range(len(target_size_scores)):
        ys1.append(target_size_scores[k][s_idx])
        ys2.append(aspect_ratio_scores[k][s_idx])
        ys3.append(red_green_scores[k][s_idx])
    
    ax = plt.gca()
    ax.plot(xs, ys1)
    ax.plot(xs, ys2)
    ax.plot(xs, ys3)
    ax.set(xlabel='Generation Number', ylabel='Fitness Subscore', title="Individual Fitness Breakdown During Evolution")
    ax.grid(True)
    ax.set_ylim([0, 52])
    ax.legend(["Target Size", "Aspect Ratio", "Red Green Split"])
    plt.savefig("fitness_breakdowns/r" + str(s_idx) + ".png")
    plt.clf()
    # plt.show()

def fitness_breakdown():
    xs = list(range(numGenerations + 1))
    ys1 = []
    ys2 = []
    ys3 = []
    for k in range(len(target_size_scores)):
        ys1.append(np.mean(target_size_scores[k]))
        ys2.append(np.mean(aspect_ratio_scores[k]))
        ys3.append(np.mean(red_green_scores[k]))

    ax = plt.gca()
    ax.plot(xs, ys1)
    ax.plot(xs, ys2)
    ax.plot(xs, ys3)
    ax.set(xlabel='Generation Number', ylabel='Fitness Subscore', title="Average Fitness Breakdown During Evolution")
    ax.grid(True)
    ax.set_ylim([0, 52])
    ax.legend(["Target Size", "Aspect Ratio", "Red Green Split"])
    # plt.savefig(SAVE_TO  + "/Avg_Fitness_smooshed.png")
    # plt.clf()
    plt.show()

def infinite_and_finite_reservoir_usage():
    xs = list(range(numGenerations + 1))
    fig = plt.figure()
    ax = plt.axes()
    a_list = [np.mean(x) for x in total_finite_hits]
    b_list = [np.mean(x) for x in total_infinite_hits]

    p_value = permutation_test(a_list, b_list,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)
    print("infinite_and_finite_reservoir_usage p-value = ", p_value)

    c = [100 * (a / (a + b)) for a, b in zip(a_list, b_list)]

    x = ["Start Of Development"] * numberOfSimulations + ["End Of Development"] * numberOfSimulations
    y = c[0] + c[numGenerations]
    print(x)
    print(y)
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'x': x, 'y': y})

    sns.swarmplot('x', 'y', data=df)
    sns.despine()
        
    plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
    plt.axhline(np.mean(y[numberOfSimulations:]), color='orange', linewidth=2)
    # plt.title("Simulations Using Only Infinite Reservoirs Take Longer to Reach " + str(threshold) + " Percent of Max Fitness")
    plt.title("title")
    plt.ylabel("Remaining Finite Fuel")
    # plt.savefig(SAVE_TO + "/" + "threshold_" + str(threshold) + ".png")
    plt.show()
    plt.clf()

    # ax.set(xlabel='Generation Number', ylabel="% FiniteUsage", title="title")
    # ax.grid(True)
    # # ax.legend(["Finite", "Infinite"])
    # # ax.set_ylim([0, 100])
    # # plt.savefig(SAVE_TO  + "/Avg_Fitness__" + FP + ".png")
    # plt.show()
    # plt.clf()

def print_top_agent_numbers(top_fitness_scores_1, top_fitness_scores_2, x):
    top_fitness_scores_1.sort()
    top_fitness_scores_2.sort()
    print(x, top_fitness_scores_1[49:52])
    print(x, top_fitness_scores_2[49:52])

def avg_fitness_per_generation(FP, graph_title):
    fitnesses = []

    for s_idx in range(numberOfSimulations):
        file = "data/" + FP + "/r" + str(s_idx) + ".json"
        with open(file, "r") as json_file:
            data = json.load(json_file)
            genomesInfo = data["genomes"]

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
    
    print("min: ", min(ys))
    print("max: ", max(ys))

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(xs, ys[:numGenerations + 1])
    ax.set(xlabel='Generation Number', ylabel='Average of Top Fitnesses of 100 Simulations', title=graph_title)
    ax.grid(True)
    ax.set_ylim([80, 150])
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
    plt.savefig(SAVE_TO + "/first_gen_pics_" + FP + "/r" + str(s_idx) + ".png")
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
    # plt.annotate(fitness_breakdown_string, 
    #             (20, 8), # these are the coordinates to position the label
    #             color='blue')
    plt.savefig(SAVE_TO + "/final_gen_pics_" + FP + "/r" + str(s_idx) + ".png")
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
    for i in [0, 100, 500, 2500, 5000, 7499]:
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

        plt.show()
        # plt.savefig(SAVE_TO + "/evolution_pics_" + FP +"/r" + str(s_idx) + "/" + str(i) + '.png')
        plt.clf()

def threshold_passing(thresholds, FP1, FP2):
    for threshold in thresholds:
        ys_a = []
        ys_b = []

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


        p_value = permutation_test(ys_a, ys_b,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)
        # f.write(str(p_value))
        # f.close()
        print("threshold compute_permutation_test, ", str(FP1), str(FP2), str(p_value))
        

        # x = [FP1] * numberOfSimulations + [FP2] * numberOfSimulations
        x = ["Infinite Or Finite"] * numberOfSimulations + ["Infinite Only"] * numberOfSimulations
        y = ys_a + ys_b
        sns.set(style='ticks', context='talk')
        df= pd.DataFrame({'x': x, 'y': y})

        sns.swarmplot('x', 'y', data=df)
        sns.despine()
            
        plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
        plt.axhline(np.mean(y[numberOfSimulations:]), color='orange', linewidth=2)
        # plt.title("Simulations Using Only Infinite Reservoirs Take Longer to Reach " + str(threshold) + " Percent of Max Fitness")
        plt.title("Generations that Passed " + str(threshold * 100) + "%" + " of Max Fitness")
        plt.ylabel("Generation")
        plt.savefig(SAVE_TO + "/" + "threshold_" + str(threshold) + ".png")
        plt.clf()

def consistency_difference(g1, g2, FP1, FP2, update_params_1, update_params_2, repeats, s_idx):
    
    # os.mkdir("new_repeats/inf_or_fin_" + str(s_idx))
    # os.mkdir("new_repeats/inf_only_" + str(s_idx))

    board = brd.Board(hp.boardWidth, hp.boardHeight)
    score = 0
    
    ys_a = []
    ys_b = []

    update_params_1()

    for i in range(repeats):
        board.reset(g1)
        g1.fillReservoirs()
        while (len(board.dynamicCells)):
            board.step()

        data = np.array(board.grid)

        rows,cols = data.shape

        plt.imshow(data, interpolation='none', 
                        extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                        aspect="equal",
                        cmap=my_cmap)
        # plt.show()
        
        # plt.savefig("new_repeats/inf_or_fin_" + str(s_idx) + "/" + str(i) + '.png')
        plt.clf()

        f = fitness.Fitness(board=board)
        f.calculate()
        score = f.totalScore
        ys_a.append(score)

    update_params_2()

    for i in range(repeats):
        board.reset(g2)
        g2.fillReservoirs()
        step = 0
        while (len(board.dynamicCells)):
            board.step()

        data = np.array(board.grid)

        rows,cols = data.shape

        plt.imshow(data, interpolation='none', 
                        extent=[0.5, 0.5+cols, 0.5, 0.5+rows], 
                        aspect="equal",
                        cmap=my_cmap)
        # plt.show()
        # plt.savefig("new_repeats/inf_only_" + str(s_idx) + "/" + str(i) + "_step_" + str(step) + '.png')
        # plt.savefig("new_repeats/inf_only_" + str(s_idx) + "/" + str(i) + '.png')
        plt.clf()
            # step += 1

        f = fitness.Fitness(board=board)
        f.calculate()
        score = f.totalScore
        ys_b.append(score)

    x = [FP1] * repeats + [FP2] * repeats
    y = ys_a + ys_b
    sns.set(style='ticks', context='talk')
    # df= pd.DataFrame({'x': x, 'y': y})

    # sns.swarmplot('x', 'y', data=df)
    # sns.despine()

    std_devs_f1.append(np.std(ys_a))
    std_devs_f2.append(np.std(ys_b))

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
    plt.clf()



def t_test(FP1, FP2, final_gen_1, final_gen_2):
    f = open(SAVE_TO + "/t_test___" + FP1 + "___" + FP2 + ".txt", "w")
    f.write(str(ttest_ind(final_gen_1, final_gen_2)))
    f.close()

def compute_permutation_test(FP1, FP2, final_gen_1, final_gen_2):
    # f = open(SAVE_TO + "/permutation_test___" + FP1 + "___" + FP2 + ".txt", "w")
    p_value = permutation_test(final_gen_1, final_gen_2,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)
    # f.write(str(p_value))
    # f.close()
    print("compute_permutation_test, ", str(FP1), str(FP2), str(p_value))

def consistency_difference_summary():
    plt.rcParams["figure.figsize"] = [2.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    y_f = std_devs_f1
    y_l = std_devs_f2

    p_value = permutation_test(std_devs_f1, std_devs_f2,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)

    print("consistency difference p-value = ", p_value)

    x = ["Infinite or Finite"] * numberOfSimulations + ["Infinite Only"] * numberOfSimulations
    # print(x)
    y = y_f + y_l
    # print(y)
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'x': x, 'y': y})

    sns.swarmplot('x', 'y', data=df)
    sns.despine()

    plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
    plt.axhline(np.mean(y[numberOfSimulations:]), color='orange', linewidth=2)
    plt.title("Finite Reservoirs Enable Consistent Growth")
    plt.ylabel("Std Dev. of Fitness Over 30 Repeats")
    # plt.savefig(SAVE_TO + "/" + "consistency_summary.png")
    plt.show()
    

def pixel_similarity_helper(g, total_scores, update_params, num_repeats, s_idx):
    boards = []

    update_params()
    b = brd.Board(hp.boardWidth, hp.boardHeight)

    for _ in range(num_repeats):
        b.reset(g)
        g.fillReservoirs()
        while (len(b.dynamicCells)):
            b.step()

        boards.append(deepcopy(b))

    spot_percentages = []
    for row in range(len(boards[0].grid)):
        for col in range(len(boards[0].grid[0])):
            white = 0
            black = 0
            green = 0
            red = 0
            for board in boards:
                cell = board.grid[row][col]
                if cell == 0:
                    white += 1
                elif cell == globs.STEM:
                    black += 1
                elif cell == globs.NERVE:
                    green += 1
                elif cell == globs.SKIN:
                    red += 1
            
            # if white == 30:
            #     continue
            
            top_count = max(white, black, red, green)
            spot_percentages.append(top_count / num_repeats)
    
    # total_scores.append(np.mean(spot_percentages))
    total_scores.append(np.std(spot_percentages))

        
    
def pixel_similarity_graph():
    plt.rcParams["figure.figsize"] = [2.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    y_f = pixel_similarities_f1
    y_l = pixel_similarities_f2

    x = ["Infinite or Finite"] * numberOfSimulations + ["Infinite Only"] * numberOfSimulations
    y = y_f + y_l
    sns.set(style='ticks', context='talk')
    df= pd.DataFrame({'x': x, 'y': y})

    sns.swarmplot('x', 'y', data=df)
    sns.despine()

    plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
    plt.axhline(np.mean(y[numberOfSimulations:]), color='orange', linewidth=2)
    plt.title("Consistency, comparing pixels")
    plt.ylabel("Pixel similarity percentage")
    # plt.savefig(SAVE_TO + "/" + "consistency_summary.png")
    plt.show()



def pixel_vs_fitness_similarity(fitness_stds, pixels_stds, title):
    plt.rcParams["figure.figsize"] = [2.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    # y_f = pixel_similarities_f1
    # y_l = pixel_similarities_f2

    x = fitness_stds
    y = pixels_stds


    print(str(title), ", Pearson product-moment correlation coefficient: ", np.corrcoef(x, y))
    # sns.set(style='ticks', context='talk')
    # df= pd.DataFrame({'x': x, 'y': y})

    # sns.swarmplot('x', 'y', data=df)
    # sns.despine()

    plt.scatter(x, y)
    # plt.axhline(np.mean(y[:numberOfSimulations]), color='blue', linewidth=2)
    # plt.axhline(np.mean(y[numberOfSimulations:]), color='orange', linewidth=2)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title("pixel_vs_fitness_similarity: " + str(title))
    plt.xlabel("Fitness Similarity")
    plt.ylabel("Pixel Similarity")
    # plt.savefig(SAVE_TO + "/" + "consistency_summary.png")
    plt.show()

if __name__ == "__main__":
    main()