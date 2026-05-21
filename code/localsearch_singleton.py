import time
import sys
from multiprocessing import Pool
import numpy as np
import random
from functools import partial
import numpy.typing as npt

import localsearch_functions

SOLS = []

TLIMIT = 1800

NUM_DATA = 0
K = 0
DIM = 0
ranking = None
points = None
mapping = None
rng = None

def init_globals(points_in, ranking_in, mapping_in, kval):
    global points, ranking, mapping, K, DIM, NUM_DATA, rng

    points = points_in
    ranking = ranking_in
    mapping = mapping_in
    K = kval
    DIM = len(points_in[0])
    NUM_DATA = len(points_in)

    rng = np.random.default_rng([K, K, DIM, NUM_DATA])

if __name__ == '__main__':

    print("----- Local Search Singletons -----")

    points = []
    ranking = []
    parameters = input().rstrip().split(" ")
    parameters = [int(i) for i in parameters]
    NUM_DATA = parameters[0]
    NUM_GROUPS = 1
    K = parameters[1]

    print("Input data has", NUM_DATA, "points. K =", K)
    for i in range(NUM_DATA):
        point = input().rstrip().split(" ")
        point = [float(i) for i in point]
        points.append(point)

    DIM = len(points[0])
    
    #Read the rankings
    #ranking[i][0] is the id
    #ranking[i][1] is the rank position (because of ties, this is not necessarily consecutive)
    #ranking[i][2] is the indicator for belonging to privileged group
    
    for i in range(NUM_DATA):
        try:
            rankpos = input().rstrip().split(" ")
            rankpos = [int(i) for i in rankpos]
            ranking.append(rankpos)
        except:
            continue

    startTime = time.time()
    
    sol = 1e9
    optsols = []
    mapping = {}
    for i in range(len(points)):
        mapping[ranking[i][0]] = i

    solved = 0
    with Pool(processes=16, initializer = localsearch_functions.init_globals, initargs = (points, ranking, mapping, K)) as pool:
        dosolvepartial = partial(localsearch_functions.dosolve, startTime = startTime, timelimit = TLIMIT)
        solutions = pool.map(dosolvepartial, [i for i in range(16)])
    for solution in solutions:
        solved += solution[0]
        if solution[1] < sol:
            sol = solution[1]
            optsols = solution[2]

    print("Sampled", solved, "weight vectors in time limit of", TLIMIT)
    print("Best objective found:", sol)
    print("Weight vectors giving the best objective found:", optsols)
    timetaken = time.time() - startTime
    print(f"{NUM_DATA}, {DIM}, Singleton, {K}, - No dominance - , Localsearch, {round(timetaken,2)}, {sol}")
    print("------------ End of Algorithm run -----------")