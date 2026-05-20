import time
import sys
from multiprocessing import Pool
from functools import partial
import numpy as np
import dominance_check

import ermb_functions

TLIMIT = 1800
planes = []
ranking = []
points = []
mapping: dict[int, int] = {}

if __name__ == '__main__':
    
    print("----- ERMB for 2D -----")

    tmp: list[str]= input().rstrip().split(" ")
    parameters: list[int] = [int(i) for i in tmp]
    NUM_DATA: int = parameters[0]
    K: int = parameters[1]

    for i in range(NUM_DATA):
        tmp = input().rstrip().split(" ")
        point = [float(i) for i in tmp]
        points.append(point)

    DIM = len(points[0])

    print("Input parameters:")
    print("Number of points:", NUM_DATA)
    print("Dim =", DIM)
    print("K =", K)
    
    if (DIM > 2):
        sys.exit("Dimension must be 2.")
    
    #Read the rankings
    #ranking[i][0] is the id
    #ranking[i][1] is the rank position (because of ties, this is not necessarily consecutive)
    #ranking[i][2] is the indicator for belonging to privileged group
    
    for i in range(NUM_DATA):
        tmp = input().rstrip().split(" ")
        rankpos: list[int] = [int(i) for i in tmp]
        ranking.append(rankpos)

    #Set up rng seed here
    rng = np.random.default_rng([K, K, DIM, NUM_DATA])
    
    #Compute begins here, start timer here
    startTime = time.time()
    
    dominated_count = 0
    with Pool(processes=16, initializer = dominance_check.init_globals, initargs = (points, ranking, DIM)) as pool:
        initset: list[int] = [i for i in range(NUM_DATA)]
        sols = pool.map(dominance_check.check_dominated_full, initset)
        remaining_points = [i for i in initset if not sols[i]]
        for i in range(len(sols)):
            if sols[i] == True:
                dominated_count += 1

    with Pool(processes=8, initializer = ermb_functions.init_globals, initargs = ([], points, [], {}, K)) as pool:
        partial_generateplanes = partial(ermb_functions.generateplanes_2d_sample, sample = remaining_points)
        nestedplanes = pool.map(partial_generateplanes, range(NUM_DATA))
    
    #initialize 0 to define 'leftmost' region
    planes = [0]
    for p in nestedplanes:
        planes += p
    planes.sort()
            
    print("Final number of planes:", len(planes))

    #"Uniformly" sample regions by picking a random permutation of them and then inspecting them in that order
    #To try improving anytime performance
    shuffle = np.arange(len(planes) - 1)
    rng.shuffle(shuffle)

    #Finally, construct the renaming of elements to 0, 1, 2, ..., n-1 in their order in input
    for i in range(len(points)):
        mapping[ranking[i][0]] = i

    points = np.array(points)

    with Pool(processes=16, initializer = ermb_functions.init_globals, initargs = (planes, points, ranking, mapping, K)) as pool:
        partialsolve = partial(ermb_functions.solve_2d, tlimit = TLIMIT, stime = startTime)
        sols = pool.map(partialsolve, shuffle.tolist(), chunksize = NUM_DATA // 16)
    
    maximum = 1e9
    optsols = []
    for i in range(len(sols)):
        if sols[i][0] < maximum:
            maximum = sols[i][0]
            optsols = [sols[i][1]]
    timetaken = time.time() - startTime
    print("Time taken (s):", timetaken)
    print("Best objective found:", maximum)
    print("Weight vectors giving the solution:", optsols)
    print(f"Final line, {NUM_DATA}, {DIM}, Singleton, {K}, {dominated_count} , ERMB, {round(timetaken,2)}, {maximum}")
    print("------------ End of Algorithm run -----------")