import gurobipy as gp
from gurobipy import GRB

import numpy as np
import time
from functools import partial

import dominance_check

from multiprocessing import Pool

#numerical precision parameter
#will need to be adjusted, based off the range of the input values as it needs to be of comparable enough accuracy
EPSILON = 1e-5
TLIMIT = 1800

if __name__ == "__main__":
    print("----- ILP Refined for groups -----")
    points = []
    tmp = input().rstrip().split(" ")
    parameters = [int(i) for i in tmp]
    NUM_DATA = parameters[0]

    
    Klist = parameters[1:]
    K = sum(Klist)
    NUM_GROUP = len(Klist)

    for i in range(NUM_DATA):
        point = input().rstrip().split(" ")
        point = [float(i) for i in point]
        points.append(point)

    DIM = len(points[0])
    print("Input parameters:")
    print("Number of points:", NUM_DATA)
    print("Dim =", DIM)
    print("Number of groups =", NUM_GROUP)
    print("K =", K)
    
    #Read the rankings
    #ranking[i][0] is the id
    #ranking[i][1] is the rank position (because of ties, this is not necessarily consecutive)
    #ranking[i][2] is the indicator for belonging to privileged group
    ranking: list[list[int]] = []
    for i in range(NUM_DATA):
        tmp = input().rstrip().split(" ")
        rankpos : list[int] = [int(i) for i in tmp]
        ranking.append(rankpos)

    #start time after input read
    start_time = time.time()

    m = gp.Model("ilprefined")
    m.params.Threads = 16
    m.params.DisplayInterval = 60
    # Create variables
    weights = m.addMVar(shape=DIM, name="weights")
    additive_vars = m.addMVar(shape = NUM_GROUP)
    indicator_vars = []
    for i in range(NUM_DATA):
        indicator_vars.append(m.addMVar(shape = NUM_GROUP, vtype=GRB.BINARY, name="indicators"))
    
    # Set objective
    m.setObjective(0, GRB.MINIMIZE)
    
    
    #one constraint per adjacent pair of ranked elements
    for i in range(1, NUM_DATA):
        
        firstid = ranking[i-1][0]
        secondid = ranking[i][0]
        coeff = np.array([points[firstid][j] - points[secondid][j] for j in range(DIM)])
        
        if ranking[i-1][1] == ranking[i][1]:
            m.addConstr(coeff @ weights + indicator_vars[i-1] @ additive_vars - indicator_vars[i] @ additive_vars == 0)
        else:
            m.addConstr(coeff @ weights + indicator_vars[i-1] @ additive_vars - indicator_vars[i] @ additive_vars >= EPSILON)
    
    #Add constraints on the additive value for speedup
    m.addConstrs(additive_vars[i] <= 100 for i in range(NUM_GROUP))
    
    m.addConstr(gp.quicksum([i.sum() for i in indicator_vars]) <= K)
    for i in range(NUM_DATA):
        m.addConstr(gp.quicksum(indicator_vars[i]) <= 1)
    
    # Applies skyline preprocessing to check for dominated points
    # Add constraint that dominated points are guaranteed to be boosted
    # Does it twice for 2 groups as discussed in the paper; currently hardcoded for that case
    with Pool(processes=16, initializer = dominance_check.init_globals, initargs = (points, ranking, DIM)) as pool:
    
        initset = [i for i in range(NUM_DATA - 1)]
        sols = pool.map(dominance_check.check_dominated_full, initset)
        levelset = [i for i in initset if sols[i]]
        domcount = []
        if len(Klist) > 1 and len(levelset) > 0:
            partial_check_dominated = partial(dominance_check.check_dominated, pos = levelset)
            sols = pool.map(partial_check_dominated, levelset)
            secondset = [levelset[i] for i in range(len(levelset)) if sols[i]]
            oldset = [levelset[i] for i in range(len(levelset)) if not sols[i]]
            cdomcount = 0
            for i in oldset:
                m.addConstr(gp.quicksum(indicator_vars[i]) == 1)
                cdomcount += 1
            domcount.append(cdomcount)
            cdomcount = 0
            for i in secondset:
                m.addConstr(indicator_vars[i][1] == 1)
                cdomcount += 1
            domcount.append(cdomcount)
        else:
            cdomcount = 0
            for i in levelset:
                m.addConstr(gp.quicksum(indicator_vars[i]) == 1)
                cdomcount += 1
            domcount.append(cdomcount)
    
    points = np.array(points)

    constr_time = time.time()
    print("Time taken to setup constraints and dominance pre-processing: ", str(round(constr_time - start_time, 2)) + "s")
    print("Problem initialised. Solving...")
    # Optimize model

    #set time remaining after doing the preprocessing
    m.params.TimeLimit = TLIMIT - int(time.time() - start_time)

    m.optimize()

    print("Time taken to solve: ", str(round(time.time() - constr_time, 2)) + "s")
    print("Time taken in total: ", str(round(time.time() - start_time, 2)) + "s")
    
    timetaken = time.time() - start_time
    try:
        print("Weights:", weights.X)
        print("Additive values: ", additive_vars.X)

        sol_weights = weights.X

        # Does a check that solution produced indeed ranks the candidates in decreasing score correctly.
        prev_val = 1e7
        disagreement_count = 0
        gsizes = [0] * NUM_GROUP
        indicators = [indicator_vars[i].X for i in range(NUM_DATA)]
        additive_vars = additive_vars.X
        for i in range(NUM_DATA):
            item = ranking[i][0]
            val = 0
            group = 0
            for j in range(DIM):
                val += points[item][j] * sol_weights[j]
            for j in range(NUM_GROUP):
                if indicators[i][j] > 0.1:
                    val += additive_vars[j]
                    gsizes[j] += 1
            if val - EPSILON >= prev_val and ranking[i][1] != ranking[i-1][1]:
                print("Inconsistent for > with ground truth at element ranked", ranking[i][1])
                print(ranking[i-1][0], prev_val)
                print(item, val)
            elif ranking[i][1] == ranking[i-1][1] and abs(val - prev_val) >= EPSILON:
                print("Inconsistent for tie with ground truth at element ranked", ranking[i][1])
                print(ranking[i-1][0], prev_val)
                print(item, val)
            prev_val = val
            
        print("Group sizes:", gsizes)
        addedpts = sum(gsizes)
        print("Points given additive bonus:", addedpts)
        print(f"Final line, {NUM_DATA}, {DIM}, {NUM_GROUP}, {K}, {domcount}, ILPrefined, {round(timetaken,2)}, {addedpts}")
        print("------------ End of Algorithm run -----------")
    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        print(f"Final line, {NUM_DATA}, {DIM}, {NUM_GROUP}, {K}, {domcount}, ILPrefined, {round(timetaken,2)}, {1000000000}")
        print("------------ End of Algorithm run -----------")
