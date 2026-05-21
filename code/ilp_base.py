import gurobipy as gp
from gurobipy import GRB

import numpy as np
import time

from functools import partial
from multiprocessing import Pool

import dominance_check

TLIMIT = 1800

if __name__ == "__main__":
    print("----- ILP Base for groups -----")

    points = []
    parameters = input().rstrip().split(" ")
    parameters = [int(i) for i in parameters]
    NUM_DATA = parameters[0]
    Klist = parameters[1:]

    K = sum(Klist)

    NUM_GROUP = len(Klist)
    
    for i in range(NUM_DATA):
        point = input().rstrip().split(" ")
        point = [float(i) for i in point]
        points.append(point)

    DIM = len(points[0])

    #Read the rankings
    #ranking[i][0] is the id
    #ranking[i][1] is the rank position (because of ties, this is not necessarily consecutive)
    #ranking[i][2] is the indicator for belonging to privileged group
    ranking = []
    for i in range(NUM_DATA):
        rankpos = input().rstrip().split(" ")
        rankpos = [int(i) for i in rankpos]
        ranking.append(rankpos)

    print("Input parameters:")
    print("Number of points:", NUM_DATA)
    print("Number of dimensions:", DIM)
    print("Group sizes:", K)
    
    #start time after input read
    start_time = time.time()

    m = gp.Model("ilpbase")
    m.params.Threads = 16
    m.params.DisplayInterval = 60

    # Create variables
    additive_vars = []
    
    weights = m.addMVar(shape=DIM, name="weights", lb = -GRB.INFINITY)
    additive_vars = m.addMVar(shape = NUM_GROUP, lb = -GRB.INFINITY)
    indicator_vars = []
    for i in range(NUM_DATA):
        indicator_vars.append(m.addMVar(shape = NUM_GROUP, vtype=GRB.BINARY, name="indicators"))
    
    # Set objective
    m.setObjective(0, GRB.MINIMIZE)

    #Constraint per adjacent pair of ranked elements that higher ranked element has at least higher score
    for i in range(1, NUM_DATA):
        first_id = ranking[i-1][0]
        second_id = ranking[i][0]
        coeff = np.array([points[first_id][j] - points[second_id][j] for j in range(DIM)])
        m.addConstr(coeff @ weights + indicator_vars[i-1] @ additive_vars - indicator_vars[i] @ additive_vars >= 0)

    #Constraints to enforce that every weight has absolute value at least 1.
    nonzero_vars = m.addMVar(shape = 2 * DIM, vtype = GRB.BINARY)
    for i in range(DIM):
        m.addConstr(nonzero_vars[2*i] + nonzero_vars[2*i+1] == 1)
        m.addConstr((nonzero_vars[2*i] == 1) >> (weights[i] >= 1))
        m.addConstr((nonzero_vars[2*i+1] == 1) >> (weights[i] <= -1))
    
    # Constraint of number of boosted tuples
    m.addConstr(gp.quicksum([i.sum() for i in indicator_vars]) <= K)

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

    for i in range(NUM_DATA):
        m.addConstr(gp.quicksum(indicator_vars[i]) <= 1)
    
    constr_time = time.time()
    print("Time taken to setup constraints: ", str(round(constr_time - start_time, 2)) + "s")
    print("Problem initialised. Solving...")
    
    #set time remaining after doing the preprocessing
    m.params.TimeLimit = TLIMIT - int(time.time() - start_time)
    m.optimize()

    print("Time taken to solve: ", str(round(time.time() - constr_time, 2)) + "s")
    print("Time taken in total: ", str(round(time.time() - start_time, 2)) + "s")
    timetaken = time.time() - start_time

    try:
        sol_weights = weights.X
        sol_additive = additive_vars.X

        print("Weights:", sol_weights)
        print("Additive values: ")
        print(sol_additive)
        indicators = [indicator_vars[i].X.tolist() for i in range(NUM_DATA)]

        # Does a check that solution produced indeed ranks the candidates in decreasing score correctly.
        prev_val = 1e7
        disagreement_count = 0
        gsizes = [0] * NUM_GROUP
        for i in range(0, len(ranking)):
            item = ranking[i][0]
            val = 0
            
            for j in range(DIM):
                val += points[item][j] * sol_weights[j]
            for j in range(NUM_GROUP):
                if indicators[i][j] > 0.1:
                    val += sol_additive[j]
                    gsizes[j] += 1
            if val > prev_val:
                print("Inconsistent, to ground truth at element ranked", ranking[i][1])
                print(ranking[i-1][0], prev_val)
                print(item, val)
            prev_val = val
                
        print("Group sizes:", gsizes)
        print("Points given additive bonus:", sum(gsizes))
        addedpts = sum(gsizes)
        print(f"{NUM_DATA}, {DIM}, {NUM_GROUP}, {K}, {domcount}, ILPbase, {timetaken}, {addedpts}")
        print("------------ End of Algorithm run -----------")
    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        print(f"{NUM_DATA}, {DIM}, {NUM_GROUP}, {K}, {domcount}, ILPbase, {timetaken}, {1000000000}")
        print("------------ End of Algorithm run -----------")
