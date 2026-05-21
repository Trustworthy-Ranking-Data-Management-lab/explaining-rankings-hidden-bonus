import gurobipy as gp
from gurobipy import GRB

import numpy as np
import time

from multiprocessing import Pool

import dominance_check

TLIMIT = 1800

if __name__ == "__main__":
    print("ILPbase Singleton")
    #numerical precision parameter
    EPSILON = 1e-5

    points = []
    parameters = input().rstrip().split(" ")
    parameters = [int(i) for i in parameters]
    NUM_DATA = parameters[0]
    K = parameters[1:]

    NUM_GROUP = len(K)

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
    additive_vars = m.addMVar(shape = NUM_DATA, lb = -GRB.INFINITY)
    indicator_vars = m.addMVar(shape = NUM_DATA, vtype=GRB.BINARY, name="indicators")

    # Set objective
    m.setObjective(gp.quicksum(indicator_vars), GRB.MINIMIZE)

    start_time = time.time()
    #one constraint per adjacent pair of ranked elements

    for i in range(1, NUM_DATA):
        first_id = ranking[i-1][0]
        second_id = ranking[i][0]
        coeff = np.array([points[first_id][j] - points[second_id][j] for j in range(DIM)])
        m.addConstr(coeff @ weights + indicator_vars[i-1] * additive_vars[i-1] - indicator_vars[i] * additive_vars[i] >= 0)

    #make every weight at least 1, or less than -1 strictly.
    nonzero_vars = m.addMVar(shape = 2 * DIM, vtype = GRB.BINARY)
    for i in range(DIM):
        m.addConstr(nonzero_vars[2*i] + nonzero_vars[2*i+1] == 1)
        m.addConstr((nonzero_vars[2*i] == 1) >> (weights[i] >= 1))
        m.addConstr((nonzero_vars[2*i+1] == 1) >> (weights[i] <= -1))

  
    with Pool(processes=16, initializer = dominance_check.init_globals, initargs = (points, ranking, DIM)) as pool:
        sols = pool.map(dominance_check.check_dominated_full, range(NUM_DATA))
        domcount: int = 0
        for i in range(len(sols)):
            if sols[i]:
                m.addConstr(indicator_vars[i] == 1)
                domcount += 1
        print("Dominated points:", domcount)

    constr_time = time.time()
    print("Time taken to setup constraints: ", str(round(constr_time - start_time, 2)) + "s")
    print("Problem initialised. Solving...")

    #set time remaining after doing the preprocessing
    m.params.TimeLimit = TLIMIT - int(time.time() - start_time)

    # Optimize model
    m.optimize()

    print("Time taken to solve: ", str(round(time.time() - constr_time, 2)) + "s")
    print("Time taken in total: ", str(round(time.time() - start_time, 2)) + "s")
    timetaken = time.time() - start_time

    try:
        sol_weights = weights.X
        add_var = additive_vars.X

        print("Weights:", sol_weights)
        indicators = indicator_vars.X.tolist()
        addedpts = sum(indicator_vars.X)
        print("Privileged set size:", addedpts)

        # Does a check that solution produced indeed ranks the candidates in decreasing score correctly.
        prev_val = 1e7
        disagreement_count = 0
        gsizes = [0] * NUM_GROUP
        for i in range(0, len(ranking)):
            item = ranking[i][0]
            val = 0
            for j in range(DIM):
                val += points[item][j] * sol_weights[j]
            
            if indicators[i] > 0.1:
                val += add_var[i]
                
            if val - EPSILON > prev_val:
                print("Inconsistent, to ground truth at element ranked", ranking[i][1])
                print(ranking[i-1][0], prev_val)
                print(item, val)
            prev_val = val
        
        print(f"Final line, {NUM_DATA}, {DIM}, Singleton, {K}, {domcount}, ILPbase, {round(timetaken,2)}, {addedpts}")
        print("------------ End of Algorithm run -----------")
    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        print(f"{NUM_DATA}, {DIM}, Singleton, {K}, {domcount}, ILPbase, {round(timetaken,2)}, {1000000000}")
        print("------------ End of Algorithm run -----------")
