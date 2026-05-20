import gurobipy as gp
from gurobipy import GRB

import numpy as np
import time
from functools import partial

import dominance_check

from multiprocessing import Pool


#numerical precision parameter
EPSILON = 1e-5
TLIMIT = 1800

if __name__ == "__main__":
    try:
        print("----- ILP Refined for singleton -----")
        points: list[list[float]] = []
        tmp = input().rstrip().split(" ")
        parameters : list[int] = [int(i) for i in tmp]
        NUM_DATA = parameters[0]
        K = parameters[1]
        
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
            try:
                rankpos = input().rstrip().split(" ")
                rankpos = [int(i) for i in rankpos]
                ranking.append(rankpos)
            except:
                continue

        print("Input parameters:")
        print("Number of points:", NUM_DATA)
        print("Number of dimensions:", DIM)
        print("Number of perturbed points allowed:", K)

        #start time after input read
        start_time = time.time()

        m = gp.Model("refined_singleton")
        m.params.Threads = 16
        m.params.DisplayInterval = 60

        # Create variables
        weights = m.addMVar(shape=DIM, name="weights")
        additive_vars = m.addMVar(shape = NUM_DATA, ub = 100)
        indicator_vars = m.addMVar(shape = NUM_DATA, vtype=GRB.BINARY, name="indicators")
        
        # Set objective
        m.setObjective(gp.quicksum(indicator_vars), GRB.MINIMIZE)

        #one constraint per adjacent pair of ranked elements
        for i in range(NUM_DATA - 1):
            
            first_id = ranking[i][0]
            second_id = ranking[i+1][0]
            coeff = np.array([points[first_id][j] - points[second_id][j] for j in range(DIM)])
            
            if ranking[i][1] == ranking[i+1][1]:
                m.addConstr(coeff @ weights + indicator_vars[i] * additive_vars[i] - indicator_vars[i+1] * additive_vars[i+1] == 0)
            else:
                m.addConstr(coeff @ weights + indicator_vars[i] * additive_vars[i] - indicator_vars[i+1] * additive_vars[i+1] >= EPSILON)
        
        #check for dominated points and set privileged
        pool = Pool(processes=16, initializer = dominance_check.init_globals, initargs = (points, ranking, DIM))
    
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
        
        m.params.TimeLimit = TLIMIT - int(time.time() - start_time)
        
        # Optimize model
        m.optimize()

        print("Time taken to solve: ", str(round(time.time() - constr_time, 2)) + "s")
        print("Time taken in total: ", str(round(time.time() - start_time, 2)) + "s")
        timetaken = time.time() - start_time
        print(weights.X)

        add_var = additive_vars.X

        addedpts = sum(indicator_vars.X)
        print("Privileged set size:", addedpts)
        sol_weights = weights.X

        # Does a check that solution produced indeed ranks the candidates in decreasing score correctly.
        prev_val = 1e7
        disagreement_count = 0
        indicator_vars = indicator_vars.X
        for i in range(NUM_DATA):
            item = ranking[i][0]
            val = 0
            for j in range(DIM):
                val += points[item][j] * sol_weights[j]
            if indicator_vars[i] > 0.1:
                val += add_var[i]
            if val - EPSILON >= prev_val:
                print("Inconsistent with ground truth at element ranked", ranking[i][1])
                print(ranking[i-1][0], prev_val)
                print(item, val)
            prev_val = val
            
        print("Points given additive bonus:", addedpts)
        print(f"Final line, {NUM_DATA}, {DIM}, Singleton, {K}, {domcount}, ILPrefined, {round(timetaken,2)}, {addedpts}")
        print("------------ End of Algorithm run -----------")
        
    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
