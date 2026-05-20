import gurobipy as gp
from gurobipy import GRB

import numpy as np
import time

# Standard implementation to find LIS
def do_lis(idxs):
    # predecessors[i] stores the index of the element that comes
    # *before* idxs[i] in the increasing subsequence ending at idxs[i].
    predecessors = [-1] * NUM_DATA

    tails: list[int] = []

    for i in range(NUM_DATA):
        num = idxs[i]
        j = len(tails)
        left: int = 0
        right: int = len(tails) - 1
        while left <= right:
            mid: int = (left + right) // 2
            if tails[mid] <= num:
                left = mid + 1
            else:
                if mid > 0 and tails[mid-1] <= num or mid == 0:
                    j = mid
                    break
                right = mid - 1

        if j == len(tails):
            tails.append(num)
        else:
            tails[j] = num

        # Update the predecessor for num
        # Its predecessor is the last element of the subsequence
        # of length 'j' (which is at tails_indices[j-1]).
        if j > 0:
            predecessors[num] = tails[j - 1]

    # Reconstruct the LIS by backtracking from the last element.
    lis: list[int] = []
    num = tails[-1]
    
    while num != -1:
        lis.append(num)
        num = predecessors[num]

    # The list is built backward, so we reverse it.
    return lis[::-1]

if __name__ == "__main__":
    print("----- Ordinal Regression -----")

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

    dataarray = np.array(points)
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
    points = np.array(points)

    m = gp.Model("ilprefined")
    m.params.Threads = 16
    m.params.DisplayInterval = 60

    #This implements the dual algorithm described in Linear Programming Computational Procedures for Ordinal Regression (Srinivasan, 1976)
    #That is, LP2 of the paper
    mu = m.addVar(name="mu")
    m.setObjective(mu, GRB.MAXIMIZE)
    pairs = NUM_DATA * NUM_DATA
    mu_arr = [m.addVar(ub=1.0, name=f"mu_arr_{j}_{k}") for j in range(NUM_DATA) for k in range(NUM_DATA)]
    for d in range(DIM):
        mu_term = gp.quicksum((points[ranking[j][0], d] - points[ranking[k][0], d]) * mu_arr[j * NUM_DATA + k] for j in range(NUM_DATA) for k in range(j+1, NUM_DATA))
        mu_coeff = sum(points[ranking[j][0], d] - points[ranking[k][0], d] for j in range(NUM_DATA) for k in range(j+1, NUM_DATA))
        m.addConstr(mu_term + mu_coeff * mu <= 0, name=f"constr_{d}")

    m.params.TimeLimit = 1800 - int(time.time() - start_time)
    m.optimize()

    #Now, compute the ranking from using weights from linear regression
    #Since dual is solved, need to obtain weights from the dual variables of the constraints
    weights = []
    for d in range(DIM):
        constr = m.getConstrByName(f"constr_{d}")
        weights.append(constr.Pi)
    scores = np.dot(points, np.array(weights))
    sorted_indices = np.argsort(scores)[::-1]

    #Relabel the IDs so that the 1st item in input ranking is 1, 2nd item is 2, and so on.
    #Then this allows us to reduce the problem of finding LCS of two rankings to finding LIS of the relabeled ranking from using learned weights
    mapping = {}
    for i in range(len(points)):
        mapping[ranking[i][0]] = i
    renamed_idxs: list[int] = []
    for i in range(len(points)):
        renamed_idxs.append(mapping[int(sorted_indices[i])])

    addedpts = NUM_DATA - len(do_lis(renamed_idxs))
    print("Boosted:", addedpts)
    timetaken = time.time() - start_time

    print(f"Time taken: {timetaken:.4f} seconds")
    print(f"Final line, {NUM_DATA}, {DIM}, {NUM_GROUP}, {K}, -- Dominance ignored --, Linear Regression, {timetaken}, {addedpts}")
    print("------------ End of Algorithm run -----------")