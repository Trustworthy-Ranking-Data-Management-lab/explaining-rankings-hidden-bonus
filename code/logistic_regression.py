from itertools import combinations
from sklearn.linear_model import LogisticRegression

import numpy as np
import time

from functools import partial
from multiprocessing import Pool

def process_chunk(chunk_indices, data_array, ranking, DIM):
    """
    Processes a subset of pairs and returns local x and y buffers.
    """
    num_pairs = len(chunk_indices)
    x_local = np.empty((num_pairs * 2, DIM))
    y_local = np.empty(num_pairs * 2, dtype=int)
    
    for idx, (i, j) in enumerate(chunk_indices):
        id1 = ranking[i][0]
        id2 = ranking[j][0]
        diff = data_array[id1] - data_array[id2]
        
        # Entry 1: Positive diff -> Label 1
        x_local[idx * 2] = diff
        y_local[idx * 2] = 1
        
        # Entry 2: Negative diff -> Label 0
        x_local[idx * 2 + 1] = -diff
        y_local[idx * 2 + 1] = 0
        
    return x_local, y_local

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
    print("----- Logistic Regression -----")

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

    pairs = NUM_DATA * (NUM_DATA - 1)
    x_train = np.empty((pairs, DIM))
    y_train = np.empty(pairs, dtype=int)

    data_array = np.array(points)
    all_pairs = list(combinations(range(NUM_DATA), 2))
    total_pairs = len(all_pairs)
    
    num_workers = 16
    chunk_size = total_pairs // num_workers
    chunks = [all_pairs[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]

    with Pool(processes=num_workers) as pool:
        func = partial(process_chunk, data_array=data_array, ranking=ranking, DIM=DIM)
        results = pool.map(func, chunks)

    x_train = np.vstack([res[0] for res in results])
    y_train = np.concatenate([res[1] for res in results])

    model = LogisticRegression(fit_intercept=False)
    model.fit(x_train, y_train)

    weights = model.coef_[0].tolist()
    print("Feature Weights (w):", weights)

    #Now, compute the ranking from using weights from linear regression
    scores = np.dot(data_array, weights)
    sorted_indices = np.argsort(scores)[::-1]

    #Relabel the IDs so that the 1st item in input ranking is id1, 2nd item is id2, and so on.
    mapping = {}
    for i in range(len(points)):
        mapping[ranking[i][0]] = i
    renamed_idxs: list[int] = []
    for i in range(len(points)):
        renamed_idxs.append(mapping[int(sorted_indices[i])])

    lcs = do_lis(renamed_idxs)
    addedpts = NUM_DATA - len(lcs)
    print("Boosted:", addedpts)
    timetaken = time.time() - start_time

    #to do: for g-gblr, need to check how many groups are needed
    reverse_mapping = {}
    for key in mapping:
        reverse_mapping[mapping[key]] = key
    lcsid = [reverse_mapping[i] for i in lcs]
    ranges = []
    for i in range(NUM_DATA):
        idx = ranking[i][0]
        if idx not in lcsid:
            #compute the additive amount based off nearest neighbours in lcs
            #note: prev is higher in ranking (more score), next is lower in ranking (less score)
            prev = i - 1
            next = i + 1
            while ranking[prev][0] not in lcsid and prev >= 0:
                prev -= 1
            while ranking[next][0] not in lcsid and next < NUM_DATA:
                next += 1
            thisbound = []
            if next < NUM_DATA:
                nextidx = ranking[next][0]
                thisbound.append(scores[nextidx] - scores[idx])
            else:
                thisbound.append(0)
            if prev >= 0:
                previdx = ranking[prev][0]
                thisbound.append(scores[previdx] - scores[idx])
            else:
                thisbound.append(10000)
            ranges.append(thisbound)

    ranges.sort(key = lambda tup : tup[0])
    addvals: list[float] = []
    upper_so_far = ranges[0][1]
    for i in range(len(ranges)):
        if ranges[i][0] > upper_so_far:
            addvals.append(upper_so_far)
            upper_so_far = ranges[i][1]
        else:
            upper_so_far = min(upper_so_far, ranges[i][1])
    print(len(addvals))

    print(f"Time taken: {timetaken:.4f} seconds")
    print(f"{NUM_DATA}, {DIM}, {NUM_GROUP}, {K}, -- Dominance ignored --, Logistic Regression, {timetaken}, {addedpts}, {len(addvals)}")
    print("------------ End of Algorithm run -----------")