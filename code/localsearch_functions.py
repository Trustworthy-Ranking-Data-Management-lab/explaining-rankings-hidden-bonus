import numpy as np
import numpy.typing as npt
import time
from numpy.random import Generator

TLIMIT = 1800

NUM_DATA: int = 0
K: int = 0
DIM: int = 0
ranking: list[list[int]]
points: npt.NDArray[np.float64]
mapping: dict[int, int]
rng: Generator
NUM_GROUPS: int = 0

def compute_objective(weights: npt.NDArray[np.float64]) -> float:
    values = points @ weights
    sorted_indices = np.flip(np.argsort(values))

    idxs = np.arange(len(points))
    idxs = idxs[sorted_indices]
    
    #Relabel the IDs so that the 1st item in input ranking is id1, 2nd item is id2, and so on.
    renamed_idxs: list[int] = []
    for i in range(len(points)):
        renamed_idxs.append(mapping[int(idxs[i])])
    
    distance: int = len(points) - do_lis(renamed_idxs)
    return distance

#Standard patience sort for LIS
def do_lis(idxs: list[int]) -> int:
    tails: list[int] = []

    for num in idxs:
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

    return len(tails)

def do_lis_full(idxs: list[int]) -> list[int]:
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

def dosolve(i: int, startTime: float, timelimit: float) -> tuple[int, float, list[float]]:
    global rng
    rng = np.random.default_rng([NUM_GROUPS, K, DIM, NUM_DATA, i])
    maximum = 1e9
    optsols = []
    solved = 0
    while time.time() - startTime < timelimit:
        weights = sample_KD(DIM)
        result = compute_objective(weights)
        if result < maximum:
            maximum = result
            optsols = weights.tolist()
            print(f"New solution found with objective: {maximum}. Time: {round(time.time() - startTime, 2)}.")
            print(f"Weights: {optsols}")
        solved += 1
    return (solved, maximum, optsols)

def dosolve_groups(i: int, startTime: float, timelimit: float) -> tuple[int, float, list[float]]:
    global rng
    rng = np.random.default_rng([K, K, DIM, NUM_DATA, i])
    maximum = 1e9
    optsols = []
    solved = 0
    while time.time() - startTime < timelimit:
        weights = sample_KD(DIM)
        result = compute_objective_groups(weights)
        if result < maximum:
            maximum = result
            optsols = weights.tolist()
            print(f"New solution found with objective: {maximum}. Time: {round(time.time() - startTime, 2)}.")
            print(f"Weights: {optsols}")
        solved += 1
    return (solved, maximum, optsols)

def compute_objective_groups(weights: npt.NDArray[np.float64]) -> float:
    values = points @ weights
    sorted_indices = np.flip(np.argsort(values))

    idxs = np.arange(len(points))
    idxs = idxs[sorted_indices]
    
    #Relabel the IDs so that the 1st item in input ranking is id1, 2nd item is id2, and so on.
    renamed_idxs: list[int] = []
    for i in range(len(points)):
        renamed_idxs.append(mapping[int(idxs[i])])
    
    lis = do_lis_full(renamed_idxs)
    if NUM_DATA - len(lis) > K:
        return 1e9
    lisset = set(lis)
    #now for each element, e.g. i, not in the LIS, find the closest elements to it that are closest in the LIS
    #These can be used to find a range for the additive value needed to be given to i to place it in the right position in ranking
    #then, find them in this ranking as well, to determine the minimum and maximum bonus that needs to be given to the point to place it in the right place
    #then, sort the ranges. Greedily place additive bonus to overlap as many ranges as possible, and if we need to place too many - infeasible solution
    reverse_mapping: dict[int, int] = {}
    for key in mapping:
        reverse_mapping[mapping[key]] = key
    ranges = []
    
    #to generate ranges
    for element in range(NUM_DATA):
        if element in lisset:
            continue
        #find most recent previous element in LIS
        
        prev_elem = element - 1
        while prev_elem not in lisset and prev_elem >= 0:
            prev_elem -= 1
            
        #find the most recent next element in LIS
        next_elem = element + 1
        while next_elem not in lisset and next_elem <= NUM_DATA:
            next_elem += 1
        #now compute range of additive bonus
        element_pos = reverse_mapping[element]
        element_value = float(values[element_pos])
        if prev_elem >= 0:
            prev_pos = reverse_mapping[prev_elem]
            prev_value = float(values[prev_pos])
        else:
            prev_value = -1e6
        if next_elem <= NUM_DATA - 1:
            next_pos = reverse_mapping[next_elem]
            next_value = float(values[next_pos])
        else:
            next_value = 1e6
        range_small = next_value - element_value
        range_large = prev_value - element_value
        ranges.append([range_small, range_large])
    ranges.sort(key = lambda tup : tup[0])
    addvals: list[float] = []
    upper_so_far = ranges[0][1]
    for i in range(len(ranges)):
        if ranges[i][0] > upper_so_far:
            addvals.append(upper_so_far)
            upper_so_far = ranges[i][1]
        else:
            upper_so_far = min(upper_so_far, ranges[i][1])
    if len(addvals) > NUM_GROUPS:
        return 1e9
    
    #if feasible, return the number of elements not in LIS
    return NUM_DATA - len(lis)

#Uniformly at random sample weights
def sample_KD(dim: int) -> npt.NDArray[np.float64]:
    vector = rng.standard_normal(dim)
    norm = np.linalg.norm(vector)
    vector = vector/norm
    return vector

def init_globals(points_in, ranking_in, mapping_in, kval):
    global points, ranking, mapping, K, DIM, NUM_DATA

    points = points_in
    ranking = ranking_in
    mapping = mapping_in
    K = kval
    DIM = len(points_in[0])
    NUM_DATA = len(points)
    
def init_globals_group(points_in, ranking_in, mapping_in, kval, groups):
    global points, ranking, mapping, K, DIM, NUM_DATA, NUM_GROUPS

    points = points_in
    ranking = ranking_in
    mapping = mapping_in
    K = kval
    DIM = len(points_in[0])
    NUM_DATA = len(points)
    NUM_GROUPS = groups