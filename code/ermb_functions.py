import time
import numpy as np
import numpy.typing as npt

EPSILON: float = 1e-10
planes: npt.NDArray[np.float64]
ranking = []
points: npt.NDArray[np.float64]
mapping: dict[int, int] = {}
DIM = 0
K = 0

# Generates hyperplanes in 2D
# In 2D, hyperplanes are easy to represent since keeping x-coordinate is enough to set y coordinate as 1-x.
# i - index of first point in pair
def generateplanes_2d(i: int) -> list[float]:
    rawplanes: list[float] = []
    for j in range(i+1, len(points)):
        plane: list[float] = [points[i][k] - points[j][k] for k in range(DIM)]
        intersection = 0.0
        if plane[1] < 0 and plane[0] > 0 or plane[0] < 0 and plane[1] > 0:
            #compute intersection with y = 1-x
            intersection = float((-plane[1]) / (plane[0] - plane[1]))
            rawplanes.append(intersection)
    return rawplanes

# Generates hyperplanes in 2D
# Only generates planes between pairs of points in the array sample
# i - index of first point in pair (with respect to sample)
# sample - array of indices of points for hyperplane generation
def generateplanes_2d_sample(i: int, sample: npt.NDArray[np.int32]) -> list[float]:
    rawplanes: list[float] = []
    for j in range(i+1, len(sample)):
        plane: list[float] = [points[sample[i]][k] - points[sample[j]][k] for k in range(DIM)]
        if plane[1] == 0:
            plane[1] = 1e-9
        #manipulate it into (a, -b) form
        #if we have (a, b) where both >0, then impossible to flip their order with only positive weights
        intersection = 0.0
        if plane[1] < 0 and plane[0] > 0 or plane[0] < 0 and plane[1] > 0:

            #compute intersection with y = 1-x
            intersection = float((-plane[1]) / (plane[0] - plane[1]))
            rawplanes.append(intersection)
    return rawplanes

# Given hyperplane, computes the ranking realized by a weight vector taken in the region
# index - index of hyperplane
# tlimit - time limit
# stime - start time to check for timeout
# Output: Tuple (number of tuples boosted, weight vector) where weight vector is a list
def solve_2d(index: int, tlimit: float, stime: float) -> tuple[float, list[float]]:
    #slightly hack-ish solution to timeout
    if time.time() - stime > tlimit:
        return (1e9, [-1])
    xcoord: float = (planes[index] + planes[index+1]) / 2
    weights: list[float] = [xcoord, 1 - xcoord]
        
    values = points @ np.array(weights)
    values = values.tolist()
    for i in range(len(values)):
        values[i] = [values[i], i]
    values.sort(key = lambda x: (x[0], x[1]), reverse = True)
    renamed_idxs = []
    for i in range(len(points)):
        renamed_idxs.append(mapping[values[i][1]])

    distance: float = float(len(points) - do_lis(renamed_idxs))
    return (distance, weights)

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

#K-dimensional related functions

def generateplanes_kd(i: int) -> list[list[float]]:
    planes = []
    for j in range(i+1, len(points)):
        planes.append((points[i] - points[j]).tolist())
    return planes

def generateplanes_kd_sample(i: int, sample: npt.NDArray[np.int64]) -> list[list[float]]:
    planes: list[list[float]] = []
    for j in range(i+1, len(sample)):
        plane = (points[sample[i]] - points[sample[j]]).tolist()
        planes.append(plane)
    return planes

def compute_objective(weights: npt.NDArray[np.float64]) -> int:
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

def init_globals(pls, pts, rank, m, kval):
    global planes, points, ranking, K, DIM, mapping
    planes = pls
    points = pts
    ranking = rank
    K = kval
    mapping = m
    DIM = len(pts[0])

def init_globals_kdim(pls, pts, rank, mapping_in, kval):
    global planes, points, ranking, K, DIM, mapping
    planes = pls
    points = pts
    ranking = rank
    mapping = mapping_in
    K = kval
    DIM = len(pts[0])
