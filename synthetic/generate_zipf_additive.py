import numpy as np
import sys

EPSILON = 1e-5

#Default parameters if command line arguments not given
NUM_DATA = 50000
NUM_GROUPS = 1
DIM = 2
PRIVILEGED = [5000]

if len(sys.argv) > 1:
    NUM_DATA = int(sys.argv[1])
    DIM = int(sys.argv[2])
    KRATIO = float(sys.argv[3])
    NUM_GROUPS = int(sys.argv[4])

    PRIVILEGED = int(KRATIO * NUM_DATA)

ADD_RANGE = (5*DIM, 10*DIM)
DATA_UPPER_BOUND = 25

rng = np.random.default_rng(seed = [NUM_DATA, DIM, PRIVILEGED, NUM_GROUPS])

#assumes num_groups divides privileged exactly!
if NUM_GROUPS > 1:
    PRIVILEGED = [PRIVILEGED // NUM_GROUPS] * NUM_GROUPS
else:
    PRIVILEGED = [PRIVILEGED]

#Write ground truth to a log file for debugging
truthfile = r"zipf_sol_" + str(NUM_GROUPS) + "group_" + str(NUM_DATA) + "_" + str(DIM) + "_" + str(KRATIO) + ".txt"

with open(truthfile, "w") as f:
    weights = rng.zipf(2, DIM).tolist()
    for i in range(len(weights)):
        weights[i] = round(weights[i], 2)
    print(*weights, file = f)

    id_to_group = {}

    ADDITIVE_VAL = rng.uniform(ADD_RANGE[0], ADD_RANGE[1], NUM_GROUPS).tolist()
    ADDITIVE_VAL = [round(ADDITIVE_VAL[i], 2) for i in range(NUM_GROUPS)]

    print(*ADDITIVE_VAL, file = f)

points = []
for i in range(NUM_DATA):
    data = rng.zipf(2, DIM).tolist()
    data = [round(i, 2) for i in data]
    points.append(data)
#decide which elements belong to privileged group

idlist = [i for i in range(NUM_DATA)]
pgroup = rng.choice(idlist, sum(PRIVILEGED), False)
for i in range(NUM_DATA):
    id_to_group[i] = 0

#simply partition them into each group
#should not matter since points are generated independently at random
counter = 0
for pval in range(len(PRIVILEGED)):
    for j in range(PRIVILEGED[pval]):
        id_to_group[pgroup[counter]] = pval + 1
        counter += 1

outfile_name = r"zipf_" + str(NUM_GROUPS) + "group_" + str(NUM_DATA) + "_" + str(DIM) + "_" + str(KRATIO) + ".txt"

with open(outfile_name, "w") as f:
    print(NUM_DATA, *PRIVILEGED, file = f)
    
    #while True:
    rankings = []
    for i in range(NUM_DATA):
        data = points[i]
        score = 0
        for j in range(len(data)):
            score += data[j] * weights[j]
        if id_to_group[i] > 0:
            score += ADDITIVE_VAL[id_to_group[i]-1]
        rankings.append((score, i, id_to_group[i]))
    rankings = sorted(rankings, key = lambda tup : tup[0], reverse=True)
    
    for point in points:
        print(*point, file = f)
    
    
    rank = 1
    ties = 1
    #first element in score is always rank 1
    print(rankings[0][1], rank, file = f)

    for i in range(1, len(rankings)):
        if rankings[i-1][0] - rankings[i][0] <= EPSILON:
            ties += 1
        else:
            rank += ties
            ties = 1
        print(rankings[i][1], rank, file = f)

    print(rankings[0][0], file = sys.stderr)
    print(rankings[len(rankings)-1][0], file = sys.stderr)