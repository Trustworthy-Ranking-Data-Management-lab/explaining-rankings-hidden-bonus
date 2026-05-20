import numpy as np
import sys

EPSILON = 1e-5

#Default parameters if command line arguments not given
NUM_DATA = 10000
DIM = 2
KRATIO = 0.1

if len(sys.argv) > 1:
    NUM_DATA = int(sys.argv[1])
    DIM = int(sys.argv[2])
    KRATIO = float(sys.argv[3])

PRIVILEGED = int(NUM_DATA * KRATIO)

rng = np.random.default_rng(seed = [NUM_DATA, DIM, PRIVILEGED, PRIVILEGED])

ADD_RANGE = (5 * DIM, 10 * DIM)
DATA_UPPER_BOUND = 25

#Write ground truth to a log file for debug
truthfile = r"sol_singleton_" + str(NUM_DATA) + "_" + str(DIM) + "_" + str(KRATIO) + ".txt"
with open(truthfile, "w") as f:
    weights = rng.uniform(0.0, DATA_UPPER_BOUND, DIM).tolist()
    outstring = ""
    for i in range(len(weights)):
        weights[i] = round(weights[i], 2)
    print(*weights, file = f)

    id_to_group = {}

points = []
for i in range(NUM_DATA):
    data = rng.uniform(0, DATA_UPPER_BOUND, DIM)
    for j in range(len(data)):
        data[j] = round(data[j], 2)
    points.append(data)
#decide which elements belong to privileged group

idlist = [i for i in range(NUM_DATA)]
pgroup = rng.choice(idlist, PRIVILEGED, False)
for i in range(NUM_DATA):
    id_to_group[i] = 0

for j in pgroup:
    id_to_group[j] = 1
addvaluearr = rng.uniform(ADD_RANGE[0], ADD_RANGE[1], PRIVILEGED).tolist()
addvaluearr = [round(i, 2) for i in addvaluearr]

true_additive = 0

outfile_name = r"singleton_" + str(NUM_DATA) + "_" + str(DIM) + "_" + str(KRATIO) + ".txt"

with open(outfile_name, "w") as f:
    ones = PRIVILEGED
    print(NUM_DATA, PRIVILEGED, file = f)

    #while True:
    rankings = []
    idx = 0
    for i in range(NUM_DATA):
        data = points[i]
        score = 0
        for j in range(len(data)):
            score += data[j] * weights[j]
        if id_to_group[i] > 0:
            score += addvaluearr[idx]
            idx += 1
        rankings.append((score, i, id_to_group[i]))

    rankings = sorted(rankings, key = lambda tup : tup[0], reverse=True)
    
    for point in points:
        print(*point, file = f)

    rank = 1
    ties = 1
    #first element in score is always rank 1
    print(rankings[0][1], rank, file = f)

    for i in range(1, len(rankings)):
        if rankings[i-1][0] - rankings[i][0] > EPSILON:
            rank += ties
            ties = 1
        else:
            print("tie detected")
            ties += 1
        print(rankings[i][1], rank, file = f)

    #print(privileged, file=sys.stderr)

