import sys

#These values affect the additive benefit given
#This is for 1 group instances
gender_benefit = 4
#This is for 2 group instances. 
cat_benefit = 6

#Weight vector to use
WEIGHT = [1, 1, 1]

#This decides the number of candidates for the input ranking
#Default value, but overriden if command line argument is specified
LIMIT = 75000

if len(sys.argv) > 1:
    LIMIT = int(sys.argv[1])

#Set category to true for 2 group instances, else false gives 1 group instances
CATEGORY = True

with open("jee.in") as f:
    
    lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    for i in range(len(lines)):
        for j in range(3, len(lines[i])):
            lines[i][j] = int(lines[i][j])
        if len(lines[i]) >= 6:
            lines[i].append(lines[i][3] + lines[i][4] + lines[i][5])
        else:
            lines[i].append(-1e6)
        if "M" in lines[i][2]:
            lines[i][2] = 0
        else:
            lines[i][2] = 1
        if "GE" in lines[i][0]:
            lines[i][0] = 0
        else:
            lines[i][0] = 1
        
        
    lines.sort(key = lambda tup : tup[-1], reverse = True)
    
    rankings = []
    privset = set()
    added = 0
    added_2group = [0, 0]

    for idx in range(LIMIT):
        score = 0
        score += lines[idx][3] * WEIGHT[0]
        score += lines[idx][4] * WEIGHT[1]
        score += lines[idx][5] * WEIGHT[2]

        if CATEGORY == True:
            #Non-GE and Female gender get both boost
            if lines[idx][0] == 1:
                if lines[idx][2] == 1:
                    score += cat_benefit + gender_benefit
                    added_2group[0] += 1
                #GE and Female gender gets only gender boost
                else:
                    score += gender_benefit
                    added_2group[1] += 1
        else:
            #Only Female gender gets boosted
            if lines[idx][2] == 1:
                score += gender_benefit
                added += 1
        person = (score, idx, lines[idx][2])
        rankings.append(person)
    rankings.sort(key=lambda x: x[0], reverse=True)

    if CATEGORY == False:
        print(len(rankings), added)
    else:
        print(len(rankings), added_2group[0], added_2group[1])
    for l in lines[:LIMIT]:
        if CATEGORY == False:
            print(*l[2:6])
        else:
            print(l[0], *l[2:6])
    rank = 1
    ties = 1
    print(rankings[0][1], rank, 1 if 0 in privset else 0)
    for i in range(1, len(rankings)):
        if rankings[i-1][0] - rankings[i][0] <= 1e-2:
            ties += 1
        else:
            rank += ties
            ties = 1
        print(rankings[i][1], rank, 1 if i in privset else 0)

    addlist = [i for i in range(len(lines)) if i not in privset]
