def check_dominated(i: int, pos: list[int]) -> bool:
    first_pt = points[ranking[i][0]]
    for j in range(i+1, len(pos)):
        jrank: int = pos[j]
        dominated = True
        tied = True
        second_pt = points[ranking[jrank][0]]
        for k in range(DIM):
            if first_pt[k] > second_pt[k]:
                dominated = False
                tied = False
                break
            elif first_pt[k] < second_pt[k]:
                tied = False
        if dominated == True and tied == False:
            return True
        #if tied in all dimensions, but first point is ranked strictly before second point, it must have additive bonus
        elif tied == True and ranking[i][1] < ranking[jrank][1]:
            return True
    #if all checks don't return true, then it is not dominated
    return False

#This checks dominance across all pairs of points
#For the initial level of dominance check, to reduce passing arguments
def check_dominated_full(i: int) -> bool:
    first_pt = points[ranking[i][0]]
    for j in range(i+1, len(points)):
        dominated = True
        tied = True
        second_pt = points[ranking[j][0]]
        for k in range(DIM):
            if first_pt[k] > second_pt[k]:
                dominated = False
                tied = False
                break
            elif first_pt[k] < second_pt[k]:
                tied = False
        if dominated == True and tied == False:
            return True
        #if tied in all dimensions, but first point is ranked strictly before second point, it must have additive bonus
        elif tied == True and ranking[i][1] < ranking[j][1]:
            return True
    #if all checks don't return true, then it is not dominated
    return False

def init_globals(pts, rkg, d):
    global points
    global ranking
    global DIM
    points = pts
    ranking = rkg
    DIM = d

points: list[list[float]] = [] 
ranking: list[list[int]] = []
DIM: int = 0