import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
import numpy.typing as npt
import multiprocessing
import os
from functools import partial
import ermb_functions
import dominance_check
import signal

EPSILON = 1e-10

# --- Global variables ---
TLIMIT = 1800
startTime = 0
NUM_DATA = 0
K = 0
DIM = 0
points = None
ranking = None
planes = None
env = None
mapping = None

def init_globals(planes_in, points_in, ranking_in, mapping_in, k_in, stime):
    """
    Initializer function for each worker process.
    This runs once per worker, setting up its global state.
    
    NOTE: This function sets the globals *inside* the ermb_functions
    module for each worker process.
    """
    
    print(f"Initializing worker {os.getpid()}...")
    
    # Call the initializer inside the compiled module
    ermb_functions.init_globals_kdim(planes_in, points_in, ranking_in, mapping_in, k_in)

    global K, DIM, env, startTime
    DIM = len(points_in[0])
    K = k_in
    startTime = stime
    
    # Each worker MUST have its own Gurobi environment
    env = gp.Env()
    
    # Also update globals in the compiled module
    ermb_functions.TLIMIT = TLIMIT
            
def solve_kd(queue, lock, maxsol, bestweights_shared):
    # Access the global variables set by init_worker
    # We only need 'env' and 'startTime' locally.
    # The 'ermb_functions' module has its own globals.
    global env, startTime
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:

        try:
            # Get a task from the queue
            tmp = queue.get()
            
            # Poison pill: A 'None' task means "shut down"
            if tmp is None:
                queue.task_done()
                break
            #If time limit is reached, stop
            if time.time() - startTime > TLIMIT:
                queue.task_done()
                continue
            
            weight, signvecs = tmp
            
            # Access planes from the compiled module's globals
            worker_planes = ermb_functions.planes
            
            if len(signvecs) != len(worker_planes):
                # We are branching (internal node)
                
                index = len(signvecs)
                p = worker_planes[index]
                sign = np.dot(weight, p)
                m = gp.Model(env=env)
                m.params.OutputFlag = 0
                m.params.Method = 0

                delta = m.addVar(ub = 1)
                m.setObjective(delta, GRB.MAXIMIZE)
                pointvar = m.addMVar(shape = DIM)
                for i in range(index):
                    if signvecs[i] > 0:
                        m.addConstr(worker_planes[i] @ pointvar >= delta)
                    else:
                        m.addConstr(worker_planes[i] @ pointvar <= -delta)
                
                if sign > 0:
                    # Branch 1: Sign is already positive
                    queue.put((weight, signvecs + [1]))

                    m.addConstr(worker_planes[index] @ pointvar <= -delta)
                    m.optimize()
                    
                    if delta.X > EPSILON:
                        newweight = pointvar.X
                        
                        result = ermb_functions.compute_objective(newweight)
                
                        # --- Critical Section ---
                        with lock:
                            if result < maxsol.value:
                                maxsol.value = result
                                for i in range(len(newweight)):
                                    bestweights_shared[i] = newweight[i]
                                print(f"[Worker {os.getpid()}] New best sol: {maxsol.value}. Time: {round(time.time()-startTime, 2)}")
                        # --- End Critical Section ---
                        queue.put((newweight, signvecs + [-1]))
                
                else:
                    # Branch 2: Sign is already negative
                    queue.put((weight, signvecs + [-1]))
                                    
                    m.addConstr(worker_planes[index] @ pointvar >= delta)
                    m.optimize()
                    
                    if delta.X > EPSILON:
                        newweight = pointvar.X
                        
                        # Call the compiled function
                        result = ermb_functions.compute_objective(newweight)
                        
                        # --- Critical Section ---
                        with lock:
                            if result < maxsol.value:
                                maxsol.value = result
                                for i in range(len(newweight)):
                                    bestweights_shared[i] = newweight[i]
                                print(f"[Worker {os.getpid()}] New best sol: {maxsol.value}. Time: {round(time.time()-startTime, 2)}")
                        # --- End Critical Section ---
                        
                        queue.put((newweight, signvecs + [1]))
            queue.task_done()
        except Exception as e:
            print(f"Worker {os.getpid()} encountered an error: {e}")

def search_hyperplanes_parallel(planes_in: list[list[float]],
                                points_in: npt.NDArray,
                                mapping_in: dict[int, int],
                                k_in: int) -> int:
    
    # Determine number of workers
    num_workers = 16
    print(f"Starting search with {num_workers} workers...")
    
    # Use a Manager to create shared objects
    manager = multiprocessing.Manager()
    queue = manager.JoinableQueue()
    lock = manager.Lock()
    
    # Shared state for the best solution
    maxsol = manager.Value('i', 1_000_000_000)
    bestweights = manager.Array('d', [0.0] * DIM)

    # Create the pool of worker processes
    init_args = (planes_in, points_in, ranking,  mapping_in, k_in, startTime)
    pool = multiprocessing.Pool(num_workers, initializer=init_globals, initargs=init_args)
    
    global rng
    rng = np.random.default_rng([K, K, DIM, NUM_DATA]) # Init main rng
    vector = rng.standard_normal(DIM)
    norm = np.linalg.norm(vector)
    vector = vector/norm
    queue.put((vector, []))
    
    # Start the worker processes. They will block waiting for tasks on the queue.
    # We pass the shared objects to them.
    try:
        for _ in range(num_workers):
            pool.apply_async(solve_kd, (queue, lock, maxsol, bestweights))

        # Wait for all tasks to be processed
        # queue.join() blocks until queue.task_done() has been called
        # for every item that was ever .put() on the queue.
        print("Main process: Waiting for all tasks to complete...")
        queue.join()
        print("Main process: All tasks complete.")

        # Stop the workers by sending None
        for _ in range(num_workers):
            queue.put(None)

        # Clean up the pool
        pool.close()
        pool.join()
    except Exception as e:
        print(f"Error exception received: {e}")
        print("Likely child process received SIGKILL")
        print("Returning error value")
        pool.close()
        pool.join()
        return 1e9

    sol = []
    for w in bestweights:
        sol.append(w)
    print("Time taken (s):", time.time() - startTime)
    print(f"Best objective found: {maxsol.value}")
    print(f"Weight vectors giving the solution: {sol}")
    
    return maxsol.value


if __name__ == '__main__':
    print("----- ERMB Algorithm -----")

    parameters = input().rstrip().split(" ")
    parameters = [int(i) for i in parameters]
    NUM_DATA = parameters[0]
    K = parameters[1]

    points = []
    print("Input data has", NUM_DATA, "points. K =", K)
    for i in range(NUM_DATA):
        point = input().rstrip().split(" ")
        point = [float(i) for i in point]
        points.append(point)

    DIM = len(points[0])

    print("Input parameters:")
    print("Number of points:", NUM_DATA)
    print("Dim =", DIM)
    print("K =", K)
    
    #Read the rankings
    #ranking[i][0] is the id
    #ranking[i][1] is the rank position (because of ties, this is not necessarily consecutive)
    #ranking[i][2] is the indicator for belonging to privileged group
    ranking = []
    for i in range(NUM_DATA):
        rankpos = input().rstrip().split(" ")
        rankpos = [int(i) for i in rankpos]
        ranking.append(rankpos)

    startTime = time.time()

    dominated_count = 0
    with multiprocessing.Pool(processes=16, initializer = dominance_check.init_globals, initargs = (points, ranking, DIM)) as pool:
        initset: list[int] = [i for i in range(NUM_DATA)]
        sols = pool.map(dominance_check.check_dominated_full, initset)
        initset = np.array(initset)
        remaining_points = initset[~(np.array(sols))]
        for i in range(len(sols)):
            if sols[i] == True:
                dominated_count += 1

    # Convert points to NumPy array for efficient use in workers
    points = np.array(points)
    planes = []
    with multiprocessing.Pool(processes=16, initializer = ermb_functions.init_globals_kdim, initargs = ([], points, [], {}, K)) as pool:
        partial_generateplanes = partial(ermb_functions.generateplanes_kd_sample, sample = remaining_points)
        nestedplanes = pool.map(partial_generateplanes, remaining_points.tolist())
    
    for p in nestedplanes:
        planes += p

    #Finally, construct the renaming of elements to 0, 1, 2, ..., n-1 in their order in input
    mapping = {}
    for i in range(len(points)):
        mapping[ranking[i][0]] = i
    
    print("Plane count:", len(planes))
    
    planes = np.array(planes)
    # Call the new parallel function
    sol = search_hyperplanes_parallel(planes, points, mapping, K)
    timetaken = time.time() - startTime
    print(f"Final line, {NUM_DATA}, {DIM}, Singleton, {K}, {dominated_count} , ERMB, {round(timetaken,2)}, {sol}")
    print("------------ End of Algorithm run -----------")