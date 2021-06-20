""" 
Causal Confusion CEM Planner.
Test if causal curiosity can help distinguish between two causal factors
not introduced in training, but in testing.
"""
import random
import time
import os
from datetime import datetime
from pathlib import Path
from functools import partial
import gym

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from loguru import logger
from tqdm import tqdm
import dill

from gym.wrappers.monitoring.video_recorder import VideoRecorder

import multiprocessing_on_dill as mp
import psutil
# local imports
from cem_planner import CEMPlanner
from plan_action_spaces import get_plan_action_space
from cem.uniform_bounds import UniformBounds
from causal_world.task_generators.task import task_generator
from causal_world.envs.causalworld import CausalWorld


seeds = [122, 233, 344, 455, 566, 677, 788, 889, 900, 1111]
masses = [0.1, 0.15, 0.18, 0.19, 0.2, 0.21, 0.22, 0.25, 0.3] # median 0.4, count 9
sizes = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09] # median 0.75, count 9
shapes = ['cube']
action_mode = 'RGB_asym'
obs_dim = 3

class CEMPlannerCW(CEMPlanner):
    """
    Extension of CEMPlanner class.
    New plan function that allows us to specify which env to train in.
    Returns: best-trained action plan, observations, trained KMeans model
    """

    # plan envs 
    def getEnvs(self, masses, sizes, shapes, bmass = None, bsize = None, bshape = None):
        
        envs = []

        for mass in masses:
            for size in sizes:
                for shape in shapes:
                    task = task_generator(task_generator_id ='lifting',
                        tool_block_mass = mass,
                        tool_block_shape = shape,
                        tool_block_size = size)
                    env = CausalWorld(task=task, 
                        skip_frame = self.frame_skip,
                        enable_visualization = False)
                    envs.append(env)
        if bmass != None:
            task = task_generator(task_generator_id ='lifting',
                        tool_block_mass = bmass,
                        tool_block_shape = bshape,
                        tool_block_size = bsize)

            env = CausalWorld(task=task, 
                        skip_frame = self.frame_skip, 
                        enable_visualization = False)
            envs.append(env)

        return envs

    # plan envs with test env
   


# for multiprocessing
def worker(i, q, training_action_plan, training_rel_duration_plan, training_observations, training_km_sdtw, training_planner):
    p = psutil.Process()
    # Params. Play around with these settings. They are not yet optimized.
    #scenario = 'lift'
    p.cpu_affinity([i])


    total_budget = 400
    plan_horizon = 6
    n_plan_iterations = 20
    frame_skip = 1
    sampler = 'uniform'
    warm_starts = False
    warm_start_relaxation = 0.0
    elite_fraction = 0.1
    viz_progress = True
    n_frames = 198

    plan_action_repeat = np.floor_divide(n_frames, plan_horizon)
    n_plan_cache_k = plan_horizon
    n_plans = np.floor_divide(total_budget * n_plan_cache_k,
                              plan_horizon * n_plan_iterations)
    n_plan_elite = max(1, int(round(elite_fraction * n_plans)))

    if n_plans <= n_plan_elite:
        n_plan_elite = n_plans - 1
    # action_space is gym `Box` env that defines vals for each action [-1,1]
    # action_transf is a func that returns an array for real-val actions? 
    action_space, action_transformation = get_plan_action_space(action_mode)    
   

    training_predict= training_km_sdtw.predict(training_observations)
    print("training prediction: ", training_predict)



    for i_mass, mass in enumerate(masses):
        for i_size, size in enumerate(sizes):
            for i_shape, shape in enumerate(shapes):
           
            
                envs = training_planner.getEnvs([mass],[size],[shape])
                test_observations = training_planner.simulate(envs[0], training_action_plan, training_rel_duration_plan)
                test_predict = training_km_sdtw.predict(test_observations)
                belief_size_cluster = [sizes[i] for i, label in enumerate(training_predict) if label == test_predict[0]]
            
                for i_seed, seed in enumerate(seeds):
  
                ## start from the last saved value
                    with open('predictions.pickle','rb') as f:
                        predictions = dill.load(f)
                    with open('rewards.pickle','rb') as f:
                        rewards = dill.load(f)
                    if rewards[i_mass][i_size][i_seed] != -1:
                        if i_mass == 4:
                            print(f"size {size}, belief_planner {belief_size_cluster}\n")
                        continue
                    else:
                        q.put((i_mass, i_size, i_seed, -9, -9))
                        print(f"worker {i} has started processing ({i_mass},{i_size},{i_seed})")


                    rng = np.random.RandomState(seed)
                    np.random.seed(seed)

              
                    belief_planner = CEMPlannerCW(n_plans=n_plans,
                               horizon=plan_horizon,
                               action_space=action_space,
                               sampler=sampler,
                               n_iterations=n_plan_iterations,
                               n_elite=n_plan_elite,
                               cache_k=n_plan_cache_k,
                               obs_dim=obs_dim,
                               warm_starts=warm_starts,
                               warm_start_relaxation=warm_start_relaxation,
                               plan_action_repeat=plan_action_repeat,
                               action_transformation=action_transformation,
                               rng=rng,
                               viz_progress=viz_progress,
                               )


                    envs = belief_planner.getEnvs([0.2],belief_size_cluster,shapes,mass,size,'cube')
                    belief_action_plan, belief_duration_plan, belief_observations, belief_km_sdtw, belief_best_return = belief_planner.plan(envs)
                
                    labels = belief_km_sdtw.predict(belief_observations)
                    print(labels)
                    reward = belief_best_return
                    score = 1 if (sum(labels) == 1 and labels[-1] == 1) or (sum(labels) == len(labels)-1 and labels[-1] == 0) else 0 
                
                    res = (i_mass, i_size, i_seed, reward, score)
                    q.put(res)
                    time.sleep(1)

# for multiprocessing
def writer(q):
    
    while True:
        m = q.get()
        print(m)

        with open('rewards.pickle','rb') as f:
            rewards = dill.load(f)

        with open('predictions.pickle','rb') as f:
            predictions = dill.load(f)

        if m == 'kill':
           break
        rewards[m[0]][m[1]][m[2]] = m[3]
        predictions[m[0]][m[1]][m[2]] = m[4]

        dill.dump(rewards, open('rewards.pickle','wb') )
        dill.dump(predictions, open('predictions.pickle','wb'))

        print(f"writer modified ({m[0]},{m[1]},{m[2]})")


def main(output_dir):

    retrain_training_planner = False 
    reset_output = False
    clear_processing = False

    # change into output directory
    if not output_dir.exists():
        os.mkdir(output_dir)
    os.chdir(output_dir)
    # Params. Play around with these settings. They are not yet optimized.
    #scenario = 'lift'
    
    total_budget = 400
    plan_horizon = 6
    n_plan_iterations = 20
    frame_skip = 1
    sampler = 'uniform'
    warm_starts = False
    warm_start_relaxation = 0.0
    elite_fraction = 0.1
    viz_progress = True
    n_frames = 198

    seed = 123
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    plan_action_repeat = np.floor_divide(n_frames, plan_horizon)
    n_plan_cache_k = plan_horizon
    n_plans = np.floor_divide(total_budget * n_plan_cache_k,
                              plan_horizon * n_plan_iterations)
    n_plan_elite = max(1, int(round(elite_fraction * n_plans)))

    if n_plans <= n_plan_elite:
        n_plan_elite = n_plans - 1

    # action_space is gym `Box` env that defines vals for each action [-1,1]
    # action_transf is a func that returns an array for real-val actions? 
    action_space, action_transformation = get_plan_action_space(action_mode) 


    # load pre-calculated rewards and predictions
    if reset_output:
        predictions, rewards = np.empty((len(masses),len(sizes),10)), np.empty((len(masses),len(sizes),10))
        predictions.fill(1)
        rewards.fill(1)

        dill.dump(predictions, open('predictions.pickle', 'wb'))
        dill.dump(rewards, open('rewards.pickle', 'wb'))
    else:
        with open('predictions.pickle','rb') as f:
            predictions = dill.load(f)
        with open('rewards.pickle','rb') as f:
            rewards = dill.load(f)


    if clear_processing:
        for i in range(len(rewards)):
            for j in range(len(rewards[0])):
                for k in range(len(rewards[0][0])):
                    if rewards[i][j][k] == -9:
                        rewards[i][j][k] = -1
                        predictions[i][j][k] = -1
        dill.dump(predictions, open('predictions.pickle', 'wb'))
        dill.dump(rewards, open('rewards.pickle', 'wb'))

    print(rewards)
    print(predictions)


    # load pre-trained training planner or retrain
    if retrain_training_planner:
        
        training_planner = CEMPlannerCW(n_plans=n_plans,
                               horizon=plan_horizon,
                               action_space=action_space,
                               sampler=sampler,
                               n_iterations=n_plan_iterations,
                               n_elite=n_plan_elite,
                               cache_k=n_plan_cache_k,
                               obs_dim = obs_dim,
                               warm_starts=warm_starts,
                               warm_start_relaxation=warm_start_relaxation,
                               plan_action_repeat=plan_action_repeat,
                               action_transformation=action_transformation,
                               rng=rng,
                               viz_progress=viz_progress,
                               )

        envs = training_planner.getEnvs([0.2], sizes, ['cube'])
        training_action_plan, training_rel_duration_plan, training_observations, training_km_sdtw, training_best_return = training_planner.plan(envs)
        
        dill.dump(training_action_plan, open('training_action_plan.pickle', 'wb'))
        dill.dump(training_rel_duration_plan, open('training_rel_duration_plan.pickle', 'wb'))
        dill.dump(training_observations, open('training_observations.pickle', 'wb'))
        dill.dump(training_km_sdtw, open('training_km_model.pickle', 'wb'))
        dill.dump(training_best_return, open('training_best_return.pickle', 'wb'))
        dill.dump(training_planner, open('training_planner.pickle', 'wb'))
        
   
    with open('training_action_plan.pickle','rb') as f:
        training_action_plan = dill.load(f)
    with open('training_rel_duration_plan.pickle','rb') as f:
        training_rel_duration_plan = dill.load(f)
    with open('training_observations.pickle','rb') as f:
        training_observations = dill.load(f)
    with open('training_km_model.pickle','rb') as f:
        training_km_sdtw = dill.load(f)
    with open('training_best_return.pickle','rb') as f:
        training_best_return = dill.load(f)
    with open('training_planner.pickle','rb') as f:
        training_planner = dill.load(f)

    manager = mp.Manager()
    q = manager.Queue()
    print("cpu count: ", mp.cpu_count())
    pool = mp.Pool(1+2)

    watcher = pool.apply_async(writer, (q,))

    jobs = []
    for i in range(1):
        job = pool.apply_async(worker, (i, q, training_action_plan, training_rel_duration_plan, training_observations, training_km_sdtw, training_planner))
        jobs.append(job)
        time.sleep(1)

    for job in jobs:
        job.get()

    q.put('kill')
    pool.close()
    pool.join()

    print("rewards: ",rewards)
    print("predictions: ", predictions)



if __name__ == '__main__':
    print("CEM Planner CausalWorld")

    ## change this line to store your data in whatever folder you'd like
    output_dir = Path('./pickle/CW')
    print(f"Output dir: {output_dir}")
    print("Make sure this directory is what you expected, if not change it!")
    main(output_dir=output_dir)


