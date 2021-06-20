import random
import time
from datetime import datetime
from pathlib import Path
# from causal_world.task_generators.task import task_generator
# from causal_world.envs.causalworld import CausalWorld

import moviepy.editor as mpy
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from loguru import logger
from tqdm import tqdm

from cc_planner_mujoco import CEMPlanner
# from envs.block_manipulation_env import BlockManipulationEnv, SCENARIOS
from frameskip_wrapper import FrameSkip
from plan_action_spaces import get_plan_action_space
import gym
from gym.wrappers import Monitor

def main(output_dir, viz_progress=False):
    scenario = 'lift'

    if scenario == 'spin':
        n_frames_per_episode = 50
    else:
        n_frames_per_episode = 50

    # Play around with these settings. They are not yet optimized
    total_budget = 10
    plan_horizon = 10
    n_plan_iterations = 1
    frame_skip = 1
    action_mode = 'hopper'#'ant'#humanoid
    sampler = 'uniform'
    warm_starts = False
    warm_start_relaxation = 0.0
    elite_fraction = 0.1

    plan_action_repeat = n_frames_per_episode // plan_horizon#5
    n_plan_cache_k = plan_horizon#10
    n_plans = total_budget * n_plan_cache_k // (plan_horizon * n_plan_iterations)

    n_plan_elite = max(1, int(round(elite_fraction * n_plans)))

    if n_plans <= n_plan_elite:
        n_plan_elite = n_plans - 1

    # bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
    #masses = np.arange(0.005, 0.1, 0.0005).tolist()
    #envs = []
    #size =  0.0325
    #for mass in masses:
    #    env = FrameSkip(BlockManipulationEnv(bullet_client=bullet_client,
    #                                     scenario=scenario,
    #                                     control_rate_s=0.004,
    #                                             mass=mass, 
    #                                             size=size,
    #                                             ),
    #                frame_skip=frame_skip)
    #    envs.append(env)

    #env = FrameSkip(BlockManipulationEnv(bullet_client=bullet_client,
    #                                     scenario=scenario,
    #                                     control_rate_s=0.004),
    #                frame_skip=frame_skip)
    seed = 1235

    logger.info(f'seed: {seed}')
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)

    #obs = env.reset()
    episode_actions = []
    

    action_space, action_transformation = get_plan_action_space(action_mode)

    planner = CEMPlanner(n_plans=n_plans,
                         horizon=plan_horizon,
                         action_space=action_space,
                         sampler=sampler,
                         n_iterations=n_plan_iterations,
                         n_elite=n_plan_elite,
                         cache_k=n_plan_cache_k,
                         warm_starts=warm_starts,
                         warm_start_relaxation=warm_start_relaxation,
                         plan_action_repeat=plan_action_repeat,
                         action_transformation=action_transformation,
                         rng=rng,
                         viz_progress=viz_progress,
                         action_mode=action_mode,)

    real_rewards = []

    frames = []

    #save_state = env.bullet_client.saveState
    #restore_state = env.bullet_client.restoreState
    
    actionSeq, plan_returns = planner.plan(None, None, None)
    # print('actionSeq',len(actionSeq))
    # print(actionSeq)
    print('plan_returns', plan_returns)

    if action_mode == "cheetah":
        e1 = gym.make('HalfCheetah-v3')
        a = 0.5*e1.model.body_mass
        e1.model.body_mass[:]=a
        env1 = Monitor(e1, './video', force=True)

        e2 = gym.make('HalfCheetah-v3')
        a = 1.4*e2.model.body_mass
        e2.model.body_mass[:]=a
        env2 = Monitor(e2, './video', force=True)


    elif action_mode =="humanoid":
        e1 = gym.make('Humanoid-v3')
        a = 0.5*e1.model.body_mass
        e1.model.body_mass[:]=a
        env1 = Monitor(e1, './video', force=True)

        e2 = gym.make('Humanoid-v3')
        a = 1.4*e2.model.body_mass
        e2.model.body_mass[:]=a
        env2 = Monitor(e2, './video', force=True)
        # env = Monitor(gym.make('Humanoid-v3'), './video', force=True)
    elif action_mode =="ant":
        e1 = gym.make('Ant-v3')
        a = 0.5*e1.model.body_mass
        e1.model.body_mass[:]=a
        env1 = Monitor(e1, './video', force=True)

        e2 = gym.make('Ant-v3')
        a = 1.4*e2.model.body_mass
        e2.model.body_mass[:]=a
        env2 = Monitor(e2, './video', force=True)
        # env = Monitor(gym.make('Ant-v3'), './video', force=True)
    elif action_mode =="walker":
        e1 = gym.make('Walker2d-v3')
        a = 0.5*e1.model.body_mass
        e1.model.body_mass[:]=a
        env1 = Monitor(e1, './video', force=True)

        e2 = gym.make('Walker2d-v3')
        a = 1.4*e2.model.body_mass
        e2.model.body_mass[:]=a
        env2 = Monitor(e2, './video', force=True)

        # env = Monitor(gym.make('Walker2d-v3'), './video', force=True)
    elif action_mode =="hopper":
        e1 = gym.make('Hopper-v3')
        a = 0.5*e1.model.body_mass
        e1.model.body_mass[:]=a
        env1 = Monitor(e1, './video', force=True)

        e2 = gym.make('Hopper-v3')
        a = 1.4*e2.model.body_mass
        e2.model.body_mass[:]=a
        env2 = Monitor(e2, './video', force=True)
        # env = Monitor(gym.make('Hopper-v3'), './video', force=True)
    else:
        print("invalid env")
    # bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
    # env = FrameSkip(BlockManipulationEnv(bullet_client=bullet_client,
    #                                      scenario=scenario,
    #                                      control_rate_s=0.004,
    #                                              mass=0.005,
    #                                              size=0.0325,
    #                                              ),
    #                 frame_skip=frame_skip)

    # obs = env.reset()
    
    try:
        state = env1.reset()
        for i_step in range(n_frames_per_episode):
            action = actionSeq[i_step]#planner.plan(env, save_state, restore_state)
            env_action = action_transformation(action)
            obs, reward, done, info = env1.step(env_action)
            if done:
                print(done, i_step)
                break
        env1.close()
        state = env2.reset()
        for i_step in range(n_frames_per_episode):
            action = actionSeq[i_step]#planner.plan(env, save_state, restore_state)
            env_action = action_transformation(action)
            obs, reward, done, info = env2.step(env_action)
            if done:
                print(done, i_step)
                break
        env2.close()
            # real_rewards.append(reward)
            # episode_actions.append(env_action)
            # rgb_array = env.render(mode='rgb_array')
            # frames.append((rgb_array * 255).astype(np.uint8))
    except KeyboardInterrupt:
        logger.info(f'Interrupted at step {i_step}')

    # logger.info(f'np.mean(real_rewards): {np.mean(real_rewards)}')

    print('Saving video...')
    # try:

    #     # clip = mpy.ImageSequenceClip(frames, fps=60)
    #     # video_path = str(Path(output_dir) / f'plan1.mp4')
    #     # clip.write_videofile(video_path)

    # except Exception as e:
    #     logger.info(e)

    print('Saving actions...')
    episode_actions = np.array(episode_actions)
    np.save(str(Path(output_dir) / 'episode_actions.npy'), episode_actions)
    print(str(Path(output_dir) / 'episode_actions.npy'))
    # bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
    # env = FrameSkip(BlockManipulationEnv(bullet_client=bullet_client,
    #                                      scenario=scenario,
    #                                      control_rate_s=0.004,
    #                                              mass=0.1,
    #                                              size=0.0325,
    #                                              ),
    #                 frame_skip=frame_skip)
    # obs = env.reset()
    # episode_actions1 = []
    # frames1 = []
    # try:
    #     for i_step in range(n_frames_per_episode):
    #         action = actionSeq[i_step]#planner.plan(env, save_state, restore_state)
    #         env_action = action_transformation(action)
    #         obs, reward, done, info = env.step(env_action)
    #         #real_rewards.append(reward)
    #         episode_actions1.append(env_action)
    #         rgb_array = env.render(mode='rgb_array')
    #         frames1.append((rgb_array * 255).astype(np.uint8))
    # except KeyboardInterrupt:
    #     logger.info(f'Interrupted at step {i_step}')

    # logger.info(f'np.mean(real_rewards): {np.mean(real_rewards)}')

    # print('Saving video...')
    # try:
    #     clip1 = mpy.ImageSequenceClip(frames1, fps=60)
    #     video_path = str(Path(output_dir) / f'plan2.mp4')
    #     clip1.write_videofile(video_path)

    # except Exception as e:
    #     logger.info(e)

    #print('Saving actions...')
    #episode_actions = np.array(episode_actions)
    #np.save(str(Path(output_dir) / 'episode_actions.npy'), episode_actions)



if __name__ == "__main__":
    # print(f'SCENARIOS: {SCENARIOS}')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f'./tmp/mujoco_{timestamp}'
    Path(output_dir).mkdir()

    main(output_dir=output_dir, viz_progress=True)
