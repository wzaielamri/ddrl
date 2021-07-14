import ray
import pickle5 as pickle
import os

from ray.tune.registry import get_trainable_cls
from ray.rllib.evaluation.worker_set import WorkerSet

import sys
sys.path.insert(0,'../')


import target_envs
import models
from evaluation.rollout_episodes import rollout_episodes

"""
    Visualizing a learned (multiagent) controller,
    for evaluation or visualisation.
    
    This is adapted from rllib's rollout.py
    (github.com/ray/rllib/rollout.py)
"""

# Setting number of steps and episodes
num_steps = int(300)
num_episodes = int(1)

ray.init()

smoothness_list = [1.0, 0.8, 0.6]
target_velocity_list = [0.5, 1.0, 1.5, 2.0]

# Selecting checkpoint to load
config_checkpoints = ["/home/nitro/ray_results/Tvel_eight_QuantrupedMultiEnv_Centralized_TVel/PPO_QuantrupedMultiEnv_Centralized_TVel_36825_00000_0_2021-03-04_00-43-10/checkpoint_1250/checkpoint-1250",
    "/home/nitro/ray_results/Tvel_eight_QuantrupedMultiEnv_FullyDecentral_TVel/PPO_QuantrupedMultiEnv_FullyDecentral_TVel_aa7d9_00000_0_2021-03-04_09-57-36/checkpoint_1250/checkpoint-1250",
    "/home/nitro/ray_results/Tvel_eight_QuantrupedMultiEnv_EightFullyDecentral_TVel/PPO_QuantrupedMultiEnv_EightFullyDecentral_TVel_17604_00000_0_2021-03-08_22-34-41/checkpoint_1250/checkpoint-1250",
    "/home/nitro/ray_results/Tvel_eight_QuantrupedMultiEnv_EightDecentral_neighborJoint_TVel/PPO_QuantrupedMultiEnv_EightDecentral_neighborJoint_TVel_6c84d_00000_0_2021-03-10_17-26-53/checkpoint_1250/checkpoint-1250"]
#config_checkpoint = config_checkpoints[3]

# Afterwards put together using
# ffmpeg -framerate 20 -pattern_type glob -i '*.png' -filter:v scale=720:-1 -vcodec libx264 -pix_fmt yuv420p -g 1 out.mp4

for config_checkpoint in config_checkpoints:


    config_dir = os.path.dirname(config_checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")

    for smoothness in smoothness_list:
        for target_velocity in target_velocity_list:
            # Loading configuration for checkpoint.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "rb") as f:
                        config = pickle.load(f)
                except:
                    with open(config_dir + "/../../params_py36.pkl", "rb") as f:
                        config = pickle.load(f)

            # Starting ray and setting up ray.
            if "num_workers" in config:
                config["num_workers"] = min(2, config["num_workers"])
            cls = get_trainable_cls('PPO')
            # Setting config values (required for compatibility between versions)
            config["create_env_on_driver"] = True
            config['env_config']['hf_smoothness'] = smoothness
            if "no_eager_on_workers" in config:
                del config["no_eager_on_workers"]

            # Load state from checkpoint.
            agent = cls(env=config['env'], config=config)
            agent.restore(config_checkpoint)

            # Retrieve environment for the trained agent.
            if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
                env = agent.workers.local_worker().env

            save_image_dir = 'videos/' + config_path.partition('MultiEnv_')[2].partition('/')[0] + '_smoothn_' + str(smoothness) + '_tvel_' + str(target_velocity)
            os.mkdir(save_image_dir)
            # Rolling out simulation = stepping through simulation. 
            rollout_episodes(env, agent, num_episodes=num_episodes, num_steps=num_steps, render=True, save_images=save_image_dir+"/img_", tvel=target_velocity)
            agent.stop()

            os.system('ffmpeg -framerate 20 -pattern_type glob -i "' + save_image_dir + '/*.png" -filter:v scale=720:-1 -vcodec libx264 -pix_fmt yuv420p -g 1 ' + save_image_dir + '.mp4')
            
            
