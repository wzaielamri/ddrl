import ray
import pickle5 as pickle
import os

from ray.tune.registry import get_trainable_cls
from ray.rllib.evaluation.worker_set import WorkerSet

import simulation_envs
import models
from evaluation.rollout_episodes import rollout_episodes, rollout_episodes_AttentionMap

"""
    Visualizing a learned (multiagent) controller,
    for evaluation or visualisation.
    
    This is adapted from rllib's rollout.py
    (github.com/ray/rllib/rollout.py)
"""

# Setting number of steps and episodes
num_steps = int(150)
num_episodes = int(1)

ray.init()

smoothness = 1.0

# Selecting checkpoint to load
config_checkpoints = [
    "/media/compute/homes/wzaielamri/ray_results/Cheetah_1_1_BipedMultiEnv_SixDecentral_AttentionMap_DiscreteInit0/PPO_BipedMultiEnv_SixDecentral_AttentionMap_d43b9_00000_0_2021-06-15_23-58-35/checkpoint_000313/checkpoint-313"]


# Afterwards put together using
# ffmpeg -framerate 20 -pattern_type glob -i '*.png' -filter:v scale=720:-1 -vcodec libx264 -pix_fmt yuv420p -g 1 out.mp4

# HF_10_QuantrupedMultiEnv_Local/PPO_QuantrupedMultiEnv_Local_1a49c_00003_3_2020-12-04_12-08-57
# HF_10_QuantrupedMultiEnv_TwoSides/PPO_QuantrupedMultiEnv_TwoSides_6654b_00006_6_2020-12-06_17-42-00
# HF_10_QuantrupedMultiEnv_FullyDecentral/PPO_QuantrupedMultiEnv_FullyDecentral_19697_00004_4_2020-12-04_12-08-56

for config_checkpoint in config_checkpoints:
    config_dir = os.path.dirname(config_checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")

    # Loading configuration for checkpoint.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    if os.path.isfile(config_path):
        with open(config_path, "rb") as f:
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

    save_image_dir = 'videos/' + \
        config_path.partition('MultiEnv_')[2].partition(
            '/')[0] + '_smoothn_' + str(smoothness)
    try:
        os.mkdir(save_image_dir)
    except:
        print("File exists!!!")
    # Rolling out simulation = stepping through simulation.
    rollout_episodes_AttentionMap(env, agent, num_episodes=num_episodes,
                     num_steps=num_steps, render=True, save_images=save_image_dir+"/img_")
    agent.stop()
    os.system('ffmpeg -framerate 20 -pattern_type glob -i "' + save_image_dir +
              '/*.png" -filter:v scale=720:-1,vflip -vcodec libx264 -pix_fmt yuv420p -g 1 ' + save_image_dir + '.mp4')
