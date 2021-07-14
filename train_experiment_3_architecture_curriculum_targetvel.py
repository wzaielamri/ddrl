import numpy as np
import gym
from gym import spaces

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray import tune
from ray.tune import grid_search
import time

import target_envs
import models

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--policy_scope", required=False)
parser.add_argument("--target_velocity", required=False)
args = parser.parse_args()
# Possible values:
#   QuantrupedMultiEnv_Centralized_TVel - single controller, global information
#   QuantrupedMultiEnv_FullyDecentral_TVel - four decentralized controlller, information
#       from the controlled leg only
#   QuantrupedMultiEnv_Local_TVel - four decentralized controlller, information
#       from the controlled leg plus both neighboring legs
#   QuantrupedMultiEnv_TwoSides_TVel - two decentralized controlller, one for each side,
#       information from the controlled legs

if 'policy_scope' in args and args.policy_scope:
    policy_scope = args.policy_scope
else:
    policy_scope = 'QuantrupedMultiEnv_Centralized_TVel'

if policy_scope == "QuantrupedMultiEnv_Local_TVel":
    from target_envs.quantruped_fourDecentralizedController_environments import Quantruped_Local_TVel_Env as QuantrupedEnv
elif policy_scope == "QuantrupedMultiEnv_FullyDecentral_TVel":
    from target_envs.quantruped_fourDecentralizedController_environments import QuantrupedFullyDecentralized_TVel_Env as QuantrupedEnv
elif policy_scope == "QuantrupedMultiEnv_EightFullyDecentral_TVel":
    from target_envs.quantruped_eightDecentralizedController_environments import QuantrupedEightFullyDecentralized_TVel_Env as QuantrupedEnv
elif policy_scope == "QuantrupedMultiEnv_EightDecentral_neighborJoint_TVel":
    from target_envs.quantruped_eightDecentralizedController_environments import QuantrupedDecentralized_neighborJoint_TVel_Env as QuantrupedEnv
elif policy_scope == "QuantrupedMultiEnv_TwoSides_TVel":
    from target_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoSideControllers_TVel_Env as QuantrupedEnv
else:
    from target_envs.quantruped_centralizedController_environment import Quantruped_Centralized_TVel_Env as QuantrupedEnv

#ray.init(num_cpus=6, ignore_reinit_error=True)
ray.init(ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

config['env'] = policy_scope
print("SELECTED ENVIRONMENT: ", policy_scope, " = ", QuantrupedEnv)

config['num_workers'] = 2
config['num_envs_per_worker'] = 4
# config['nump_gpus']=1

# used grid_search([4000, 16000, 65536], didn't matter too much
config['train_batch_size'] = 16000

# Baseline Defaults:
config['gamma'] = 0.99
config['lambda'] = 0.95

# again used grid_search([0., 0.01]) for diff. values from lit.
config['entropy_coeff'] = 0.

config['clip_param'] = 0.2

config['vf_loss_coeff'] = 0.5
#config['vf_clip_param'] = 4000.

config['observation_filter'] = 'MeanStdFilter'

config['sgd_minibatch_size'] = 128
config['num_sgd_iter'] = 10
config['lr'] = 3e-4
config['grad_clip'] = 0.5

config['model']['custom_model'] = "fc_glorot_uniform_init"
config['model']['fcnet_hiddens'] = [64, 64]

#config['seed'] = round(time.time())

# For running tune, we have to provide information on
# the multiagent which are part of the MultiEnvs
policies = QuantrupedEnv.return_policies()

config["multiagent"] = {
    "policies": policies,
    "policy_mapping_fn": QuantrupedEnv.policy_mapping_fn,
    "policies_to_train": QuantrupedEnv.policy_names,  # , "dec_B_policy"],
}

# grid_search([5e-4,5e-3,5e-2])
config['env_config']['ctrl_cost_weight'] = 0.25
# grid_search([5e-4,5e-3,5e-2])
config['env_config']['contact_cost_weight'] = 25e-3

# Parameters for defining environment:
# Heightfield smoothness (between 0.6 and 1.0 are OK)
config['env_config']['hf_smoothness'] = 1.0
# Defining curriculum learning
config['env_config']['curriculum_learning'] = False
config['env_config']['range_smoothness'] = [1., 0.6]
config['env_config']['range_last_timestep'] = 4000000

# Setting target velocity (range of up to 2.)
if 'target_velocity' in args and args.target_velocity:
    config['env_config']['target_velocity'] = float(args.target_velocity)


def on_train_result(info):
    result = info["result"]
    trainer = info["trainer"]
    timesteps_res = result["timesteps_total"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(lambda env: env.update_environment_after_epoch(timesteps_res)))


config["callbacks"] = {"on_train_result": on_train_result, }

# Call tune and run (for evaluation: 10 seeds up to 20M steps; only centralized controller
# required that much of time; decentralized controller should show very good results
# after 5M steps.
analysis = tune.run(
    "PPO",
    name=("Tvel_eight_" + policy_scope),
    num_samples=1,
    checkpoint_at_end=True,
    checkpoint_freq=312,
    stop={"timesteps_total": 5000000},
    config=config,
)
