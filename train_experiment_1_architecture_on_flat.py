from ray.rllib.models.catalog import ModelCatalog
import numpy as np
import gym
from gym import spaces

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray import tune
from ray.tune import grid_search
import time

import simulation_envs
import models

import argparse

# Switch between different approaches.
parser = argparse.ArgumentParser()
parser.add_argument("--policy_scope", required=False)
parser.add_argument("--mass_weight", required=False)
args = parser.parse_args()
# Possible values:
#   QuantrupedMultiEnv_Centralized - single controller, global information
#   QuantrupedMultiEnv_FullyDecentral - four decentralized controlller, information
#       from the controlled leg only
#   QuantrupedMultiEnv_SingleNeighbor - four decentralized controlller, information
#       from the controlled leg plus neighbor (ccw)
#   QuantrupedMultiEnv_SingleDiagonal - four decentralized controlller, information
#       from the controlled leg plus diagonal
#   QuantrupedMultiEnv_SingleToFront - four decentralized controlller, information
#       from the controlled leg plus one neighbor, for front legs from hind legs
#       for hind legs, the other hind leg
#   QuantrupedMultiEnv_Local - four decentralized controlller, information
#       from the controlled leg plus both neighboring legs
#   QuantrupedMultiEnv_TwoSides - two decentralized controlller, one for each side,
#       information from the controlled legs
#   QuantrupedMultiEnv_TwoDiags - two decentralized controlller, controlling a pair of
#       diagonal legs,
#       information from the controlled legs
#   QuantrupedMultiEnv_FullyDecentralGlobalCost - four decentralized controlller, information
#       from the controlled leg; variation: global costs are used.

if 'policy_scope' in args and args.policy_scope:
    policy_scope = args.policy_scope
else:
    policy_scope = 'BipedMultiEnv_Centralized'

if 'mass_weight' in args and args.mass_weight:
    mass_weight = args.mass_weight
else:
    mass_weight = 1

if policy_scope == "BipedMultiEnv_TwoSides":
    from simulation_envs.biped_twoDecentralizedController_environments import Biped_TwoSideControllers_Env as BipedEnv

elif policy_scope == "BipedMultiEnv_TwoSides_GCN":
    from simulation_envs.biped_twoDecentralizedController_environments import Biped_TwoSideControllers_GCN_Env as BipedEnv
elif policy_scope == "BipedMultiEnv_TwoSides_AllInfo":
    from simulation_envs.biped_twoDecentralizedController_environments import Biped_TwoSideControllers_AllInfo_Env as BipedEnv
elif policy_scope == "BipedMultiEnv_SixFullyDecentral":
    from simulation_envs.biped_sixDecentralizedController_environments import BipedSixFullyDecentralized_Env as BipedEnv
elif policy_scope == "BipedMultiEnv_SixDecentral_neighborJoints":
    from simulation_envs.biped_sixDecentralizedController_environments import BipedDecentralized_neighborJoints_Env as BipedEnv
elif policy_scope == "BipedMultiEnv_SixDecentral_neighborJointsAllInfo":
    from simulation_envs.biped_sixDecentralizedController_environments import BipedDecentralized_neighborJointsAllInfo_Env as BipedEnv
elif policy_scope == "BipedMultiEnv_SixDecentral_BioGraph":
    from simulation_envs.biped_sixDecentralizedController_environments import BipedSixFullyDecentralized_BioGraph_Env as BipedEnv
elif policy_scope == "BipedMultiEnv_SixDecentral_AttentionMap":
    from simulation_envs.biped_sixDecentralizedController_environments import BipedSixFullyDecentralized_AttentionMap_Env as BipedEnv
else:
    from simulation_envs.biped_centralizedController_environment import Biped_Centralized_Env as BipedEnv


# Init ray: First line on server, second for laptop
ray.init(num_cpus=10, ignore_reinit_error=True)
# ray.init(ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

config['env'] = policy_scope
print("SELECTED ENVIRONMENT: ", policy_scope, " = ", BipedEnv)

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


# Model
config['model']['custom_model'] = "RNNModel"

# LSTM
config['model']['lstm_cell_size'] = grid_search([8, 16, 32, 64])


#config['seed'] = round(time.time())

# For running tune, we have to provide information on
# the multiagent which are part of the MultiEnvs

policies = BipedEnv.return_policies(
    spaces.Box(-np.inf, np.inf, (17,), np.float64))


config["multiagent"] = {
    "policies": policies,
    "policy_mapping_fn": BipedEnv.policy_mapping_fn,
    "policies_to_train": BipedEnv.policy_names,  # , "dec_B_policy"],
}

config['env_config']['ctrl_cost_weight'] = 0.5  # grid_search([5e-4,5e-3,5e-2])
# grid_search([5e-4,5e-3,5e-2])
config['env_config']['contact_cost_weight'] = 5e-2

config['env_config']['mass_weight'] = float(mass_weight)

# Parameters for defining environment:
# Heightfield smoothness (between 0.6 and 1.0 are OK)
config['env_config']['hf_smoothness'] = 1.0
# Defining curriculum learning of smoothness
config['env_config']['curriculum_learning_hf'] = False
#config['env_config']['range_smoothness'] = [1., 0.6]
#config['env_config']['range_last_timestep_hf'] = 10000000

# update mass

# Defining curriculum learning of mass
config['env_config']['curriculum_learning_mass'] = False
config['env_config']['range_mass'] = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
config['env_config']['range_last_timestep_mass'] = [
    250000, 500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000, 2250000, 2500000]


# For curriculum learning: environment has to be updated every epoch
# added the callback class to solve callback warning

class editedCallbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        timesteps_res = result["timesteps_total"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(lambda env: env.update_environment_after_epoch(timesteps_res)))


config["callbacks"] = editedCallbacks  # {"on_train_result": on_train_result, }

# Call tune and run (for evaluation: 10 seeds up to 20M steps; only centralized controller
# required that much of time; decentralized controller should show very good results
# after 5M steps.
analysis = tune.run(
    "PPO",
    name=("Cheetah_1_"+str(mass_weight) + "_" +
          policy_scope+"RnnInitHidden8"),
    #name=("Cheetah_1_curriculumMass_" + policy_scope+"Cur"),
    num_samples=1,
    checkpoint_at_end=True,
    checkpoint_freq=312,
    stop={"timesteps_total": 5000000},
    config=config,
)
