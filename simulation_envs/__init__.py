from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit
from simulation_envs.biped import BipedEnv
from simulation_envs.cheetah_mujoco_2 import CheetahEnvMujoco2

# Importing the different multiagent environments.
from simulation_envs.biped_adaptor_multi_environment import BipedMultiPoliciesEnv

from simulation_envs.biped_twoDecentralizedController_environments import Biped_TwoSideControllers_Env
from simulation_envs.biped_twoDecentralizedController_environments import Biped_TwoSideControllers_AllInfo_Env
from simulation_envs.biped_twoDecentralizedController_environments import Biped_TwoSideControllers_GCN_Env

# for six agents
from simulation_envs.biped_sixDecentralizedController_environments import BipedSixFullyDecentralized_Env
from simulation_envs.biped_sixDecentralizedController_environments import BipedDecentralized_neighborJoints_Env
from simulation_envs.biped_sixDecentralizedController_environments import BipedDecentralized_neighborJointsAllInfo_Env
from simulation_envs.biped_sixDecentralizedController_environments import BipedSixFullyDecentralized_BioGraph_Env
from simulation_envs.biped_sixDecentralizedController_environments import BipedSixFullyDecentralized_AttentionMap_Env


# Register Gym environment.
register(
    id='Biped-v3',
    entry_point='simulation_envs.biped:BipedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Cheetah_Muj2-v3',
    entry_point='simulation_envs.cheetah_mujoco_2:CheetahEnvMujoco2',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

# Register single agent ray environment (wrapping gym environment).
register_env("Cheetah_Muj2-v3",
             lambda config: TimeLimit(CheetahEnvMujoco2(), max_episode_steps=1000))
register_env("Biped-v3",
             lambda config: TimeLimit(BipedEnv(), max_episode_steps=1000))

# Register multiagent environments (allowing individual access to individual legs).
register_env("BipedMultiEnv_Centralized",
             lambda config: BipedMultiPoliciesEnv(config))


register_env("BipedMultiEnv_TwoSides",
             lambda config: Biped_TwoSideControllers_Env(config))
register_env("BipedMultiEnv_TwoSides_AllInfo",
             lambda config: Biped_TwoSideControllers_AllInfo_Env(config))
register_env("BipedMultiEnv_TwoSides_GCN",
             lambda config: Biped_TwoSideControllers_GCN_Env(config))


register_env("BipedMultiEnv_SixFullyDecentral",
             lambda config: BipedSixFullyDecentralized_Env(config))
register_env("BipedMultiEnv_SixDecentral_neighborJoints",
             lambda config: BipedDecentralized_neighborJoints_Env(config))
register_env("BipedMultiEnv_SixDecentral_neighborJointsAllInfo",
             lambda config: BipedDecentralized_neighborJointsAllInfo_Env(config))
register_env("BipedMultiEnv_SixDecentral_BioGraph",
             lambda config: BipedSixFullyDecentralized_BioGraph_Env(config))
register_env("BipedMultiEnv_SixDecentral_AttentionMap",
             lambda config: BipedSixFullyDecentralized_AttentionMap_Env(config))
