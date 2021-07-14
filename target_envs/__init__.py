from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit
from target_envs.quantruped_v3 import QuAntruped_TVel_Env

# Importing the different multiagent environments.
from target_envs.quantruped_centralizedController_environment import Quantruped_Centralized_TVel_Env
from target_envs.quantruped_twoDecentralizedController_environments import Quantruped_TwoSideControllers_TVel_Env
from target_envs.quantruped_fourDecentralizedController_environments import Quantruped_Local_TVel_Env
from target_envs.quantruped_fourDecentralizedController_environments import QuantrupedFullyDecentralized_TVel_Env

# for eight Agent
from target_envs.quantruped_eightDecentralizedController_environments import QuantrupedEightFullyDecentralized_TVel_Env
from target_envs.quantruped_eightDecentralizedController_environments import QuantrupedDecentralized_neighborJoint_TVel_Env


# Register Gym environment.
register(
    id='QuAntrupedTvel-v3',
    entry_point='target_envs.quantruped_v3:QuAntruped_TVel_Env',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

# Register single agent ray environment (wrapping gym environment).
register_env("QuAntrupedTvel-v3",
             lambda config: TimeLimit(QuAntruped_TVel_Env(), max_episode_steps=1000))

# Register multiagent environments (allowing individual access to individual legs).
register_env("QuantrupedMultiEnv_Centralized_TVel",
             lambda config: Quantruped_Centralized_TVel_Env(config))
register_env("QuantrupedMultiEnv_TwoSides_TVel",
             lambda config: Quantruped_TwoSideControllers_TVel_Env(config))
register_env("QuantrupedMultiEnv_Local_TVel",
             lambda config: Quantruped_Local_TVel_Env(config))
register_env("QuantrupedMultiEnv_FullyDecentral_TVel",
             lambda config: QuantrupedFullyDecentralized_TVel_Env(config))
register_env("QuantrupedMultiEnv_EightFullyDecentral_TVel",
             lambda config: QuantrupedEightFullyDecentralized_TVel_Env(config))
register_env("QuantrupedMultiEnv_EightDecentral_neighborJoint_TVel",
             lambda config: QuantrupedDecentralized_neighborJoint_TVel_Env(config))
