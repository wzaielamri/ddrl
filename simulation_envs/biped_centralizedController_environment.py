import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from simulation_envs import BipedMultiPoliciesEnv


class Biped_Centralized_Env(BipedMultiPoliciesEnv):
    """ Derived environment for control of the two-legged agent.
        Allows to instantiate multiple agents for control.

        Centralized approach: Single agent (as standard DRL approach)
        controls all degrees of freedom of the agent.

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
        - concatenate_actions: how to integrate the control signals from the controllers
    """

    # This is ordering of the policies as applied here:
    policy_names = ["central_policy"]

    def __init__(self, config):
        self.obs_indices = {}
        """ 
         Observation space for the Biped model:
         exclude_current_positions_from_observation

         - rootx     slider      position (m)
         - rootz     slider      position (m)
         - rooty     hinge       angle (rad)
         - bthigh    hinge       angle (rad)
         - bshin     hinge       angle (rad)
         - bfoot     hinge       angle (rad)
         - fthigh    hinge       angle (rad)
         - fshin     hinge       angle (rad)
         - ffoot     hinge       angle (rad)
         - rootx     slider      velocity (m/s)
         - rootz     slider      velocity (m/s)
         - rooty     hinge       angular velocity (rad/s)
         - bthigh    hinge       angular velocity (rad/s)
         - bshin     hinge       angular velocity (rad/s)
         - bfoot     hinge       angular velocity (rad/s)
         - fthigh    hinge       angular velocity (rad/s)
         - fshin     hinge       angular velocity (rad/s)
         - ffoot     hinge       angular velocity (rad/s)

         For action ordering is:
             "bthigh"
             "bshin"
             "bfoot"
             "fthigh"
             "fshin"
             "ffoot"
         """

        self.obs_indices["central_policy"] = range(0, 17)
        super().__init__(config)

    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name], ]
        return obs_distributed

    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        return Biped_Centralized_Env.policy_names[0]

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (17,), np.float64)
        policies = {
            Biped_Centralized_Env.policy_names[0]: (None,
                                                    obs_space, spaces.Box(np.array([-1., -1., -1., -1., -1., -1.]), np.array([+1., +1., +1., +1., +1., +1.])), {}),
        }
        return policies
