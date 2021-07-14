from typing import Tuple
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from simulation_envs import BipedMultiPoliciesEnv


class BipedSixControllerSuperEnv(BipedMultiPoliciesEnv):
    """ Derived environment for control of the two-legged agent.
        Allows to instantiate multiple agents for control.

        Super class for all decentralized controller - control is split
        into eight different, concurrent control units (policies)
        each instantiated as a single agent.

        Class defines
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in derived classes and differs between the different architectures.
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers
        - concatenate_actions: how to integrate the control signals from the controllers
    """

    def distribute_observations(self, obs_full):
        """
        Construct dictionary that routes to each policy only the relevant
        local information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name], ]
        return obs_distributed

    def distribute_contact_cost(self):
        contact_cost = {}
        # print("CONTACT COST")
        # from mujoco_py import functions
        # functions.mj_rnePostConstraint(self.env.model, self.env.data)
        # print("From Ant Env: ", self.env.contact_cost)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.env.contact_cost_weight * \
            np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/6.   # 0 floor / 1
        contact_cost[self.policy_names[0]
                     ] = global_contact_costs + np.sum(contact_costs[2:5])/3  # teile der Beine
        contact_cost[self.policy_names[1]
                     ] = global_contact_costs + np.sum(contact_costs[2:5])/3
        contact_cost[self.policy_names[2]
                     ] = global_contact_costs + np.sum(contact_costs[2:5])/3  # teile der Beine

        contact_cost[self.policy_names[3]
                     ] = global_contact_costs + np.sum(contact_costs[5:8])/3
        contact_cost[self.policy_names[4]] = global_contact_costs + \
            np.sum(contact_costs[5:8])/3
        contact_cost[self.policy_names[5]] = global_contact_costs + \
            np.sum(contact_costs[5:8])/3

        # print(contact_cost)
        # sum_c = 0.
        # for i in self.policy_names:
        #   sum_c += contact_cost[i]
        # print("Calculated: ", sum_c)
        return contact_cost

    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR:0,1 - FL:0,1 - HL:0,1 - HR:0,1
        actions = np.concatenate((action_dict[self.policy_names[0]
                                              ], action_dict[self.policy_names[1]],
                                  action_dict[self.policy_names[2]
                                              ], action_dict[self.policy_names[3]],
                                  action_dict[self.policy_names[4]], action_dict[self.policy_names[5]]))
        return actions

    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_bthigh"):
            return "policy_bthigh"
        elif agent_id.startswith("policy_bshin"):
            return "policy_bshin"
        elif agent_id.startswith("policy_bfoot"):
            return "policy_bfoot"
        elif agent_id.startswith("policy_fthigh"):
            return "policy_fthigh"
        elif agent_id.startswith("policy_fshin"):
            return "policy_fshin"
        elif agent_id.startswith("policy_ffoot"):
            return "policy_ffoot"
        else:
            return "policy_ffoot"


class BipedSixFullyDecentralized_Env(BipedSixControllerSuperEnv):
    """ Derived environment for control of the two-legged agent.
        Uses six different, concurrent control units (policies)
        each instantiated as a single agent.

        Input scope of each controller: only the controlled leg.

        Class defines
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
    """

    # This is ordering of the policies as applied here:
    policy_names = ["policy_bthigh", "policy_bshin", "policy_bfoot", "policy_fthigh",
                    "policy_fshin", "policy_ffoot"]

    def __init__(self, config):
        self.obs_indices = {}
        """
         Observation space for the Biped model:
         exclude_current_positions_from_observation

          - rootx     slider      position (m)
         0 - rootz     slider      position (m)
         1 - rooty     hinge       angle (rad)
         2 - bthigh    hinge       angle (rad)
         3 - bshin     hinge       angle (rad)
         4 - bfoot     hinge       angle (rad)
         5 - fthigh    hinge       angle (rad)
         6 - fshin     hinge       angle (rad)
         7 - ffoot     hinge       angle (rad)
         8 - rootx     slider      velocity (m/s)
         9 - rootz     slider      velocity (m/s)
         10- rooty     hinge       angular velocity (rad/s)
         11- bthigh    hinge       angular velocity (rad/s)
         12- bshin     hinge       angular velocity (rad/s)
         13- bfoot     hinge       angular velocity (rad/s)
         14- fthigh    hinge       angular velocity (rad/s)
         15- fshin     hinge       angular velocity (rad/s)
         16- ffoot     hinge       angular velocity (rad/s)

         For action ordering is:
             "bthigh"
             "bshin"
             "bfoot"
             "fthigh"
             "fshin"
             "ffoot"
         """

        self.obs_indices["policy_bthigh"] = [0, 1, 8, 9, 10, 2, 11]
        self.obs_indices["policy_bshin"] = [0, 1, 8, 9, 10, 3, 12]
        self.obs_indices["policy_bfoot"] = [0, 1, 8, 9, 10, 4, 13]

        self.obs_indices["policy_fthigh"] = [0, 1, 8, 9, 10, 5, 14]

        self.obs_indices["policy_fshin"] = [0, 1, 8, 9, 10, 6, 15]
        self.obs_indices["policy_ffoot"] = [0, 1, 8, 9, 10, 7, 16]

        super().__init__(config)

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (7,), np.float64)
        policies = {
            BipedSixFullyDecentralized_Env.policy_names[0]: (None,
                                                             obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_Env.policy_names[1]: (None,
                                                             obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_Env.policy_names[2]: (None,
                                                             obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_Env.policy_names[3]: (None,
                                                             obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_Env.policy_names[4]: (None,
                                                             obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_Env.policy_names[5]: (None,
                                                             obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
        }
        return policies


class BipedDecentralized_neighborJoints_Env(BipedSixControllerSuperEnv):
    """ Derived environment for control of the two-legged agent.
        Uses six different, concurrent control units (policies)
        each instantiated as a single agent.

        Input scope of each controller:
        - controlled leg
        - plus from an additional neighboring joint

        Class defines
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
    """

    # This is ordering of the policies as applied here:
    policy_names = ["policy_bthigh", "policy_bshin", "policy_bfoot", "policy_fthigh",
                    "policy_fshin", "policy_ffoot"]

    def __init__(self, config):
        self.obs_indices = {}
        """
         Observation space for the Biped model:
         exclude_current_positions_from_observation

          - rootx     slider      position (m)
         0 - rootz     slider      position (m)
         1 - rooty     hinge       angle (rad)
         2 - bthigh    hinge       angle (rad)
         3 - bshin     hinge       angle (rad)
         4 - bfoot     hinge       angle (rad)
         5 - fthigh    hinge       angle (rad)
         6 - fshin     hinge       angle (rad)
         7 - ffoot     hinge       angle (rad)
         8 - rootx     slider      velocity (m/s)
         9 - rootz     slider      velocity (m/s)
         10- rooty     hinge       angular velocity (rad/s)
         11- bthigh    hinge       angular velocity (rad/s)
         12- bshin     hinge       angular velocity (rad/s)
         13- bfoot     hinge       angular velocity (rad/s)
         14- fthigh    hinge       angular velocity (rad/s)
         15- fshin     hinge       angular velocity (rad/s)
         16- ffoot     hinge       angular velocity (rad/s)

         For action ordering is:
             "bthigh"
             "bshin"
             "bfoot"
             "fthigh"
             "fshin"
             "ffoot"
         """

        self.obs_indices["policy_bthigh"] = [
            0, 1, 8, 9, 10, 2, 3, 4, 11, 12, 13]
        self.obs_indices["policy_bshin"] = [
            0, 1, 8, 9, 10, 2, 3, 4, 11, 12, 13]
        self.obs_indices["policy_bfoot"] = [
            0, 1, 8, 9, 10, 2, 3, 4, 11, 12, 13]

        self.obs_indices["policy_fthigh"] = [
            0, 1, 8, 9, 10, 5, 6, 7, 14, 15, 16]
        self.obs_indices["policy_fshin"] = [
            0, 1, 8, 9, 10, 5, 6, 7, 14, 15, 16]
        self.obs_indices["policy_ffoot"] = [
            0, 1, 8, 9, 10, 5, 6, 7, 14, 15, 16]

        super().__init__(config)

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (11,), np.float64)
        policies = {
            BipedDecentralized_neighborJoints_Env.policy_names[0]: (None,
                                                                    obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedDecentralized_neighborJoints_Env.policy_names[1]: (None,
                                                                    obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedDecentralized_neighborJoints_Env.policy_names[2]: (None,
                                                                    obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedDecentralized_neighborJoints_Env.policy_names[3]: (None,
                                                                    obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedDecentralized_neighborJoints_Env.policy_names[4]: (None,
                                                                    obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedDecentralized_neighborJoints_Env.policy_names[5]: (None,
                                                                    obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
        }
        return policies


class BipedDecentralized_neighborJointsAllInfo_Env(BipedSixControllerSuperEnv):
    """ Derived environment for control of the two-legged agent.
        Uses six different, concurrent control units (policies)
        each instantiated as a single agent.

        Input scope of each controller:
        - controlled leg
        - plus from all other joints

        Class defines
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
    """

    # This is ordering of the policies as applied here:
    policy_names = ["policy_bthigh", "policy_bshin", "policy_bfoot", "policy_fthigh",
                    "policy_fshin", "policy_ffoot"]

    def __init__(self, config):
        self.obs_indices = {}
        """
         Observation space for the Biped model:
         exclude_current_positions_from_observation

          - rootx     slider      position (m)
         0 - rootz     slider      position (m)
         1 - rooty     hinge       angle (rad)
         2 - bthigh    hinge       angle (rad)
         3 - bshin     hinge       angle (rad)
         4 - bfoot     hinge       angle (rad)
         5 - fthigh    hinge       angle (rad)
         6 - fshin     hinge       angle (rad)
         7 - ffoot     hinge       angle (rad)
         8 - rootx     slider      velocity (m/s)
         9 - rootz     slider      velocity (m/s)
         10- rooty     hinge       angular velocity (rad/s)
         11- bthigh    hinge       angular velocity (rad/s)
         12- bshin     hinge       angular velocity (rad/s)
         13- bfoot     hinge       angular velocity (rad/s)
         14- fthigh    hinge       angular velocity (rad/s)
         15- fshin     hinge       angular velocity (rad/s)
         16- ffoot     hinge       angular velocity (rad/s)

         For action ordering is:
             "bthigh"
             "bshin"
             "bfoot"
             "fthigh"
             "fshin"
             "ffoot"
         """

        self.obs_indices["policy_bthigh"] = range(0, 17)
        self.obs_indices["policy_bshin"] = range(0, 17)
        self.obs_indices["policy_bfoot"] = range(0, 17)

        self.obs_indices["policy_fthigh"] = range(0, 17)
        self.obs_indices["policy_fshin"] = range(0, 17)
        self.obs_indices["policy_ffoot"] = range(0, 17)

        super().__init__(config)

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (17,), np.float64)
        policies = {
            BipedDecentralized_neighborJointsAllInfo_Env.policy_names[0]: (None,
                                                                           obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedDecentralized_neighborJointsAllInfo_Env.policy_names[1]: (None,
                                                                           obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedDecentralized_neighborJointsAllInfo_Env.policy_names[2]: (None,
                                                                           obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedDecentralized_neighborJointsAllInfo_Env.policy_names[3]: (None,
                                                                           obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedDecentralized_neighborJointsAllInfo_Env.policy_names[4]: (None,
                                                                           obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedDecentralized_neighborJointsAllInfo_Env.policy_names[5]: (None,
                                                                           obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
        }
        return policies


class BipedSixFullyDecentralized_BioGraph_Env(BipedSixControllerSuperEnv):
    """ Derived environment for control of the two-legged agent.
        Uses six different, concurrent control units (policies)
        each instantiated as a single agent. The previous action of the agents is used for the next action.

        There is one controller for each side of the agent.
        Input scope of each controller:
        - one leg of that side.

        Class defines
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers
        - concatenate_actions: how to integrate the control signals from the controllers
    """
    # This is ordering of the policies as applied here:
    policy_names = ["policy_bthigh", "policy_bshin", "policy_bfoot", "policy_fthigh",
                    "policy_fshin", "policy_ffoot"]

    def __init__(self, config):
        self.obs_indices = {}
        self.act_indices = {}
        """
         Observation space for the Biped model:
         exclude_current_positions_from_observation

          - rootx     slider      position (m)
         0 - rootz     slider      position (m)
         1 - rooty     hinge       angle (rad)
         2 - bthigh    hinge       angle (rad)
         3 - bshin     hinge       angle (rad)
         4 - bfoot     hinge       angle (rad)
         5 - fthigh    hinge       angle (rad)
         6 - fshin     hinge       angle (rad)
         7 - ffoot     hinge       angle (rad)
         8 - rootx     slider      velocity (m/s)
         9 - rootz     slider      velocity (m/s)
         10- rooty     hinge       angular velocity (rad/s)
         11- bthigh    hinge       angular velocity (rad/s)
         12- bshin     hinge       angular velocity (rad/s)
         13- bfoot     hinge       angular velocity (rad/s)
         14- fthigh    hinge       angular velocity (rad/s)
         15- fshin     hinge       angular velocity (rad/s)
         16- ffoot     hinge       angular velocity (rad/s)

         For action ordering is:
            0- "bthigh"
            1- "bshin"
            2- "bfoot"
            3- "fthigh"
            4- "fshin"
            5- "ffoot"
         """

        self.obs_indices["policy_bthigh"] = [0, 1, 8, 9, 10, 2, 11]
        self.obs_indices["policy_bshin"] = [0, 1, 8, 9, 10, 3, 12]
        self.obs_indices["policy_bfoot"] = [0, 1, 8, 9, 10, 4, 13]

        self.obs_indices["policy_fthigh"] = [0, 1, 8, 9, 10, 5, 14]
        self.obs_indices["policy_fshin"] = [0, 1, 8, 9, 10, 6, 15]
        self.obs_indices["policy_ffoot"] = [0, 1, 8, 9, 10, 7, 16]

        # Each controller only gets actions from the other body side
        self.act_indices["policy_bthigh"] = [3]
        self.act_indices["policy_bshin"] = [0]
        self.act_indices["policy_bfoot"] = [1]

        self.act_indices["policy_fthigh"] = [0]
        self.act_indices["policy_fshin"] = [3]
        self.act_indices["policy_ffoot"] = [4]

        super().__init__(config)

    def distribute_observations(self, obs_full, prev_actions):
        """
        Construct dictionary that routes to each policy only the relevant
        local information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = np.concatenate((
                obs_full[self.obs_indices[policy_name], ], prev_actions[self.act_indices[policy_name], ]))
        return obs_distributed

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (8,), np.float64)
        policies = {
            BipedSixFullyDecentralized_BioGraph_Env.policy_names[0]: (None,
                                                                      obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_BioGraph_Env.policy_names[1]: (None,
                                                                      obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_BioGraph_Env.policy_names[2]: (None,
                                                                      obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_BioGraph_Env.policy_names[3]: (None,
                                                                      obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_BioGraph_Env.policy_names[4]: (None,
                                                                      obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_BioGraph_Env.policy_names[5]: (None,
                                                                      obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
        }
        return policies

    def reset(self):
        obs_original = self.env.reset()

        # added for printing
        self.acc_forw_rew = 0
        self.acc_ctrl_cost = 0
        self.acc_contact_cost = 0
        self._elapsed_steps = 0
        self.acc_step = 0
        return self.distribute_observations(obs_original, np.zeros(6))

    def step(self, action_dict):
        # Stepping the environment.

        # Use with mujoco 2.
        # functions.mj_rnePostConstraint(self.env.model, self.env.data)

        # Combine actions from all agents and step environment.
        obs_full, rew_w, done_w, info_w = self.env.step(self.concatenate_actions(
            action_dict))  # self.env.step( np.concatenate( (action_dict[self.policy_A],
        # action_dict[self.policy_B]) ))

        # Distribute observations and rewards to the individual agents.
        obs_dict = self.distribute_observations(obs_full, self.concatenate_actions(
            action_dict))
        rew_dict = self.distribute_reward(rew_w, info_w, action_dict)

        done = {
            "__all__": done_w,
        }

        # uncommented
        self.acc_forw_rew += info_w['reward_run']
        self.acc_ctrl_cost += info_w['reward_ctrl']
        self.acc_contact_cost += info_w['reward_contact']
        self.acc_step += 1

        # Print results from an episode.
        if done_w or self.acc_step >= self._max_episode_steps:
            print("Mass : ", self.mass_weight * self.cheetah_init_mass, "/ REWARDS: forw: ", info_w['reward_run'], " / ", self.acc_forw_rew/self.acc_step, "; ctr: ",
                  info_w['reward_ctrl'], " / ", self.acc_ctrl_cost /
                  (self.acc_step*self.env.ctrl_cost_weight), "; cont: ",
                  info_w['reward_contact'], " / ", self.acc_contact_cost/(self.acc_step*self.env.contact_cost_weight), self.env.contact_cost_weight)

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info_w['TimeLimit.truncated'] = not done
            done["__all__"] = True

        return obs_dict, rew_dict, done, {}


class BipedSixFullyDecentralized_AttentionMap_Env(BipedMultiPoliciesEnv):
    """ Derived environment for control of the two-legged agent.
        Uses six different, concurrent control units (policies)
        each instantiated as a single agent. The previous action of the agents is used for the next action.

        There is one controller for each side of the agent.
        Input scope of each controller:
        - one leg of that side.

        Class defines
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers
        - concatenate_actions: how to integrate the control signals from the controllers
    """
    # This is ordering of the policies as applied here:
    policy_names = ["policy_bthigh", "policy_bshin", "policy_bfoot", "policy_fthigh",
                    "policy_fshin", "policy_ffoot", "policy_central"]

    def __init__(self, config):
        self.obs_indices = {}
        self.act_indices = {}
        """
         Observation space for the Biped model:
         exclude_current_positions_from_observation

          - rootx     slider      position (m)
         0 - rootz     slider      position (m)
         1 - rooty     hinge       angle (rad)
         2 - bthigh    hinge       angle (rad)
         3 - bshin     hinge       angle (rad)
         4 - bfoot     hinge       angle (rad)
         5 - fthigh    hinge       angle (rad)
         6 - fshin     hinge       angle (rad)
         7 - ffoot     hinge       angle (rad)
         8 - rootx     slider      velocity (m/s)
         9 - rootz     slider      velocity (m/s)
         10- rooty     hinge       angular velocity (rad/s)
         11- bthigh    hinge       angular velocity (rad/s)
         12- bshin     hinge       angular velocity (rad/s)
         13- bfoot     hinge       angular velocity (rad/s)
         14- fthigh    hinge       angular velocity (rad/s)
         15- fshin     hinge       angular velocity (rad/s)
         16- ffoot     hinge       angular velocity (rad/s)

         For action ordering is:
            0- "bthigh"
            1- "bshin"
            2- "bfoot"
            3- "fthigh"
            4- "fshin"
            5- "ffoot"
         """

        self.obs_indices["policy_bthigh"] = range(17)
        self.obs_indices["policy_bshin"] = range(17)
        self.obs_indices["policy_bfoot"] = range(17)

        self.obs_indices["policy_fthigh"] = range(17)
        self.obs_indices["policy_fshin"] = range(17)
        self.obs_indices["policy_ffoot"] = range(17)

        self.obs_indices["policy_central"] = range(17)

        # Each controller only gets actions from the other body side

        self.act_indices["policy_bthigh"] = []
        self.act_indices["policy_bshin"] = []
        self.act_indices["policy_bfoot"] = []

        self.act_indices["policy_fthigh"] = []
        self.act_indices["policy_fshin"] = []
        self.act_indices["policy_ffoot"] = []

        self.act_indices["policy_central"] = range(6)

        super().__init__(config)

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (17,), np.float64)
        obs_space_central = spaces.Box(-np.inf, np.inf, (23,), np.float64)

        
        spaceBox = spaces.Discrete(2)
        central_spaces = spaces.Tuple((spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox, spaceBox))
                          
        """
        spaceBox = spaces.Box(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ]), np.array(
            [+1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., +1., ]))

        central_spaces = spaceBox
        """

        policies = {
            BipedSixFullyDecentralized_AttentionMap_Env.policy_names[0]: (None,
                                                                          obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_AttentionMap_Env.policy_names[1]: (None,
                                                                          obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_AttentionMap_Env.policy_names[2]: (None,
                                                                          obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_AttentionMap_Env.policy_names[3]: (None,
                                                                          obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_AttentionMap_Env.policy_names[4]: (None,
                                                                          obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_AttentionMap_Env.policy_names[5]: (None,
                                                                          obs_space, spaces.Box(np.array([-1.]), np.array([+1.])), {}),
            BipedSixFullyDecentralized_AttentionMap_Env.policy_names[6]: (None,
                                                                          obs_space_central, central_spaces, {}),
        }
        return policies

    def reset(self):
        obs_original = self.env.reset()

        # added for printing
        self.acc_forw_rew = 0
        self.acc_ctrl_cost = 0
        self.acc_contact_cost = 0
        self._elapsed_steps = 0
        self.acc_step = 0

        """
        central_act_init = (1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                            1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0,
                            1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
                            1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
                            1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                            1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1)
        """
        central_act_init = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        
        
        return self.distribute_observations(obs_original, np.zeros(6), central_act_init)

    def step(self, action_dict):
        # Stepping the environment.

        # Use with mujoco 2.
        # functions.mj_rnePostConstraint(self.env.model, self.env.data)

        # Combine actions from all agents and step environment.

        central_act = action_dict[self.policy_names[-1]]

        actions = self.concatenate_actions(action_dict)

        # self.env.step( np.concatenate( (action_dict[self.policy_A],
        obs_full, rew_w, done_w, info_w = self.env.step(actions)
        # action_dict[self.policy_B]) ))

        # Distribute observations and rewards to the individual agents.
        obs_dict = self.distribute_observations(obs_full, actions, central_act)
        rew_dict = self.distribute_reward(rew_w, info_w, action_dict)

        done = {
            "__all__": done_w,
        }
        # uncommented
        self.acc_forw_rew += info_w['reward_run']
        self.acc_ctrl_cost += info_w['reward_ctrl']
        self.acc_contact_cost += info_w['reward_contact']
        self.acc_step += 1

        # Print results from an episode.
        if done_w or self.acc_step >= self._max_episode_steps:
            print("Mass : ", self.mass_weight * self.cheetah_init_mass, "/ REWARDS: forw: ", info_w['reward_run'], " / ", self.acc_forw_rew/self.acc_step, "; ctr: ",
                  info_w['reward_ctrl'], " / ", self.acc_ctrl_cost /
                  (self.acc_step*self.env.ctrl_cost_weight), "; cont: ",
                  info_w['reward_contact'], " / ", self.acc_contact_cost/(self.acc_step*self.env.contact_cost_weight), self.env.contact_cost_weight)

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info_w['TimeLimit.truncated'] = not done
            done["__all__"] = True

        return obs_dict, rew_dict, done, {}

    def distribute_contact_cost(self):
        contact_cost = {}
        # print("CONTACT COST")
        # from mujoco_py import functions
        # functions.mj_rnePostConstraint(self.env.model, self.env.data)
        # print("From Ant Env: ", self.env.contact_cost)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.env.contact_cost_weight * \
            np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/6.   # 0 floor / 1
        contact_cost[self.policy_names[0]
                     ] = global_contact_costs + np.sum(contact_costs[2:5])/3  # teile der Beine
        contact_cost[self.policy_names[1]
                     ] = global_contact_costs + np.sum(contact_costs[2:5])/3
        contact_cost[self.policy_names[2]
                     ] = global_contact_costs + np.sum(contact_costs[2:5])/3  # teile der Beine

        contact_cost[self.policy_names[3]
                     ] = global_contact_costs + np.sum(contact_costs[5:8])/3
        contact_cost[self.policy_names[4]] = global_contact_costs + \
            np.sum(contact_costs[5:8])/3
        contact_cost[self.policy_names[5]] = global_contact_costs + \
            np.sum(contact_costs[5:8])/3

        # central
        contact_cost[self.policy_names[6]] = self.env.contact_cost_weight * np.sum(
            np.square(contact_forces))

        # print(contact_cost)
        # sum_c = 0.
        # for i in self.policy_names:
        #   sum_c += contact_cost[i]
        # print("Calculated: ", sum_c)
        return contact_cost

    def distribute_observations(self, obs_full, prev_actions, central_act):
        """
        Construct dictionary that routes to each policy only the relevant
        local information.
        """
        obs_distributed = {}

        for policy_ind, policy_name in enumerate(self.policy_names[:-1]):

            obs_distributed[policy_name] = obs_full * \
                (central_act[policy_ind*17:(policy_ind+1)*17])

        # central
        obs_distributed[self.policy_names[-1]
                        ] = np.concatenate((obs_full, prev_actions))
        return obs_distributed

    def distribute_reward(self, reward_full, info, action_dict):
        """ Describe how to distribute reward.
        """
        fw_reward = info['reward_run']
        rew = {}
        rew[self.policy_names[-1]] = 0
        contact_costs = self.distribute_contact_cost()
        for policy_name in self.policy_names[:-1]:
            rew[policy_name] = fw_reward / (len(self.policy_names)-1) \
                - self.env.ctrl_cost_weight * np.sum(np.square(action_dict[policy_name])) \
                - contact_costs[policy_name]
            # for central
            rew[self.policy_names[-1]] -= self.env.ctrl_cost_weight * \
                np.sum(np.square(action_dict[policy_name]))
        # central

        rew[self.policy_names[-1]] += fw_reward \
            - contact_costs[self.policy_names[-1]]

        return rew

    def concatenate_actions(self, action_dict):
        # Return actions in the
        actions = np.concatenate((action_dict[self.policy_names[0]
                                              ], action_dict[self.policy_names[1]],
                                  action_dict[self.policy_names[2]
                                              ], action_dict[self.policy_names[3]],
                                  action_dict[self.policy_names[4]], action_dict[self.policy_names[5]]))
        return actions

    @ staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_central"):
            return "policy_central"
        elif agent_id.startswith("policy_bthigh"):
            return "policy_bthigh"
        elif agent_id.startswith("policy_bshin"):
            return "policy_bshin"
        elif agent_id.startswith("policy_bfoot"):
            return "policy_bfoot"
        elif agent_id.startswith("policy_fthigh"):
            return "policy_fthigh"
        elif agent_id.startswith("policy_fshin"):
            return "policy_fshin"
        elif agent_id.startswith("policy_ffoot"):
            return "policy_ffoot"
        else:
            return "policy_ffoot"
