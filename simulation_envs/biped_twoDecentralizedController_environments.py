import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from simulation_envs import BipedMultiPoliciesEnv


class Biped_TwoSideControllers_Env(BipedMultiPoliciesEnv):
    """ Derived environment for control of the two-legged agent.
        Uses two different, concurrent control units (policies) 
        each instantiated as a single agent. 

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
    policy_names = ["policy_BEHIND", "policy_FRONT"]

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

        # Each controller only gets information from that body side: Behind
        self.obs_indices["policy_BEHIND"] = [
            0, 1, 8, 9, 10, 2, 3, 4, 11, 12, 13]
        # Each controller only gets information from that body side: Front
        self.obs_indices["policy_FRONT"] = [
            0, 1, 8, 9, 10, 5, 6, 7, 14, 15, 16]

        super().__init__(config)

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
        #print("CONTACT COST")
        #from mujoco_py import functions
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)
        #print("From Ant Env: ", self.env.contact_cost)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.env.contact_cost_weight * \
            np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/2.
        contact_cost[self.policy_names[0]] = global_contact_costs + \
            np.sum(contact_costs[2:5])
        contact_cost[self.policy_names[1]] = global_contact_costs + \
            np.sum(contact_costs[5:8])

        #sum_c = 0.
        # for i in self.policy_names:
        #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost

    def concatenate_actions(self, action_dict):
        # Return actions
        actions = np.concatenate((action_dict[self.policy_names[0]],
                                  action_dict[self.policy_names[1]]))
        return actions

    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_BEHIND"):
            return "policy_BEHIND"
        else:
            return "policy_FRONT"

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (11,), np.float64)
        policies = {
            Biped_TwoSideControllers_Env.policy_names[0]: (None,
                                                           obs_space, spaces.Box(np.array([-1., -1., -1.]), np.array([+1., +1., +1.])), {}),
            Biped_TwoSideControllers_Env.policy_names[1]: (None,
                                                           obs_space, spaces.Box(np.array([-1., -1., -1.]), np.array([+1., +1., +1.])), {})
        }
        return policies


class Biped_TwoSideControllers_AllInfo_Env(BipedMultiPoliciesEnv):
    """ Derived environment for control of the two-legged agent.
        Uses two different, concurrent control units (policies) 
        each instantiated as a single agent. 

        There is one controller for each side of the agent.
        Input scope of each controller: 
        - All information.

        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
        - concatenate_actions: how to integrate the control signals from the controllers
    """
    # This is ordering of the policies as applied here:
    policy_names = ["policy_BEHIND", "policy_FRONT"]

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

        # Each controller only gets information from that body side: Behind
        self.obs_indices["policy_BEHIND"] = range(0, 17)
        # Each controller only gets information from that body side: Front
        self.obs_indices["policy_FRONT"] = range(0, 17)
        super().__init__(config)

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
        #print("CONTACT COST")
        #from mujoco_py import functions
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)
        #print("From Ant Env: ", self.env.contact_cost)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.env.contact_cost_weight * \
            np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/2.
        contact_cost[self.policy_names[0]] = global_contact_costs + \
            np.sum(contact_costs[2:5])
        contact_cost[self.policy_names[1]] = global_contact_costs + \
            np.sum(contact_costs[5:8])
        # print(contact_cost)
        #sum_c = 0.
        # for i in self.policy_names:
        #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost

    def concatenate_actions(self, action_dict):
        # Return actions
        actions = np.concatenate((action_dict[self.policy_names[0]],
                                  action_dict[self.policy_names[1]]))
        return actions

    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_BEHIND"):
            return "policy_BEHIND"
        else:
            return "policy_FRONT"

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (17,), np.float64)
        policies = {
            Biped_TwoSideControllers_AllInfo_Env.policy_names[0]: (None,
                                                                   obs_space, spaces.Box(np.array([-1., -1., -1.]), np.array([+1., +1., +1.])), {}),
            Biped_TwoSideControllers_AllInfo_Env.policy_names[1]: (None,
                                                                   obs_space, spaces.Box(np.array([-1., -1., -1.]), np.array([+1., +1., +1.])), {})
        }
        return policies


class Biped_TwoSideControllers_GCN_Env(BipedMultiPoliciesEnv):
    """ Derived environment for control of the two-legged agent.
        Uses two different, concurrent control units (policies) 
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
    policy_names = ["policy_BEHIND", "policy_FRONT"]

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

        # Each controller only gets information from that body side: Behind
        self.obs_indices["policy_BEHIND"] = [
            0, 1, 8, 9, 10, 2, 3, 4, 11, 12, 13]
        # Each controller only gets information from that body side: Front
        self.obs_indices["policy_FRONT"] = [
            0, 1, 8, 9, 10, 5, 6, 7, 14, 15, 16]

        # Each controller only gets actions from the other body side: Front
        self.act_indices["policy_BEHIND"] = [3, 4, 5]
        # Each controller only gets actions from the body side: Behind
        self.act_indices["policy_FRONT"] = [0, 1, 2]

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

    def distribute_contact_cost(self):
        contact_cost = {}
        #print("CONTACT COST")
        #from mujoco_py import functions
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)
        #print("From Ant Env: ", self.env.contact_cost)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.env.contact_cost_weight * \
            np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/2.
        contact_cost[self.policy_names[0]] = global_contact_costs + \
            np.sum(contact_costs[2:5])
        contact_cost[self.policy_names[1]] = global_contact_costs + \
            np.sum(contact_costs[5:8])
        # print(contact_cost)
        #sum_c = 0.
        # for i in self.policy_names:
        #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost

    def concatenate_actions(self, action_dict):
        # Return actions
        actions = np.concatenate((action_dict[self.policy_names[0]],
                                  action_dict[self.policy_names[1]]))
        return actions

    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_BEHIND"):
            return "policy_BEHIND"
        else:
            return "policy_FRONT"

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (14,), np.float64)
        policies = {
            Biped_TwoSideControllers_GCN_Env.policy_names[0]: (None,
                                                               obs_space, spaces.Box(np.array([-1., -1., -1.]), np.array([+1., +1., +1.])), {}),
            Biped_TwoSideControllers_GCN_Env.policy_names[1]: (None,
                                                               obs_space, spaces.Box(np.array([-1., -1., -1.]), np.array([+1., +1., +1.])), {})
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
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)

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
