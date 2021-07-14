import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces
from mujoco_py import functions
import random


class BipedMultiPoliciesEnv(MultiAgentEnv):
    """ RLLib multiagent Environment that encapsulates a biped walker environment.

        This is the parent class for rllib environments in which control can be 
        distributed onto multiple agents.
        One simulation environment is spawned (a Biped) and this wrapper
        class organizes control and sensory signals.

        This parent class realizes still a central approach which means that
        all sensory inputs are routed to the single, central control instance and 
        all of the control signals of that instance are directly send towards the 
        simulation.

        Deriving classes have to overwrite basically four classes when distributing 
        control to different controllers:
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
        - concatenate_actions: how to integrate the control signals from the controllers
    """

    policy_names = ["centr_A_policy"]

    def __init__(self, config):
        if 'contact_cost_weight' in config.keys():
            contact_cost_weight = config['contact_cost_weight']
        else:
            contact_cost_weight = 5e-4

        if 'ctrl_cost_weight' in config.keys():
            ctrl_cost_weight = config['ctrl_cost_weight']
        else:
            ctrl_cost_weight = 0.5

        if 'hf_smoothness' in config.keys():
            hf_smoothness = config['hf_smoothness']
        else:
            hf_smoothness = 1.

        if 'mass_weight' in config.keys():
            mass_weight = config['mass_weight']
        else:
            mass_weight = 1

        self.env = gym.make("Biped-v3",
                            ctrl_cost_weight=ctrl_cost_weight,
                            contact_cost_weight=contact_cost_weight, hf_smoothness=hf_smoothness)

        cheetah_mass = mujoco_py.functions.mj_getTotalmass(self.env.model)
        self.cheetah_init_mass = cheetah_mass
        mujoco_py.functions.mj_setTotalmass(
            self.env.model, mass_weight * cheetah_mass)
        self.mass_weight = mass_weight
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # For curriculum learning: scale smoothness of height field linearly over time
        # Set parameter
        if 'curriculum_learning_hf' in config.keys():
            self.curriculum_learning_hf = config['curriculum_learning_hf']
        else:
            self.curriculum_learning_hf = False
        if 'range_smoothness' in config.keys():
            self.curriculum_initial_smoothness = config['range_smoothness'][0]
            self.current_smoothness = self.curriculum_initial_smoothness
            self.curriculum_target_smoothness = config['range_smoothness'][1]
        if 'range_last_timestep_hf' in config.keys():
            self.curriculum_last_timestep_hf = config['range_last_timestep_hf']

        # For curriculum learning of the mass: scale mass of the robot linearly over time
        # Set parameter
        if 'curriculum_learning_mass' in config.keys():
            self.curriculum_learning_mass = config['curriculum_learning_mass']
        else:
            self.curriculum_learning_mass = False
        if 'range_mass' in config.keys():
            self.curriculum_initial_mass = config['range_mass'][0]
            self.range_mass = config['range_mass']
        if 'range_last_timestep_mass' in config.keys():
            self.curriculum_last_timestep_mass = config['range_last_timestep_mass']

        self._max_episode_steps = 1000

    def update_environment_after_epoch(self, timesteps_total):
        """
            Called after each training epoch.
            Can be used to set a curriculum during learning.
        """
        if self.curriculum_learning_hf:
            if self.curriculum_last_timestep_hf > timesteps_total:
                # Two different variants:
                # First one is simply decreasing smoothness while in curriculum interval.
                #self.current_smoothness = self.curriculum_initial_smoothness - (self.curriculum_initial_smoothness - self.curriculum_target_smoothness) * (timesteps_total/self.curriculum_last_timestep_hf)
                # Second one is selecting randomly a smoothness, chosen from an interval
                # from flat (1.) towards the decreased minimum smoothness
                self.current_smoothness = self.curriculum_initial_smoothness - np.random.rand()*(self.curriculum_initial_smoothness -
                                                                                                 self.curriculum_target_smoothness) * (timesteps_total/self.curriculum_last_timestep_hf)
            else:
                # Two different variants:
                # First one is simply decreasing smoothness while in curriculum interval.
                #self.curriculum_learning_hf = False
                #self.current_smoothness = self.curriculum_target_smoothness
                # Second one is selecting randomly a smoothness, chosen from an interval
                # from flat (1.) towards the decreased minimum smoothness
                self.current_smoothness = self.curriculum_target_smoothness + \
                    np.random.rand()*(self.curriculum_initial_smoothness -
                                      self.curriculum_target_smoothness)

            self.env.set_hf_parameter(self.current_smoothness)

            # First one is simply increasing mass while in curriculum interval.
            # Second one is selecting randomly a mass, chosen from a list
        if self.curriculum_learning_mass:
            if timesteps_total < self.curriculum_last_timestep_mass[0]:
                self.mass_weight = self.range_mass[0]
            elif timesteps_total > self.curriculum_last_timestep_mass[-1]:
                #self.mass_weight = random.choice(self.range_mass)
                self.mass_weight = self.range_mass[-1]
            else:
                for rg in range(1, len(self.curriculum_last_timestep_mass)):
                    if timesteps_total > self.curriculum_last_timestep_mass[rg-1] and timesteps_total < self.curriculum_last_timestep_mass[rg]:
                        self.mass_weight = self.range_mass[rg]
                        break

            print("Training with a mass of: ",
                  self.mass_weight * self.cheetah_init_mass)
            mujoco_py.functions.mj_setTotalmass(
                self.env.model, self.mass_weight * self.cheetah_init_mass)

        self.env.create_new_random_hfield()
        self.env.reset()

    def distribute_observations(self, obs_full):
        """ Distribute observations in the multi agent environment.
        """
        return {
            self.policy_names[0]: obs_full,
        }

    def distribute_contact_cost(self):
        """ Calculate contact costs and describe how to distribute them.
        """
        contact_cost = {}
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_cost[self.policy_names[0]] = self.env.contact_cost_weight * np.sum(
            np.square(contact_forces))
        return contact_cost

    def distribute_reward(self, reward_full, info, action_dict):
        """ Describe how to distribute reward.
        """
        fw_reward = info['reward_run']
        rew = {}
        contact_costs = self.distribute_contact_cost()
        for policy_name in self.policy_names:
            rew[policy_name] = fw_reward / len(self.policy_names) \
                - self.env.ctrl_cost_weight * np.sum(np.square(action_dict[policy_name])) \
                - contact_costs[policy_name]
        return rew

    def concatenate_actions(self, action_dict):
        """ Collect actions from all agents and combine them for the single 
            call of the environment.
        """
        return action_dict[self.policy_names[0]]  # np.concatenate( (action_dict[self.policy_A],

    def reset(self):
        obs_original = self.env.reset()

        # added for printing
        self.acc_forw_rew = 0
        self.acc_ctrl_cost = 0
        self.acc_contact_cost = 0
        self._elapsed_steps = 0
        self.acc_step = 0
        return self.distribute_observations(obs_original)

    def step(self, action_dict):
        # Stepping the environment.

        # Use with mujoco 2.
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)

        # Combine actions from all agents and step environment.
        obs_full, rew_w, done_w, info_w = self.env.step(self.concatenate_actions(
            action_dict))  # self.env.step( np.concatenate( (action_dict[self.policy_A],
        # action_dict[self.policy_B]) ))

        # Distribute observations and rewards to the individual agents.
        obs_dict = self.distribute_observations(obs_full)
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

    def render(self):
        self.env.render()

    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        return BipedMultiPoliciesEnv.policy_names[0]

    @staticmethod
    def return_policies(obs_space):
        # For each agent the policy interface has to be defined.
        policies = {
            BipedMultiPoliciesEnv.policy_names[0]: (None,
                                                    obs_space, spaces.Box(np.array([-1., -1., -1., -1., -1., -1.]), np.array([+1., +1., +1., +1., +1., +1.])), {}),
        }
        return policies
