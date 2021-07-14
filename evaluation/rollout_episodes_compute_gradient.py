import collections

from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env import MultiAgentEnv

import numpy as np
import mujoco_py
from mujoco_py import functions

import matplotlib.pyplot as plt
from matplotlib import cm

"""
    Running a learned (multiagent) controller,
    for evaluation - numerically compute the importance matrix 
    through collecting the gradient.
    
    This is adapted from rllib's rollout.py
    (github.com/ray/rllib/rollout.py)
"""


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


"""
    Rollout an episode:
    step through an episode, using the 
        - agent = trained policies (is a multiagent consisting of a dict of agents)
        - env = in the given environment
    for num_steps control steps and running num_episodes episodes.
        
    render: shows OpenGL window
"""


def rollout_episodes(env, agent, num_episodes=1, num_steps=1000, render=True, experiment_nr=0):
    multiagent = isinstance(env, MultiAgentEnv)
    if agent.workers.local_worker().multiagent:
        policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]
    policy_map = agent.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    reward_eps = []
    cot_eps = []
    vel_eps = []
    dist_eps = []
    steps_eps = []
    power_total_eps = []

    # For numerical calculation of gradients:
    # for each input dimension set the step size which
    # is based on the standard deviation of the specific input channel.
    step_dim_low = np.zeros((17, 17))
    step_dim_high = np.zeros((17, 17))
    for i in range(0, 17):
        step_dim_low[i, i] = -0.1 * \
            agent.workers.local_worker().get_filters()[
            'central_policy'].rs.std[i]
        step_dim_high[i, i] = 0.1 * \
            agent.workers.local_worker().get_filters()[
            'central_policy'].rs.std[i]
    manual_grads = np.zeros((17, 6))
    manual_grads_abs = np.zeros((17, 6))
    for episodes in range(0, num_episodes):

        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        #    saver.begin_rollout()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        power_total = 0.0
        steps = 0
        done = False
        # env.env.create_new_random_hfield()
        obs = env.reset()
        start_pos = env.env.sim.data.qpos[0]
        while not done and steps < num_steps:

            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)

                        # Variation of inputs by small steps
                        # and observe how the output (for the mean value)
                        # changes for all action channels.
                        variation_obs_high = a_obs.reshape(
                            1, 17) + step_dim_high
                        variation_obs_low = a_obs.reshape(1, 17) + step_dim_low
                        act_low = np.zeros((17, 6))
                        act_high = np.zeros((17, 6))
                        for i in range(0, 17):
                            act_low[i] = agent.compute_action(
                                variation_obs_low[i, :], policy_id=policy_id, explore=False)
                            act_high[i] = agent.compute_action(
                                variation_obs_high[i, :], policy_id=policy_id, explore=False)
                        manual_grads += act_high - act_low
                        manual_grads_abs += np.abs(act_high - act_low)
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict
            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward
            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if render:
                env.render()
            #        saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs
            # Calculated as torque (during last time step - or in this case sum of
            # proportional control signal (clipped to [-1,1], multiplied by 150 to torque)
            # multiplied by joint velocity for each joint.

            current_power = np.sum(
                np.abs(env.env.sim.data.ctrl * env.env.sim.data.qvel[3:]))
            power_total += current_power

        #    saver.end_rollout()
        distance_x = env.env.sim.data.qpos[0] - start_pos
        com_vel = distance_x/steps
        cost_of_transport = (
            power_total/steps) / (mujoco_py.functions.mj_getTotalmass(env.env.model) * com_vel)
        # Weight is 8.78710174560547
        # print(mujoco_py.functions.mj_getTotalmass(env.env.model))
        #print(steps, " - ", power_total, " / ", power_total/steps, "; CoT: ", cost_of_transport)
        cot_eps.append(cost_of_transport)
        reward_eps.append(reward_total)
        vel_eps.append(com_vel)
        dist_eps.append(distance_x)
        steps_eps.append(steps)
        power_total_eps.append(power_total)
        print(episodes, ' - ', reward_total, '; CoT: ',
              cost_of_transport, '; Vel: ', 20*com_vel)
    #print("GRADS: ", manual_grads)
    np.save("grads_" + str(experiment_nr) + ".npy", manual_grads)
    np.save("grads_abs_" + str(experiment_nr) + ".npy", manual_grads_abs)

    return (reward_eps, steps_eps, dist_eps, power_total_eps, vel_eps, cot_eps)


def rollout_episodes_TwoSides(env, agent, num_episodes=1, num_steps=1000, render=True, experiment_nr=0):
    multiagent = isinstance(env, MultiAgentEnv)
    if agent.workers.local_worker().multiagent:
        policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]
    policy_map = agent.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    reward_eps = []
    cot_eps = []
    vel_eps = []
    dist_eps = []
    steps_eps = []
    power_total_eps = []

    # For numerical calculation of gradients:
    # for each input dimension set the step size which
    # is based on the standard deviation of the specific input channel.
    step_dim_low = np.zeros((2, 11, 11))  # 2 for each agent
    step_dim_high = np.zeros((2, 11, 11))
    ind_0 = 0
    ind_1 = 0
    for i in range(0, 17):
        if i in env.obs_indices["policy_BEHIND"]:
            step_dim_low[0, ind_0, ind_0] = -0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_BEHIND'].rs.std[ind_0]
            step_dim_high[0, ind_0, ind_0] = 0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_BEHIND'].rs.std[ind_0]
            ind_0 += 1

        if i in env.obs_indices["policy_FRONT"]:
            step_dim_low[1, ind_1, ind_1] = -0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_FRONT'].rs.std[ind_1]
            step_dim_high[1, ind_1, ind_1] = 0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_FRONT'].rs.std[ind_1]
            ind_1 += 1

    manual_grads = np.zeros((2, 11, 3))
    manual_grads_abs = np.zeros((2, 11, 3))
    for episodes in range(0, num_episodes):

        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        #    saver.begin_rollout()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        power_total = 0.0
        steps = 0
        done = False
        # env.env.create_new_random_hfield()
        obs = env.reset()
        start_pos = env.env.sim.data.qpos[0]
        while not done and steps < num_steps:

            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)

                        # Variation of inputs by small steps
                        # and observe how the output (for the mean value)
                        # changes for all action channels.

                        if agent_id == "policy_BEHIND":
                            variation_obs_high = a_obs.reshape(
                                1, 11) + step_dim_high[0]
                            variation_obs_low = a_obs.reshape(
                                1, 11) + step_dim_low[0]
                            act_low = np.zeros((11, 3))
                            act_high = np.zeros((11, 3))
                            for i in range(0, 11):
                                act_low[i] = agent.compute_action(
                                    variation_obs_low[i, :], policy_id=policy_id, explore=False)
                                act_high[i] = agent.compute_action(
                                    variation_obs_high[i, :], policy_id=policy_id, explore=False)
                            manual_grads[0] += act_high - act_low
                            manual_grads_abs[0] += np.abs(act_high - act_low)
                        else:
                            variation_obs_high = a_obs.reshape(
                                1, 11) + step_dim_high[1]
                            variation_obs_low = a_obs.reshape(
                                1, 11) + step_dim_low[1]
                            act_low = np.zeros((11, 3))
                            act_high = np.zeros((11, 3))
                            for i in range(0, 11):
                                act_low[i] = agent.compute_action(
                                    variation_obs_low[i, :], policy_id=policy_id, explore=False)
                                act_high[i] = agent.compute_action(
                                    variation_obs_high[i, :], policy_id=policy_id, explore=False)
                            manual_grads[1] += act_high - act_low
                            manual_grads_abs[1] += np.abs(act_high - act_low)

                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action

            action = action_dict
            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward
            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if render:
                env.render()
            #        saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs
            # Calculated as torque (during last time step - or in this case sum of
            # proportional control signal (clipped to [-1,1], multiplied by 150 to torque)
            # multiplied by joint velocity for each joint.

            current_power = np.sum(
                np.abs(env.env.sim.data.ctrl * env.env.sim.data.qvel[3:]))
            power_total += current_power

        #    saver.end_rollout()
        distance_x = env.env.sim.data.qpos[0] - start_pos
        com_vel = distance_x/steps
        cost_of_transport = (
            power_total/steps) / (mujoco_py.functions.mj_getTotalmass(env.env.model) * com_vel)
        # Weight is 8.78710174560547
        # print(mujoco_py.functions.mj_getTotalmass(env.env.model))
        #print(steps, " - ", power_total, " / ", power_total/steps, "; CoT: ", cost_of_transport)
        cot_eps.append(cost_of_transport)
        reward_eps.append(reward_total)
        vel_eps.append(com_vel)
        dist_eps.append(distance_x)
        steps_eps.append(steps)
        power_total_eps.append(power_total)
        print(episodes, ' - ', reward_total, '; CoT: ',
              cost_of_transport, '; Vel: ', 20*com_vel)
    #print("GRADS: ", manual_grads)
    np.save("grads_TwoSides.npy", manual_grads)
    np.save("grads_abs_TwoSides.npy", manual_grads_abs)

    return (reward_eps, steps_eps, dist_eps, power_total_eps, vel_eps, cot_eps)


def rollout_episodes_TwoSides_GCN(env, agent, num_episodes=1, num_steps=1000, render=True, experiment_nr=0):
    multiagent = isinstance(env, MultiAgentEnv)
    if agent.workers.local_worker().multiagent:
        policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]
    policy_map = agent.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    reward_eps = []
    cot_eps = []
    vel_eps = []
    dist_eps = []
    steps_eps = []
    power_total_eps = []

    # For numerical calculation of gradients:
    # for each input dimension set the step size which
    # is based on the standard deviation of the specific input channel.
    step_dim_low = np.zeros((2, 14, 14))  # 2 for each agent
    step_dim_high = np.zeros((2, 14, 14))
    ind_0 = 0
    ind_1 = 0

    # obs
    for i in range(0, 17):
        if i in env.obs_indices["policy_BEHIND"]:
            step_dim_low[0, ind_0, ind_0] = -0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_BEHIND'].rs.std[ind_0]
            step_dim_high[0, ind_0, ind_0] = 0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_BEHIND'].rs.std[ind_0]
            ind_0 += 1

        if i in env.obs_indices["policy_FRONT"]:
            step_dim_low[1, ind_1, ind_1] = -0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_FRONT'].rs.std[ind_1]
            step_dim_high[1, ind_1, ind_1] = 0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_FRONT'].rs.std[ind_1]
            ind_1 += 1

    # actions:
    for i in range(0, 6):
        if i in env.act_indices["policy_BEHIND"]:
            step_dim_low[0, ind_0, ind_0] = -0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_BEHIND'].rs.std[ind_0]
            step_dim_high[0, ind_0, ind_0] = 0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_BEHIND'].rs.std[ind_0]
            ind_0 += 1

        if i in env.act_indices["policy_FRONT"]:
            step_dim_low[1, ind_1, ind_1] = -0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_FRONT'].rs.std[ind_1]
            step_dim_high[1, ind_1, ind_1] = 0.1 * \
                agent.workers.local_worker().get_filters()[
                'policy_FRONT'].rs.std[ind_1]
            ind_1 += 1

    manual_grads = np.zeros((2, 14, 3))
    manual_grads_abs = np.zeros((2, 14, 3))
    for episodes in range(0, num_episodes):

        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        #    saver.begin_rollout()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        power_total = 0.0
        steps = 0
        done = False
        # env.env.create_new_random_hfield()
        obs = env.reset()
        start_pos = env.env.sim.data.qpos[0]
        while not done and steps < num_steps:

            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)

                        # Variation of inputs by small steps
                        # and observe how the output (for the mean value)
                        # changes for all action channels.

                        if agent_id == "policy_BEHIND":
                            variation_obs_high = a_obs.reshape(
                                1, 14) + step_dim_high[0]
                            variation_obs_low = a_obs.reshape(
                                1, 14) + step_dim_low[0]
                            act_low = np.zeros((14, 3))
                            act_high = np.zeros((14, 3))
                            for i in range(0, 14):
                                act_low[i] = agent.compute_action(
                                    variation_obs_low[i, :], policy_id=policy_id, explore=False)
                                act_high[i] = agent.compute_action(
                                    variation_obs_high[i, :], policy_id=policy_id, explore=False)
                            manual_grads[0] += act_high - act_low
                            manual_grads_abs[0] += np.abs(act_high - act_low)
                        else:
                            variation_obs_high = a_obs.reshape(
                                1, 14) + step_dim_high[1]
                            variation_obs_low = a_obs.reshape(
                                1, 14) + step_dim_low[1]
                            act_low = np.zeros((14, 3))
                            act_high = np.zeros((14, 3))
                            for i in range(0, 14):
                                act_low[i] = agent.compute_action(
                                    variation_obs_low[i, :], policy_id=policy_id, explore=False)
                                act_high[i] = agent.compute_action(
                                    variation_obs_high[i, :], policy_id=policy_id, explore=False)
                            manual_grads[1] += act_high - act_low
                            manual_grads_abs[1] += np.abs(act_high - act_low)

                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action

            action = action_dict
            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward
            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if render:
                env.render()
            #        saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs
            # Calculated as torque (during last time step - or in this case sum of
            # proportional control signal (clipped to [-1,1], multiplied by 150 to torque)
            # multiplied by joint velocity for each joint.

            current_power = np.sum(
                np.abs(env.env.sim.data.ctrl * env.env.sim.data.qvel[3:]))
            power_total += current_power

        #    saver.end_rollout()
        distance_x = env.env.sim.data.qpos[0] - start_pos
        com_vel = distance_x/steps
        cost_of_transport = (
            power_total/steps) / (mujoco_py.functions.mj_getTotalmass(env.env.model) * com_vel)
        # Weight is 8.78710174560547
        # print(mujoco_py.functions.mj_getTotalmass(env.env.model))
        #print(steps, " - ", power_total, " / ", power_total/steps, "; CoT: ", cost_of_transport)
        cot_eps.append(cost_of_transport)
        reward_eps.append(reward_total)
        vel_eps.append(com_vel)
        dist_eps.append(distance_x)
        steps_eps.append(steps)
        power_total_eps.append(power_total)
        print(episodes, ' - ', reward_total, '; CoT: ',
              cost_of_transport, '; Vel: ', 20*com_vel)
    #print("GRADS: ", manual_grads)
    np.save("grads_TwoSides_GCN.npy", manual_grads)
    np.save("grads_abs_TwoSides_GCN.npy", manual_grads_abs)

    return (reward_eps, steps_eps, dist_eps, power_total_eps, vel_eps, cot_eps)
