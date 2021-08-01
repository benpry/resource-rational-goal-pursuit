"""
A collection of helper functions to be in computing the model's fit to given data
"""
import numpy as np
import torch
from Microworld_experiment import Microworld
from MicroworldMacroAgent import MicroworldMacroAgent
import ast
import pandas as pd
from mpmath import *
from scipy.special import i0
from linear_quadratic_regulator import OptimalAgent, SparseLQRAgent


def distances_to_log_likelihood(euclidean_distances, sigma_err):
    """
    Convert Euclidean distances from the model to log-likelihoods under the model

    :param euclidean_distances: the distances to convert
    :type euclidean_distances: list
    :param sigma_err: the error parameters
    :type sigma_err: float64
    :return: log likelihood under the model
    :rtype: float64
    """
    N = len(euclidean_distances)
    var = sigma_err ** 2
    summed_errors = np.sum([x ** 2 for x in euclidean_distances])

    ll = -N / 2 * np.log(2 * np.pi * var) - 1 / (2 * var) * summed_errors

    return ll


def log_likelihood_von_mises(angles_data, k):
    """
    Log likelihood of independent von mises distributions

    k: var parameter
    """
    lls = []
    for angles in angles_data:
        N = len(angles)
        lls.append(k * np.sum(np.cos(angles)) - N * (np.log(2 * np.pi) + np.log(i0(k))))
    return np.sum(lls)


def log_likelihood_exponential(lengths, ld):
    """
    Log likelihood of independent exponential distributions

    lengths: difference in length
    ld: rate parameter
    """
    N = len(lengths)

    return N * np.log(ld) - ld * np.sum(lengths)


def human_and_agent_states_to_log_likelihood(human_states, agent_states, ld, k):
    """
    A function to compute the log likelihood of the visited states of humans

    human_states: all states a pp vistited
    agent_states: all states the model visited
    ld: rate parameter of the exponential distribution
    k: var parameter of the von mises distribution
    """
    distances_length, all_angles = [], []
    angles_diff_means = []
    for human_state, agent_state in zip(human_states, agent_states):
        angles_h, length_h = to_spherical(human_state)
        angles_a, length_a = to_spherical(agent_state)
        angles_diff = np.sqrt((angles_h - angles_a) ** 2)
        distances_length.append(np.sqrt((length_h - length_a) ** 2))
        all_angles.append(angles_diff)
        angles_diff_means.append(np.mean(angles_diff))

    ll_angles = log_likelihood_von_mises(all_angles, k)
    ll_distance = log_likelihood_exponential(distances_length, ld)
    ll = ll_distance + ll_angles
    return np.round(ll, 4)


def run_lqr_once(A, B, situation, human_data, exo_cost, log_likelihoods, exp_param, vm_param, n_rounds=10,
                 attention_cost=None):

    if type(situation) != torch.Tensor:
        situation = torch.tensor(situation, dtype=torch.float64)

    init_endogenous = situation
    Q = torch.zeros(A.shape[0], dtype=torch.float64)
    Qf = torch.diag(torch.ones(A.shape[0], dtype=torch.float64))
    R = exo_cost * torch.diag(torch.ones(B.shape[1], dtype=torch.float64))

    agent_states = []
    human_states = []

    for t in range(n_rounds):
        if t > 0:
            init_endogenous = torch.tensor(ast.literal_eval(human_data.iloc[t - 1]['endogenous']), dtype=torch.float64)

        # in case not all timesteps are recorded for a pp
        try:
            next_state_human = ast.literal_eval(human_data.iloc[t]['endogenous'])
            # if we get a bug here we should insert except: break below
        except IndexError:
            break

        # define the appropriate LQR agent
        if attention_cost is None:
            lqr_agent = OptimalAgent(A, B, Q, Qf, R, T=n_rounds - t, init_endogenous=init_endogenous)
        else:
            lqr_agent = SparseLQRAgent(A, B, Q, Qf, R, T=n_rounds - t, init_endogenous=init_endogenous,
                                       attention_cost=attention_cost)
        agent_action = lqr_agent.get_actions()[0]

        # define the microworld and take a step in it
        env = Microworld(A, B, init_endogenous)
        env.step(agent_action)
        s_next = env.endogenous_state

        agent_states.append(s_next.numpy())
        human_states.append(next_state_human)

    log_likelihoods.append(human_and_agent_states_to_log_likelihood(human_states, agent_states, exp_param, vm_param))


def run_agent_once(A, B, goal, step_size, final_goal, human_data, attention_cost, agent_type, exo_cost,
                   log_likelihoods, exp_param, vm_param, continuous_attention, use_exo_cost, use_input_cost,
                   input_cost, decision_type):
    """
    Run the agent of the specified type once, starting from the previous human state on each iteration
    """
    init_endogenous = goal[1]
    init_exogenous = [0., 0., 0., 0.]
    subgoal_dimensions = [0, 1, 2, 3, 4]

    agent_states = []
    human_states = []

    for t in range(20):
        if t > 0:
            init_endogenous = ast.literal_eval(human_data.iloc[t - 1]['endogenous'])
        # in case not all 20 timesteps are recorded for a pp
        try:
            next_state_human = ast.literal_eval(human_data.iloc[t]['endogenous'])
            # if we get a bug here we should insert except: break below
        except IndexError:
            break
        env = MicroworldMacroAgent(A=A, B=B, init_endogenous=init_endogenous, nr_subgoals=0,
                                   subgoal_dimensions=subgoal_dimensions, lr=step_size,
                                   init_exogenous=init_exogenous, T=10, final_goal=final_goal,
                                   cost=attention_cost, clamp=25, agent_class=agent_type,
                                   exponential_parameter=exp_param, von_mises_parameter=vm_param,
                                   step_with_model=False, exo_cost=exo_cost, continuous_attention=continuous_attention,
                                   use_exo_cost=use_exo_cost, use_input_cost=use_input_cost, input_cost=input_cost,
                                   decision_type=decision_type, verbose=False)

        _, s_next, _ = env.step(stop_t=1)
        agent_states.append(s_next.numpy())
        human_states.append(next_state_human)

    log_likelihoods.append(human_and_agent_states_to_log_likelihood(human_states, agent_states, exp_param, vm_param))


def to_spherical(vec):
    """
    Transform a vector of 5 dimensions to spherical coordinates

    vec: numpy array of len 5 reprenting a visited state
    """
    r = np.linalg.norm(vec)
    if np.linalg.norm(np.delete(vec, [0])) == 0.:
        phi_1 = 0.
    else:
        phi_1 = float(acot(vec[0] / np.linalg.norm(np.delete(vec, [0]))))

    if np.linalg.norm(np.delete(vec, [0, 1])) == 0.:
        phi_2 = 0.
    else:
        phi_2 = float(acot(vec[1] / np.linalg.norm(np.delete(vec, [0, 1]))))

    if np.linalg.norm(np.delete(vec, [0, 1, 2])) == 0.:
        phi_3 = 0.
    else:
        phi_3 = float(acot(vec[2] / np.linalg.norm(np.delete(vec, [0, 1, 2]))))

    if vec[4] == 0.:
        phi_4 = 0.
    else:
        phi_4 = 2 * float(acot(vec[3] + np.linalg.norm(np.delete(vec, [0, 1, 2])) / vec[4]))

    return np.array([phi_1, phi_2, phi_3, phi_4]), r


def to_endogenous(r, angles):
    """
    Transform radian,angles to endogenous state

    r: radian of a state
    angles: angles of a state
    """

    x_1 = r * np.cos(angles[0])
    x_2 = r * np.sin(angles[0]) * np.cos(angles[1])
    x_3 = r * np.sin(angles[0]) * np.sin(angles[1]) * np.cos(angles[2])
    x_4 = r * np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.cos(angles[3])
    x_5 = r * np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[3])

    return np.array([x_1, x_2, x_3, x_4, x_5])


def make_individual_cost_function(human_data=None, pp_id=None, goals=None, agent_type='sparse_max',
                                  continuous_attention=False, use_exo_cost=False, exo_cost=0.01, use_input_cost=False,
                                  input_cost=None, decision_type='grad_opt_step_size', distance_metric="euclidean"):
    """
    Returns a cost function for Bayesian optimization that applies to one goal

    human_data : data of all pps
    pp_id : Id of pp currently fitting a model to
    goals: all situations used in experiment#
    weighting: scaler in [0,1], deciding on the weight to put on euclidean distance vs angles
    agent_type: the model we're fitting
    """

    def cost_function_individual(step_size=None, attention_cost=None, input_cost=None, human_data=human_data,
                                 goals=goals, agent_type=agent_type, exp_param=None, vm_param=None):
        """
        The actual cost function

        learning_rate: learning rate of agent
        cost_param: cost param of sparse-max agent
        exponential_param: parameter of exponetial distribution of distances
        von_mises_param: parameter of von mises distribution of angles
        """
        human_data = human_data[human_data['pp_id'] == pp_id]
        goal_id = int(list(set(human_data['condition']))[0])
        log_likelihoods = []
        goal = goals[goal_id]

        final_goal = torch.tensor(goal[0], dtype=torch.float64)

        final_goal[1] = 1 / final_goal[1]
        A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                          [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]], dtype=torch.float64)
        B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]],
                         dtype=torch.float64)

        if agent_type in ('lqr', 'sparse_lqr'):
            run_lqr_once(A, B, goal[1], human_data, exo_cost, log_likelihoods, exp_param, vm_param, n_rounds=10,
                         attention_cost=attention_cost)
        else:
            for i in range(25):

                run_agent_once(A, B, goal, step_size, final_goal, human_data, attention_cost, agent_type,
                               exo_cost, log_likelihoods, exp_param, vm_param, continuous_attention, use_exo_cost,
                               use_input_cost, input_cost, decision_type)

                if agent_type != 'sparse_max_softmax':
                    break

        return np.mean(log_likelihoods)

    return cost_function_individual


def return_run_data(A, B, step_size, attention_cost, clamp, final_goal, init_endogenous,
                    agent_class=None, von_mises_parameter=None, exponential_parameter=None,
                    step_with_model=None, continuous_attention=True, use_exo_cost=False, exo_cost=0.01,
                    final_value=20000):
    """
    Run an agent in the microworld with a given starting location and goal and return all the relevant data
    """
    agent_data = {}

    init_exogenous = [0., 0., 0., 0.]
    subgoal_dimensions = [0, 1, 2, 3, 4]

    env = MicroworldMacroAgent(A=A, B=B, init_endogenous=init_endogenous, nr_subgoals=0,
                               subgoal_dimensions=subgoal_dimensions, lr=step_size, init_exogenous=init_exogenous, T=20,
                               final_goal=final_goal, cost=attention_cost, clamp=clamp, agent_class=agent_class,
                               von_mises_parameter=von_mises_parameter, exponential_parameter=exponential_parameter,
                               step_with_model=step_with_model, continuous_attention=continuous_attention,
                               use_exo_cost=use_exo_cost, exo_cost=exo_cost, verbose=False)

    done = False
    i = 0
    while not done:
        starting_state, s_tc, done = env.step(stop_t=1)
        i += 1

    closeness = env.closeness_all

    agent_data['exogenous_input'] = env.agent.all_exogenous
    agent_data['endogenous'] = [str(x) for x in env.all_endogenous]
    agent_data['closeness'] = [float(x[0]) for x in env.closeness_all]
    agent_data['final_goal_distance'] = env.final_goal_dist_all
    agent_data['final_reached'] = env.final_goal_reached_all

    return agent_data, closeness, np.sum(env.final_goal_reached_all) > 0.


def generate_agent_data(goals, step_size, attention_cost=None, step_with_model=None, agent_type=None,
                        continuous_attention=True, use_exo_cost=False, exo_cost=0.01):
    """
    Run the agent on many situations and return data
    """
    diff = 0
    # go through each of the goals
    for goal_id, goal in enumerate(goals):

        final_goal = torch.tensor(goal[0])
        init_endogenous = goal[1]
        final_goal[1] = 1 / final_goal[1]
        A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                          [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]])
        B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]])
        if goal_id == 0:
            agent_data_all, closeness, final_reached = return_run_data(A, B, step_size, attention_cost, 25, final_goal,
                                                                       init_endogenous, agent_class=agent_type,
                                                                       step_with_model=step_with_model,
                                                                       final_value=20000,
                                                                       continuous_attention=continuous_attention,
                                                                       use_exo_cost=use_exo_cost, exo_cost=exo_cost)
            #print(agent_data_all)
            agent_data_all = pd.DataFrame(agent_data_all)
            agent_data_all['condition'] = [goal_id] * 21
            agent_data_all['difficulty'] = [0] * 21

        else:
            diff += 1
            agent_data, closeness, final_reached = return_run_data(A, B, step_size, attention_cost, 25, final_goal,
                                                                   init_endogenous, agent_class=agent_type,
                                                                   step_with_model=step_with_model,
                                                                   final_value=20000,
                                                                   continuous_attention=continuous_attention,
                                                                   use_exo_cost=use_exo_cost, exo_cost=exo_cost)
            agent_data = pd.DataFrame(agent_data)
            agent_data['condition'] = [goal_id] * 20
            agent_data['difficulty'] = [diff] * 20
            agent_data_all = pd.concat([agent_data_all, agent_data], ignore_index=True)

            if diff == 3:
                diff = 0

    return agent_data_all


def null_model(n, b, endogenous, goal_loc):
    """
    Implementtation of null model 1, which randomly selects the variables to pay attention to, then sets the
    exogenous variables to random amounts in the direction of interest
    """

    B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]])

    target_vars = np.random.choice(5, n, p=[1 / 5] * 5)

    budget = b
    exogenous = [0., 0., 0., 0.]
    for target_var in target_vars:

        if endogenous[target_var] < goal_loc[target_var]:
            exogenous_input = float(np.random.uniform(0, budget, 1))
        else:
            exogenous_input = float(np.random.uniform(-budget, 0, 1))

        possible_inputs = [i for i, j in enumerate(B[target_var]) if j > 0]

        if len(possible_inputs) > 1:
            input_var = np.random.choice(2, 1, p=[1 / 2] * 2)[0]
        else:
            input_var = possible_inputs[0]

        exogenous[input_var] = exogenous_input
    return exogenous


def make_individual_cost_function_null_1(human_data=None, pp_id=None, goals=None):
    """
    Return a cost function that assesses the null model based on its log-likelihood compared to the human data
    """

    def cost_function_null_1(n=None, b=None, exp_param=None, vm_param=None, human_data=human_data,
                             pp_id=pp_id, goals=goals):
        """
        A cost function that assesses the null model based on the participant data passed in
        """
        n = int(np.round(n))
        human_data = human_data[human_data['pp_id'] == pp_id]
        goal_id = int(list(set(human_data['condition']))[0])
        log_likelihoods = []
        goals = goals[goal_id]
        final_goal = torch.tensor(goals[0])
        init_endogenous = goals[1]
        final_goal[1] = 1 / final_goal[1]
        A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                          [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]], dtype=torch.float64)
        B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]],
                         dtype=torch.float64)

        for i in range(10):

            i += 1
            human_states = []
            agent_states = []

            for t in range(20):
                if t > 0:
                    init_endogenous = ast.literal_eval(str(human_data.iloc[t - 1]['endogenous']))

                try:
                    next_state_human = ast.literal_eval(str(human_data.iloc[t]['endogenous']))
                except IndexError:
                    break

                env = Microworld(A=A, B=B, init=init_endogenous, agent="hillclimbing")

                agent_input = torch.tensor(null_model(n, b, init_endogenous, final_goal[0]), dtype=torch.float64)

                env.step(agent_input.unsqueeze(0))

                agent_states.append(env.endogenous_state.numpy()[0])
                human_states.append(next_state_human)

            log_likelihoods.append(human_and_agent_states_to_log_likelihood(human_states, agent_states, exp_param,
                                                                            vm_param))

        return np.mean(log_likelihoods)

    return cost_function_null_1


def make_individual_cost_function_null_2(human_data=None, pp_id=None, goals=None):
    """
    Return a cost function that assesses the null model based on its log-likelihood compared to the human data
    """

    def cost_function_null_2(exp_param=None, vm_param=None, human_data=human_data, pp_id=pp_id, goals=goals):
        """
        A cost function that assesses the null model based on the participant data passed in
        """
        human_data = human_data[human_data['pp_id'] == pp_id]
        goal_id = list(set(human_data['condition']))[0]
        log_likelihoods = []
        goals = goals[goal_id]
        final_goal = torch.tensor(goals[0])
        init_endogenous = goals[1]

        final_goal[1] = 1 / final_goal[1]
        A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                          [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]], dtype=torch.float64)
        B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]],
                         dtype=torch.float64)

        agent_states = []
        human_states = []

        for t in range(20):
            if t > 0:
                init_endogenous = ast.literal_eval(str(human_data.iloc[t - 1]['endogenous']))
            try:
                next_state_human = ast.literal_eval(str(human_data.iloc[t]['endogenous']))
            except IndexError:
                break

            env = Microworld(A=A, B=B, init=init_endogenous, agent="hillclimbing")

            agent_input = torch.zeros(B.shape[1], dtype=torch.float64)

            env.step(agent_input.unsqueeze(0))

            agent_states.append(env.endogenous_state.numpy()[0])
            human_states.append(next_state_human)

        log_likelihoods.append(human_and_agent_states_to_log_likelihood(human_states, agent_states, exp_param,
                                                                        vm_param))

        return np.mean(log_likelihoods)

    return cost_function_null_2


def convert_to_agent_situations(all_conditions):
    """
    Convert read the goals from the conditions file and return them for future use
    """

    data = []
    for k in range(len(all_conditions)):
        full = []
        goal = [ast.literal_eval(all_conditions['location'][k]), ast.literal_eval(all_conditions['scale'][k])]
        full.append(goal)
        full.append(ast.literal_eval(all_conditions['initial_endogenous'][k]))
        data.append(full)
    return data
