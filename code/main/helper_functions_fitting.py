"""
A collection of helper functions to be in computing the model's fit to given data
"""
import numpy as np
import torch
from Microworld_experiment import Microworld
from MicroworldMacroAgent import MicroworldMacroAgent
import ast
from mpmath import *
from scipy.special import i0
from linear_quadratic_regulator import OptimalAgent, SparseLQRAgent


def log_likelihood_von_mises(angles_data, k):
    """
    Log likelihood of independent von mises distributions

    k: concentration parameter
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
    ld: exponential parameter
    """
    N = len(lengths)

    return N * np.log(ld) - ld * np.sum(lengths)


def human_and_agent_states_to_log_likelihood(human_states, agent_states, ld, k):
    """
    A function to compute the log likelihood of the visited states of humans

    human_states: all states a participant vistited
    agent_states: all states the model visited
    ld: rate parameter of the exponential distribution
    k: var parameter of the von mises distribution
    """
    distances_length, all_angles = [], []
    for human_state, agent_state in zip(human_states, agent_states):
        angles_h, length_h = to_spherical(human_state)
        angles_a, length_a = to_spherical(agent_state)
        # compute the differences between angles and lengths
        angles_diff = angles_h - angles_a
        all_angles.append(angles_diff)
        distances_length.append(np.sqrt((length_h - length_a) ** 2))

    ll_angles = log_likelihood_von_mises(all_angles, k)
    ll_distance = log_likelihood_exponential(distances_length, ld)
    ll = ll_distance + ll_angles
    return np.round(ll, 6)  # round a double to 6 decimal places to improve reproducibility


def run_lqr_once(A, B, situation, human_data, exo_cost, exp_param, vm_param, n_rounds=10,
                 attention_cost=None):

    # ensure that the situation is the right datatype
    if type(situation) != torch.Tensor:
        situation = torch.tensor(situation, dtype=torch.float64)

    # set up the cost matrices
    init_endogenous = situation
    Q = torch.zeros(A.shape[0], dtype=torch.float64)
    Qf = torch.diag(torch.ones(A.shape[0], dtype=torch.float64))
    R = exo_cost * torch.diag(torch.ones(B.shape[1], dtype=torch.float64))

    # initialize lists for the human and agent states
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
                                       attention_cost=attention_cost * (n_rounds - t) / n_rounds)
        agent_action = lqr_agent.get_actions()[0]

        # define the microworld and take a step in it
        env = Microworld(A, B, init_endogenous)
        env.step(agent_action)
        s_next = env.endogenous_state

        # append the states
        agent_states.append(s_next.numpy())
        human_states.append(next_state_human)

    return human_and_agent_states_to_log_likelihood(human_states, agent_states, exp_param, vm_param)


def run_agent_once(A, B, goal, step_size, final_goal, human_data, attention_cost, exo_cost,
                   exp_param, vm_param, continuous_attention, n_rounds=10):
    """
    Run the agent of the specified type once, starting from the previous human state on each iteration
    """
    init_endogenous = goal[1]
    init_exogenous = [0., 0., 0., 0.]
    subgoal_dimensions = [0, 1, 2, 3, 4]

    agent_states = []
    human_states = []

    for t in range(n_rounds):
        if t > 0:
            init_endogenous = ast.literal_eval(human_data.iloc[t - 1]['endogenous'])
        # in case not all 10 timesteps are recorded for a pp
        try:
            next_state_human = ast.literal_eval(human_data.iloc[t]['endogenous'])
            # if we get a bug here we should insert except: break below
        except IndexError:
            break
        # define a goal pursuit agent
        agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=init_endogenous, subgoal_dimensions=subgoal_dimensions,
                                     init_exogenous=init_exogenous, T=n_rounds - t, final_goal=final_goal,
                                     step_size=step_size, cost=attention_cost, von_mises_parameter=vm_param,
                                     exponential_parameter=exp_param, step_with_model=False, exo_cost=exo_cost,
                                     continuous_attention=continuous_attention, verbose=False)

        # take a step with the agent
        _, s_next, _ = agent.step(stop_t=1)
        # append the next human and agent states
        agent_states.append(s_next.numpy())
        human_states.append(next_state_human)

    return human_and_agent_states_to_log_likelihood(human_states, agent_states, exp_param, vm_param)


def arccot(x):
    """
    compute arc cotangent of x, adding pi if result is nonpositive.
    """
    if x > 0:
        return acot(x)
    else:
        return np.pi + acot(x)


def to_spherical(vec):
    """
    Transform a vector of 5 dimensions to spherical coordinates

    vec: numpy array of len 5 reprenting a visited state
    """
    r = np.linalg.norm(vec)
    if np.linalg.norm(np.delete(vec, [0])) == 0.:
        if vec[0] >= 0:
            phi_1 = 0.
        else:
            phi_1 = np.pi
    else:
        phi_1 = float(arccot(vec[0] / np.sqrt(np.sum(np.delete(vec, [0]) ** 2))))

    if np.linalg.norm(np.delete(vec, [0, 1])) == 0.:
        if vec[1] >= 0:
            phi_2 = 0.
        else:
            phi_2 = np.pi
    else:
        phi_2 = float(arccot(vec[1] / np.sqrt(np.sum(np.delete(vec, [0, 1]) ** 2))))

    if np.linalg.norm(np.delete(vec, [0, 1, 2])) == 0.:
        if vec[2] >= 0:
            phi_3 = 0.
        else:
            phi_3 = np.pi
    else:
        phi_3 = float(arccot(vec[2] / np.sqrt(np.sum(np.delete(vec, [0, 1, 2]) ** 2))))

    if vec[4] == 0.:
        if vec[3] >= 0.:
            phi_4 = 0.
        else:
            phi_4 = np.pi
    else:
        phi_4 = 2 * float(arccot((vec[3] + np.sqrt(np.sum(np.delete(vec, [0, 1, 2]) ** 2))) / vec[4]))
    return np.array([phi_1, phi_2, phi_3, phi_4]), r


def to_endogenous(r, angles):
    """
    Transform radius,angles to endogenous state

    r: radius of a state
    angles: angles of a state
    """

    x_1 = r * np.cos(angles[0])
    x_2 = r * np.sin(angles[0]) * np.cos(angles[1])
    x_3 = r * np.sin(angles[0]) * np.sin(angles[1]) * np.cos(angles[2])
    x_4 = r * np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.cos(angles[3])
    x_5 = r * np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[3])

    return np.array([x_1, x_2, x_3, x_4, x_5])


def make_individual_cost_function(human_data=None, pp_id=None, goals=None, agent_type=None,
                                  continuous_attention=False, exo_cost=0.01):
    """
    Returns a cost function for Bayesian optimization that applies to one goal

    human_data : data of all pps
    pp_id : id of the participant we are currently fitting a model to
    goals: all situations used in experiment
    agent_type: the model we're fitting
    continuous_attention: whether to use continuous or discrete attention (for sparse hill_climbing)
    exo_cost: the cost associated with exogenous actions in the environment
    """

    def cost_function_individual(step_size=None, attention_cost=None, human_data=human_data,
                                 goals=goals, agent_type=agent_type, exp_param=None, vm_param=None):
        """
        The actual cost function

        step_size: step size of agent
        attention_cost: attention cost for the agent
        exp_param: parameter of exponetial distribution for distances
        vm_param: parameter of von mises distribution for angles
        """
        # filter out all but the data from the participant we're studying
        human_data = human_data[human_data['pp_id'] == pp_id]

        # get the corresponding goal (consists of the starting position and final goal)
        goal_id = int(list(set(human_data['condition']))[0])
        goal = goals[goal_id]

        # set attention cost to 0 for the hill-climbing agent:
        if agent_type == "hill_climbing":
            attention_cost = 0

        # get the goal scale and location
        final_goal = torch.tensor(goal[0], dtype=torch.float64)
        final_goal[1] = 1 / final_goal[1]

        # set up the goal pursuit environment
        A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                          [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]], dtype=torch.float64)
        B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]],
                         dtype=torch.float64)

        # run the agent and get the log-likelihood
        if agent_type in ('lqr', 'sparse_lqr'):
            log_likelihood = run_lqr_once(A, B, goal[1], human_data, exo_cost, exp_param, vm_param, n_rounds=10,
                                          attention_cost=attention_cost)
        else:
            log_likelihood = run_agent_once(A, B, goal, step_size, final_goal, human_data, attention_cost, exo_cost,
                                            exp_param, vm_param, continuous_attention)

        # return the log-likelihood
        return log_likelihood

    return cost_function_individual

def null_model(n, b, endogenous, goal_loc):
    """
    Implementtation of null model 1, which randomly selects the endogenous variables to pay attention to, then sets the
    exogenous variables to random amounts in the direction of interest
    """
    # the exogenous-to-endogenous transition matrix
    B = torch.tensor([[0., 0., 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]])

    # choose n endogenous variables to target
    target_vars = np.random.choice(5, n, p=[1 / 5] * 5)

    # set up the budget and initialize an exogenous variable
    budget = b
    exogenous = [0., 0., 0., 0.]
    for target_var in target_vars:

        # choose the direction and magnitude of the input targeting this variable
        if endogenous[target_var] < goal_loc[target_var]:
            exogenous_input = float(np.random.uniform(0, budget, 1))
        else:
            exogenous_input = float(np.random.uniform(-budget, 0, 1))

        # choose which of the possible input variables to target
        possible_inputs = [i for i, j in enumerate(B[target_var]) if j > 0]
        if len(possible_inputs) > 1:
            input_var = np.random.choice(2, 1, p=[1 / 2] * 2)[0]
        else:
            input_var = possible_inputs[0]

        exogenous[input_var] = exogenous_input  # set the chosen exogenous variable

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
        # round the number of variables targeted to an integer
        n = int(np.round(n))

        # select the right human data and set up the situation
        human_data = human_data[human_data['pp_id'] == pp_id]
        goal_id = int(list(set(human_data['condition']))[0])
        goals = goals[goal_id]
        final_goal = torch.tensor(goals[0])
        final_goal[1] = 1 / final_goal[1]

        # set up the microworld environment
        A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                          [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]], dtype=torch.float64)
        B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]],
                         dtype=torch.float64)

        # Run the model 10 times to average log likelihoods
        log_likelihoods = []
        for i in range(10):

            human_states = []
            agent_states = []

            init_endogenous = goals[1]  # set init_endogenous to the starting value
            # iterate over timesteps in the experiment
            for t in range(10):
                if t > 0:
                    init_endogenous = ast.literal_eval(str(human_data.iloc[t - 1]['endogenous']))
                try:
                    next_state_human = ast.literal_eval(str(human_data.iloc[t]['endogenous']))
                except IndexError:
                    break

                # set up the microworld and get an input from the agent
                env = Microworld(A=A, B=B, init=init_endogenous)
                agent_input = torch.tensor(null_model(n, b, init_endogenous, final_goal[0]), dtype=torch.float64)

                # take a step and store the next states
                env.step(agent_input.unsqueeze(0))
                agent_states.append(env.endogenous_state.numpy()[0])
                human_states.append(next_state_human)

            log_likelihoods.append(human_and_agent_states_to_log_likelihood(human_states, agent_states, exp_param,
                                                                            vm_param))

        # return the mean log-likelihood
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
        # parse the human data and select the right situation
        human_data = human_data[human_data['pp_id'] == pp_id]
        goal_id = list(set(human_data['condition']))[0]
        goals = goals[goal_id]
        final_goal = torch.tensor(goals[0])
        final_goal[1] = 1 / final_goal[1]

        # set up the transition matrices
        A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                          [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]], dtype=torch.float64)
        B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]],
                         dtype=torch.float64)

        agent_states = []
        human_states = []
        init_endogenous = goals[1]
        for t in range(10):
            if t > 0:
                init_endogenous = ast.literal_eval(str(human_data.iloc[t - 1]['endogenous']))
            try:
                next_state_human = ast.literal_eval(str(human_data.iloc[t]['endogenous']))
            except IndexError:
                break

            # set up the environment and take a step in it
            env = Microworld(A=A, B=B, init=init_endogenous)
            agent_input = torch.zeros(B.shape[1], dtype=torch.float64)
            env.step(agent_input.unsqueeze(0))

            # save the human and agent states
            agent_states.append(env.endogenous_state.numpy()[0])
            human_states.append(next_state_human)

        # compute and return the log likelihood
        log_likelihood = human_and_agent_states_to_log_likelihood(human_states, agent_states, exp_param, vm_param)
        return log_likelihood

    return cost_function_null_2


def convert_to_agent_situations(df_conditions):
    """
    Convert read the goals from the conditions file and return them for future use
    """
    data = []
    # iterate over the rows of the condition dataframe
    for k in range(len(df_conditions)):
        full = []
        # get the location and scale of the goal
        goal = [ast.literal_eval(df_conditions['location'][k]), ast.literal_eval(df_conditions['scale'][k])]
        full.append(goal)
        # get the starting state corresponding to the goal
        full.append(ast.literal_eval(df_conditions['initial_endogenous'][k]))
        data.append(full)  # append the full goal to the list of all goals
    return data
