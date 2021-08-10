"""
This file generates random starting states and compares how the LQR and sparsemax models perform on them
"""
import sys
import torch
import numpy as np
import pandas as pd
from ast import literal_eval
from bayes_opt import BayesianOptimization
sys.path.append("../main")
from linear_quadratic_regulator import OptimalAgent, SparseLQRAgent
from Microworld_experiment import Microworld
from MicroworldMacroAgent import MicroworldMacroAgent
from helper_functions_fitting import human_and_agent_states_to_log_likelihood, null_model

# set up the general parameters of the environment
A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                  [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]], dtype=torch.float64)
B = torch.tensor([[0., 0., 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]],
                 dtype=torch.float64)
Q = torch.zeros((5, 5), dtype=torch.float64)
Qf = torch.diag(torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float64))
R = torch.diag(torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=torch.float64))
T = 10
exo_cost = 0.01
goal = torch.tensor([[0., 0., 0., 0., 0.], [1, 1, 1, 1, 1]], dtype=torch.float64)

# exponential and von mises parameters
exp_param_default = 0.1
vm_param_default = 40
exp_range = (0.001, 1.)
vm_range = (0., 10.)
step_size_range = (0., 1.5)

# seed the random number generators
np.random.seed(21)
torch.manual_seed(22)

OPT_ITERS = 200  # number of iterations of Bayesian optimization to do


def generate_sparsemax_data(continuous_attention, goal, init_endogenous, attention_cost, step_size,
                            exp_param=exp_param_default, vm_param=vm_param_default, add_noise=False):
    """
    generate the data using the sparsemax model, assuming it is correct
    """
    agent_states = []

    # set up a sparse max agent
    macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=init_endogenous, subgoal_dimensions=[0, 1, 2, 3, 4],
                                       init_exogenous=torch.tensor([0., 0., 0., 0.], dtype=torch.float64), T=T,
                                       final_goal=goal, lr=step_size, cost=attention_cost,
                                       von_mises_parameter=vm_param, exponential_parameter=exp_param,
                                       continuous_attention=continuous_attention, exo_cost=exo_cost,
                                       step_with_model=add_noise, verbose=False)

    # run the agent for ten iterations
    for t in range(10):
        _, s_next, _ = macro_agent.step(stop_t=1)
        agent_states.append(s_next)

    return agent_states

def generate_sparse_lqr_data(init_endogenous, attention_cost, exp_param=exp_param_default, vm_param=vm_param_default,
                             add_noise=False):
    """
    generate data from the sparse LQR
    """
    agent_states = []

    state = init_endogenous
    for i in range(10):
        # set up the microworld and agent
        mw = Microworld(A, B, state, von_mises_parameter=vm_param, exponential_parameter=exp_param)
        agent = SparseLQRAgent(A, B, Q, Qf, R, T - i, state, attention_cost * (T - i) / T)
        # take a step
        actions = agent.get_actions()
        if add_noise:
            mw.step_with_model(actions[0])
        else:
            mw.step(actions[0])
        # save the state
        state = mw.endogenous_state
        agent_states.append(state)

    return agent_states


def compute_sparsemax_log_likelihood(states, continuous_attention, goal, init_endogenous):
    """
    Compute the likelihood of the given states under the sparse hill-climbing model
    """
    def cost_function(ac, ss, exp, vm):
        data_states = [s.numpy() for s in states]
        agent_states = []
        for t in range(10):
            if t == 0:
                endogenous = init_endogenous
            else:
                endogenous = states[t-1]

            # set up the agent
            macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=endogenous,
                                               subgoal_dimensions=[0, 1, 2, 3, 4],
                                               init_exogenous=torch.tensor([0., 0., 0., 0.], dtype=torch.float64),
                                               T=T-t, final_goal=goal, cost=ac, lr=ss,  von_mises_parameter=vm,
                                               exponential_parameter=exp, continuous_attention=continuous_attention,
                                               exo_cost=exo_cost, step_with_model=False, verbose=False)

            # take a step
            _, s_next, _ = macro_agent.step(stop_t=1)
            agent_states.append(s_next.numpy())

        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    pbounds = {'ac': (0., 30.), 'ss': step_size_range, 'exp': exp_range, 'vm': vm_range}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    optimizer.maximize(
        init_points=OPT_ITERS,
        n_iter=OPT_ITERS
    )

    best_llh = optimizer.max['target']

    return best_llh


def compute_nm1_log_likelihood(states, init_endogenous):
    """
    Compute the log-likelihood of the given states under null model 2
    """
    def cost_function(exp, vm, n, b):

        # round the n parameter
        n = int(np.round(n))
        log_likelihoods = []
        data_states = [s.numpy() for s in states]
        for i in range(10):
            agent_states = []
            for t in range(10):
                if t == 0:
                    endogenous = init_endogenous
                else:
                    endogenous = states[t-1]

                # set up the microworld
                env = Microworld(A=A, B=B, init=endogenous)
                # take a step inthe microworld
                agent_input = torch.tensor(null_model(n, b, endogenous, torch.tensor([0, 0, 0, 0, 0], dtype=torch.float64)),
                                           dtype=torch.float64)
                env.step(agent_input.unsqueeze(0))
                agent_states.append(env.endogenous_state.numpy()[0])
            log_likelihoods.append(human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm))

        return np.mean(log_likelihoods)

    pbounds = {'n': (1., 5.), 'b': (0., 50.), 'exp': exp_range, 'vm': vm_range}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    optimizer.maximize(
        init_points=OPT_ITERS,
        n_iter=OPT_ITERS
    )

    best_llh = optimizer.max['target']

    return best_llh


def compute_nm2_log_likelihood(states, init_endogenous):
    """
    Compute the log-likelihood of the given states under null model 2
    """
    def cost_function(exp, vm):
        data_states = [s.numpy() for s in states]
        agent_states = []
        for t in range(10):
            if t == 0:
                endogenous = init_endogenous
            else:
                endogenous = states[t-1]

            # set up the microworld
            env = Microworld(A=A, B=B, init=endogenous)
            # take a step in the microworld
            agent_input = torch.zeros(B.shape[1], dtype=torch.float64)
            env.step(agent_input.unsqueeze(0))
            agent_states.append(env.endogenous_state.numpy()[0])

        # return the log-likelihood
        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    pbounds = {'exp': exp_range, 'vm': vm_range}
    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    optimizer.maximize(
        init_points=OPT_ITERS,
        n_iter=OPT_ITERS
    )

    best_llh = optimizer.max['target']

    return best_llh


def compute_lqr_log_likelihood(states, init_endogenous):
    """
    Compute the log likelihood of the human states under the LQR
    """
    def cost_function(exp, vm):
        data_states = [s.numpy() for s in states]
        agent_states = []
        for t in range(10):
            if t == 0:
                endogenous = init_endogenous
            else:
                endogenous = states[t - 1]

            # get the LQR action sequence
            optimal_agent = OptimalAgent(A, B, Q, Qf, R, T - t, endogenous)
            action_sequence = optimal_agent.get_actions()
            # take a step in a microworld
            env = Microworld(A=A, B=B, init=endogenous)
            env.step(action_sequence[0])
            # save the next state
            next_state = env.endogenous_state
            agent_states.append(next_state.numpy())

        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    pbounds = {'exp': exp_range, 'vm': vm_range}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    optimizer.maximize(
        init_points=OPT_ITERS,
        n_iter=OPT_ITERS
    )

    best_llh = optimizer.max['target']

    return best_llh


def compute_sparse_lqr_log_likelihood(states, init_endogenous):
    """
    Compute the log likelihood of the human states under the sparse LQR
    """

    def cost_function(ac, exp, vm):
        data_states = [s.numpy() for s in states]
        agent_states = []

        for t in range(10):
            if t == 0:
                endogenous = init_endogenous
            else:
                endogenous = states[t - 1]

            # set up the microworld and agent
            mw = Microworld(A, B, endogenous)
            agent = SparseLQRAgent(A, B, Q, Qf, R, T-t, endogenous, ac * (T-t) / T)
            # take an action in the microworld
            actions = agent.get_actions()
            mw.step(actions[0])
            # save the endogenous state
            agent_states.append(mw.endogenous_state.numpy())

        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    pbounds = {'ac': (0., 300.), 'exp': exp_range, 'vm': vm_range}  # higher attention cost to balance out more rounds

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    optimizer.maximize(
        init_points=OPT_ITERS,
        n_iter=OPT_ITERS
    )

    best_llh = optimizer.max['target']

    return best_llh


def compute_hill_climbing_log_likelihood(states, goal, init_endogenous):
    """
    compute the likelihood of the given states under the hill-climbing model
    """

    def cost_function(ss, exp, vm):
        data_states = [s.numpy() for s in states]
        agent_states = []
        for t in range(10):
            if t == 0:
                endogenous = init_endogenous
            else:
                endogenous = states[t - 1]

            # set up the hill-climbing agent
            macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=endogenous, subgoal_dimensions=[0, 1, 2, 3, 4],
                                               init_exogenous=torch.tensor([0., 0., 0., 0.], dtype=torch.float64), T=T,
                                               final_goal=goal, cost=0, lr=ss, von_mises_parameter=vm,
                                               exponential_parameter=exp, continuous_attention=True, exo_cost=exo_cost,
                                               step_with_model=False, verbose=False)
            # take a step and save the state
            _, s_next, _ = macro_agent.step(stop_t=1)
            agent_states.append(s_next.numpy())

        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    pbounds = {'ss': step_size_range, 'exp': exp_range, 'vm': vm_range}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    optimizer.maximize(
        init_points=OPT_ITERS,
        n_iter=OPT_ITERS
    )

    best_llh = optimizer.max['target']

    return best_llh


# define input and output filepaths
FIT_PARAMS_FILE = "../../data/fitting_results/best_fitting_models.csv"
PPID_FILE = "../../data/experimental_data/experiment_ppids.csv"
ACTIONS_FILE = "../../data/experimental_data/experiment_actions.csv"
CONDITIONS_FILE = "../../data/experimental_data/experiment_conditions.csv"
OUTPUT_FILE = "../../data/model_recovery/recovery_results_{}.csv"

if __name__ == '__main__':

    # read the relevant row from the parameters file
    row_idx = int(sys.argv[1])
    df_params = pd.read_csv(FIT_PARAMS_FILE)
    row = df_params.loc[row_idx]

    # get the starting situation
    df_ppids = pd.read_csv(PPID_FILE)
    ppid = df_ppids.loc[row_idx]["id"]
    df_actions = pd.read_csv(ACTIONS_FILE)
    condition_num = df_actions[df_actions["pp_id"] == ppid]["condition"].min()
    df_conditions = pd.read_csv(CONDITIONS_FILE)
    situation = torch.tensor(literal_eval(df_conditions.loc[condition_num]["initial_endogenous"]), dtype=torch.float64)

    goal = torch.tensor([[0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.]], dtype=torch.float64)
    agent_type = row["agent_type"]

    print(agent_type)
    # generate data from the relevant agent type
    if agent_type in ("sparse_max_continuous", "hill_climbing"):
        attention_cost = row["attention_cost"]
        step_size = row["step_size"]
        continuous_attention = True
        model_data = generate_sparsemax_data(continuous_attention, goal, situation, attention_cost, step_size)
    elif agent_type == "sparse_max_discrete":
        attention_cost = row["attention_cost"]
        step_size = row["step_size"]
        continuous_attention = False
        model_data = generate_sparsemax_data(continuous_attention, goal, situation, attention_cost, step_size)
    elif agent_type == "sparse_lqr":
        attention_cost = row["attention_cost"]
        model_data = generate_sparse_lqr_data(situation, attention_cost)
    else:
        raise RuntimeError(f"Unrecognized agent type: {agent_type}")

    # compute log likelihoods for each model type
    agent_llhs = dict()
    agent_llhs["sparse_max_continuous"] = compute_sparsemax_log_likelihood(model_data, True, goal, situation)
    agent_llhs["sparse_max_discrete"] = compute_sparsemax_log_likelihood(model_data, False, goal, situation)
    agent_llhs["sparse_lqr"] = compute_sparse_lqr_log_likelihood(model_data, situation)
    agent_llhs["lqr"] = compute_lqr_log_likelihood(model_data, situation)
    agent_llhs["nm1"] = compute_nm1_log_likelihood(model_data, situation)
    agent_llhs["nm2"] = compute_nm2_log_likelihood(model_data, situation)
    agent_llhs["hc"] = compute_hill_climbing_log_likelihood(model_data, goal, situation)

    df = pd.DataFrame([{"situation": situation.tolist(), "generating_model": agent_type, **agent_llhs}])
    df.to_csv(OUTPUT_FILE.format(row_idx), index=False)
