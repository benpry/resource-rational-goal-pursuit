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
                  [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]])
B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]])
Q = torch.zeros((5, 5))
Qf = torch.diag(torch.tensor([1., 1., 1., 1., 1.]))
R = torch.diag(torch.tensor([0.01, 0.01, 0.01, 0.01]))
T = 10
agent_class = 'sparse_max'
exo_cost = 0.01
use_exo_cost = True
goal = torch.tensor([[0., 0., 0., 0., 0.], [1, 1, 1, 1, 1]])
clamp = 25
OPT_ITERS = 200
exp_param_default = 5.
vm_param_default = 40
np.random.seed(21)
torch.seed(22)

def generate_sparsemax_data(continuous_attention, goal, init_endogenous, use_exo_cost, attention_cost, step_size,
                            exp_param=exp_param_default, vm_param=vm_param_default, add_noise=False):
    """
    generate the data using the sparsemax model, assuming it is correct
    """
    agent_states = []

    if add_noise:
        macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=init_endogenous, subgoal_dimensions=[0, 1, 2, 3, 4],
                                           nr_subgoals=0, init_exogenous=torch.tensor([0., 0., 0., 0.]), T=T,
                                           final_goal=goal, clamp=clamp, agent_class=agent_class, lr=step_size,
                                           cost=attention_cost, von_mises_parameter=vm_param,
                                           exponential_parameter=exp_param, continuous_attention=continuous_attention,
                                           use_exo_cost=use_exo_cost, exo_cost=exo_cost, step_with_model=True,
                                           verbose=False)
    else:
        macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=init_endogenous, subgoal_dimensions=[0, 1, 2, 3, 4],
                                           nr_subgoals=0, init_exogenous=torch.tensor([0., 0., 0., 0.]), T=T,
                                           final_goal=goal, clamp=clamp, agent_class=agent_class, lr=step_size,
                                           cost=attention_cost, continuous_attention=continuous_attention,
                                           use_exo_cost=use_exo_cost, exo_cost=exo_cost, step_with_model=False,
                                           verbose=False)

    for t in range(10):
        _, s_next, _ = macro_agent.step(stop_t=1)
        agent_states.append(s_next)

    return agent_states


def generate_nm2_data(init_endogenous, exp_param=exp_param_default, vm_param=vm_param_default, add_noise=True):
    """
    generate the data using the sparsemax model, assuming it is correct
    """
    agent_states = []

    env = Microworld(A, B, init=init_endogenous, exponential_parameter=exp_param,
                     von_mises_parameter=vm_param)

    exo = torch.zeros(4)
    for t in range(10):
        if add_noise:
            env.step_with_model(exo)
        else:
            env.step(exo)

        s_next = env.endogenous_state
        agent_states.append(s_next)

    return agent_states


def generate_lqr_data(init_endogenous, exp_param=exp_param_default, vm_param=vm_param_default, add_noise=False):
    """
    generate data from the LQR
    """
    all_runs = []
    agent_states = [init_endogenous]

    endogenous = init_endogenous
    optimal_agent = OptimalAgent(A, B, Q, Qf, R, 10, endogenous)
    action_sequence = optimal_agent.get_actions()

    i = 0
    for state in range(10):
        mw = Microworld(A, B, endogenous, von_mises_parameter=vm_param, exponential_parameter=exp_param)
        mw.step(action_sequence[i])
        agent_states.append(mw.endogenous_state)
    all_runs.append(agent_states)

    return all_runs


def generate_sparse_lqr_data(init_endogenous, attention_cost, exp_param=exp_param_default, vm_param=vm_param_default,
                             add_noise=False):
    """
    generate data from the sparse LQR
    """
    agent_states = []

    state = init_endogenous
    if add_noise:
        for i in range(10):
            mw = Microworld(A, B, state, von_mises_parameter=vm_param, exponential_parameter=exp_param)
            agent = SparseLQRAgent(A, B, Q, Qf, R, T - i, state, attention_cost)
            actions = agent.get_actions()
            mw.step_with_model(actions[0])
            state = mw.endogenous_state
            agent_states.append(state)
    else:
        agent = SparseLQRAgent(A, B, Q, Qf, R, T, state, attention_cost)
        actions = agent.get_actions()
        mw = Microworld(A, B, state)
        for action in actions:
            mw.step(action)
            state = mw.endogenous_state
            agent_states.append(state)

    return agent_states


def compute_sparsemax_log_likelihood(states, continuous_attention, goal, init_endogenous, use_exo_cost, clusters=None):
    """
    Compute the likelihood of the given states under the sparse hill-climbing model
    """
    def cost_function(ac, ss, exp, vm):
        agent_states = []
        for t in range(10):
            if t == 0:
                endogenous = init_endogenous
            else:
                endogenous = states[t-1]

            macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=endogenous,
                                               subgoal_dimensions=[0, 1, 2, 3, 4],
                                               nr_subgoals=0, init_exogenous=torch.tensor([0., 0., 0., 0.]), T=T,
                                               final_goal=goal, clamp=clamp, agent_class=agent_class,
                                               cost=ac, lr=ss,  von_mises_parameter=vm,
                                               exponential_parameter=exp,
                                               continuous_attention=continuous_attention,
                                               use_exo_cost=use_exo_cost, exo_cost=exo_cost, step_with_model=False,
                                               verbose=False)

            _, s_next, _ = macro_agent.step(stop_t=1)

            agent_states.append(s_next.numpy())
        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    data_states = [s.numpy() for s in states]

    pbounds = {'ac': (0., 30.), 'ss': (0., 1.5), 'exp': (0.01, 1.), 'vm': (0., 10.)}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    if clusters is not None:
        for cluster in clusters:
            optimizer.probe(
                params={'ac': cluster[0], 'ss': cluster[1], 'exp': cluster[2], 'vm': cluster[3]},
                lazy=True
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

        n = int(round(n))
        data_states = [s.numpy() for s in states]
        agent_states = []
        for t in range(10):
            if t == 0:
                endogenous = init_endogenous
            else:
                endogenous = states[t-1]

            env = Microworld(A=A, B=B, init=endogenous)
            agent_input = torch.tensor(null_model(n, b, endogenous, torch.tensor([0, 0, 0, 0, 0])))
            env.step(agent_input.unsqueeze(0))
            agent_states.append(env.endogenous_state.numpy()[0])

        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    pbounds = {'n': (1., 5.), 'b': (1., 50.), 'exp': (0.001, 1.), 'vm': (0., 10.)}

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


def compute_nm2_log_likelihood(states, init_endogenous, clusters=None):
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

            env = Microworld(A=A, B=B, init=endogenous, agent="hillclimbing")

            agent_input = torch.zeros(B.shape[1])

            env.step(agent_input.unsqueeze(0))

            agent_states.append(env.endogenous_state.numpy()[0])

        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp,
                                                        vm)

    pbounds = {'exp': (0.01, 1.), 'vm': (0., 10.)}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    if clusters is not None:
        for cluster in clusters:
            optimizer.probe(
                params={'exp': cluster[1], 'vm': cluster[2]},
                lazy=True
            )

    optimizer.maximize(
        init_points=OPT_ITERS,
        n_iter=OPT_ITERS
    )

    best_llh = optimizer.max['target']

    return best_llh


def compute_lqr_log_likelihood(states, init_endogenous, exp_param=None, vm_param=None):
    """
    Compute the log likelihood of the human states under the LQR
    """
    agent_states = []
    for t in range(10):
        if t == 0:
            endogenous = init_endogenous
        else:
            endogenous = states[t - 1]

        optimal_agent = OptimalAgent(A, B, Q, Qf, R, 10, endogenous)
        action_sequence = optimal_agent.get_actions()

        env = Microworld(A=A, B=B, init=endogenous)
        env.step(action_sequence[0])

        next_state = env.endogenous_state
        agent_states.append(next_state.numpy())

    data_states = [s.numpy() for s in states]

    def cost_function(exp, vm):
        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    pbounds = {'exp': (0.01, 1.), 'vm': (0., 10.)}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    if exp_param and vm_param:
        optimizer.probe(
            params={'exp': exp_param, 'vm': vm_param},
            lazy=True
        )

    optimizer.maximize(
        init_points=OPT_ITERS,
        n_iter=OPT_ITERS
    )

    best_llh = optimizer.max['target']

    return best_llh


def compute_sparse_lqr_log_likelihood(states, init_endogenous, exp_param=None, vm_param=None):
    """
    Compute the log likelihood of the human states under the sparse LQR
    """

    def cost_function(ac, exp, vm):
        data_states = [s.numpy() for s in states]
        agent_states = []

        state = init_endogenous
        mw = Microworld(A, B, state)
        agent = SparseLQRAgent(A, B, Q, Qf, R, T, state, ac)
        actions = agent.get_actions()
        for i in range(10):
            mw.step(actions[i])
            state = mw.endogenous_state
            agent_states.append(state.numpy())

        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    pbounds = {'ac': (0., 300.), 'exp': (0.01, 1.), 'vm': (0., 10.)}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    if exp_param and vm_param:
        optimizer.probe(
            params={'exp': exp_param, 'vm': vm_param},
            lazy=True
        )

    optimizer.maximize(
        init_points=OPT_ITERS,
        n_iter=OPT_ITERS
    )

    best_llh = optimizer.max['target']

    return best_llh


def compute_hill_climbing_log_likelihood(states, goal, init_endogenous, clusters=None):
    """
    compute the likelihood of the given states under the hill-climbing model
    """

    def cost_function(ss, exp, vm):
        agent_states = []
        for t in range(10):
            if t == 0:
                endogenous = init_endogenous
            else:
                endogenous = states[t - 1]

            macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=endogenous,
                                               subgoal_dimensions=[0, 1, 2, 3, 4],
                                               nr_subgoals=0, init_exogenous=torch.tensor([0., 0., 0., 0.]), T=T,
                                               final_goal=goal, clamp=clamp, agent_class=agent_class, cost=0,
                                               lr=ss, von_mises_parameter=vm, exponential_parameter=exp,
                                               continuous_attention=True,
                                               use_exo_cost=use_exo_cost, exo_cost=exo_cost, step_with_model=False,
                                               verbose=False)

            _, s_next, _ = macro_agent.step(stop_t=1)

            agent_states.append(s_next.numpy())
        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    data_states = [s.numpy() for s in states]

    pbounds = {'ss': (0., 2.), 'exp': (0.01, 1.), 'vm': (0., 10.)}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=2021,
    )

    if clusters is not None:
        for cluster in clusters:
            optimizer.probe(
                params={'ss': cluster[1], 'exp': cluster[2], 'vm': cluster[3]},
                lazy=True
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

    row_idx = int(sys.argv[1])
    df_params = pd.read_csv(FIT_PARAMS_FILE)
    row = df_params.loc[row_idx]

    # get the starting situation
    df_ppids = pd.read_csv(PPID_FILE)
    ppid = df_ppids.loc[row_idx]["id"]
    df_actions = pd.read_csv(ACTIONS_FILE)
    condition_num = df_actions[df_actions["pp_id"] == ppid]["condition"].min()
    df_conditions = pd.read_csv(CONDITIONS_FILE)
    situation = torch.tensor(literal_eval(df_conditions.loc[condition_num]["initial_endogenous"])).float()

    goal = torch.tensor([[0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.]])

    agent_type = row["agent_type"]

    # generate data from the relevant agent type
    if agent_type in ("sparse_max_continuous", "hill_climbing"):
        attention_cost = row["attention_cost"]
        step_size = row["step_size"]
        continuous_attention = True
        model_data = generate_sparsemax_data(continuous_attention, goal, situation, True, attention_cost, step_size)
    elif agent_type == "sparse_max_discrete":
        attention_cost = row["attention_cost"]
        step_size = row["step_size"]
        continuous_attention = False
        model_data = generate_sparsemax_data(continuous_attention, goal, situation, True, attention_cost, step_size)
    elif agent_type == "sparse_lqr":  # agent_type == "sparse_lqr":
        attention_cost = row["attention_cost"]
        model_data = generate_sparse_lqr_data(situation, attention_cost)

    # compute log likelihoods for each model type
    agent_llhs = dict()
    agent_llhs["sparse_max_continuous"] = compute_sparsemax_log_likelihood(model_data, True, goal, situation, True)
    agent_llhs["sparse_max_discrete"] = compute_sparsemax_log_likelihood(model_data, False, goal, situation, True)
    agent_llhs["sparse_lqr"] = compute_sparse_lqr_log_likelihood(model_data, situation)
    agent_llhs["lqr"] = compute_lqr_log_likelihood(model_data, situation)
    agent_llhs["nm1"] = compute_nm1_log_likelihood(model_data, situation)
    agent_llhs["nm2"] = compute_nm2_log_likelihood(model_data, situation)
    agent_llhs["hc"] = compute_hill_climbing_log_likelihood(model_data, goal, situation)

    df = pd.DataFrame([{"situation": situation.tolist(), "generating_model": agent_type, **agent_llhs}])
    df.to_csv(OUTPUT_FILE.format(row_idx), index=False)
