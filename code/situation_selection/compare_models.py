"""
This file generates random starting states and compares how the LQR and sparsemax models perform on them
"""
import torch
import numpy as np
import pandas as pd
import sys
from bayes_opt import BayesianOptimization
sys.path.append("../main")
from linear_quadratic_regulator import OptimalAgent
from Microworld_experiment import Microworld
from MicroworldMacroAgent import MicroworldMacroAgent
from helper_functions_fitting import human_and_agent_states_to_log_likelihood

A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                  [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]], dtype=torch.float64)
B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]],
                 dtype=torch.float64)
Q = torch.zeros((5, 5), dtype=torch.float64)
Qf = torch.diag(torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float64))
R = torch.diag(torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=torch.float64))
T = 10
exo_cost = 0.01
continuous_attention = True
goal = torch.tensor([[0., 0., 0., 0., 0.], [1, 1, 1, 1, 1]], dtype=torch.float64)
NUM_ITER = 10
exp_range = (0.001, 1.)
vm_range = (0., 10.)
step_size_range = (0., 1.5)

exp_param = 0.04
vm_param = 34


def generate_sparsemax_data(continuous_attention, goal, init_endogenous, attention_cost, step_size,
                            exp_param, vm_param):
    """
    generate the data using the sparsemax model, assuming it is correct
    """
    agent_states = []

    # set up the sparsemax agent
    macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=init_endogenous, subgoal_dimensions=[0, 1, 2, 3, 4],
                                       init_exogenous=torch.tensor([0, 0, 0, 0]), T=T,
                                       final_goal=goal, lr=step_size,
                                       cost=attention_cost, von_mises_parameter=vm_param,
                                       exponential_parameter=exp_param, continuous_attention=continuous_attention,
                                       exo_cost=exo_cost, step_with_model=True, verbose=False)

    # run the agent and save the states
    for t in range(10):
        _, s_next, _ = macro_agent.step(stop_t=1)
        agent_states.append(s_next)

    return agent_states


def generate_nm2_data(init_endogenous, exp_param, vm_param, add_noise):
    """
    generate the data using the sparsemax model, assuming it is correct
    """
    agent_states = []

    env = Microworld(A, B, init=init_endogenous, exponential_parameter=exp_param,
                     von_mises_parameter=vm_param)

    exo = torch.zeros(4, dtype=torch.float64)
    # take steps with the all-zero exogenous inputs
    for t in range(10):
        if add_noise:
            env.step_with_model(exo)
        else:
            env.step(exo)

        s_next = env.endogenous_state
        agent_states.append(s_next)

    return agent_states

def generate_lqr_data(init_endogenous, exp_param, vm_param):
    all_runs = []
    agent_states = []

    # set up the LQR agent and get the optimal sequence of steps
    endogenous = init_endogenous
    optimal_agent = OptimalAgent(A, B, Q, Qf, R, 10, endogenous)
    action_sequence = optimal_agent.get_actions()

    # set up a microworld and take a the optimal sequence of steps
    mw = Microworld(A, B, endogenous, von_mises_parameter=vm_param, exponential_parameter=exp_param)
    for i in range(10):
        mw.step(action_sequence[i])
        agent_states.append(mw.endogenous_state)
    all_runs.append(agent_states)

    return all_runs


def compute_sparsemax_log_likelihood(states, continuous_attention, goal, init_endogenous, clusters):

        def cost_function(ac, ss, exp, vm):
            data_states = [s.numpy() for s in states]
            agent_states = []
            for t in range(10):
                if t == 0:
                    endogenous = init_endogenous
                else:
                    endogenous = states[t-1]

                # set up the sparse-max agent
                macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=endogenous,
                                                   subgoal_dimensions=[0, 1, 2, 3, 4],
                                                   init_exogenous=torch.tensor([0, 0, 0, 0]), T=T,
                                                   final_goal=goal, cost=ac, lr=ss,  von_mises_parameter=vm,
                                                   exponential_parameter=exp,
                                                   continuous_attention=continuous_attention, exo_cost=exo_cost,
                                                   step_with_model=False, verbose=False)

                _, s_next, _ = macro_agent.step(stop_t=1)

                agent_states.append(s_next.numpy())
            return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

        pbounds = {'ac': (0., 30.), 'ss': step_size_range, 'exp': exp_range, 'vm': vm_range}

        optimizer = BayesianOptimization(
            f=cost_function,
            pbounds=pbounds,
            random_state=1,
        )

        for cluster in clusters:
            optimizer.probe(
                params={'ac': cluster[0], 'ss': cluster[1], 'exp': cluster[2], 'vm': cluster[3]},
                lazy=True
            )

        optimizer.maximize(
            init_points=NUM_ITER,
            n_iter=NUM_ITER
        )

        best_llh = optimizer.max['target']

        return best_llh


def compute_nm2_log_likelihood(states, init_endogenous, clusters):

    def cost_function(exp, vm):
        data_states = [s.numpy() for s in states]
        agent_states = []
        for t in range(10):
            if t == 0:
                endogenous = init_endogenous
            else:
                endogenous = states[t-1]

            env = Microworld(A=A, B=B, init=endogenous)

            agent_input = torch.zeros(B.shape[1], dtype=torch.float64)

            env.step(agent_input.unsqueeze(0))

            agent_states.append(env.endogenous_state.numpy()[0])

        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp,
                                                        vm)

    pbounds = {'exp': exp_range, 'vm': vm_range}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=1,
    )

    for cluster in clusters:
        optimizer.probe(
            params={'exp': cluster[1], 'vm': cluster[2]},
            lazy=True
        )

    optimizer.maximize(
        init_points=NUM_ITER,
        n_iter=NUM_ITER
    )

    best_llh = optimizer.max['target']

    return best_llh


def compute_lqr_log_likelihood(states, init_endogenous, exp_param, vm_param):

    def cost_function(exp, vm):
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
        return human_and_agent_states_to_log_likelihood(data_states, agent_states, exp, vm)

    pbounds = {'exp': exp_range, 'vm': vm_range}

    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.probe(
        params={'exp': exp_param, 'vm': vm_param},
        lazy=True
    )

    optimizer.maximize(
        init_points=NUM_ITER,
        n_iter=NUM_ITER
    )

    best_llh = optimizer.max['target']

    return best_llh


def compute_sparsemax_score(continuous_attention, goal, init_endogenous, attention_cost, step_size,
                            exp_param, vm_param, add_noise):

    all_costs = []
    # set up the macro agent
    macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=init_endogenous,
                                       subgoal_dimensions=[0, 1, 2, 3, 4],
                                       init_exogenous=torch.tensor([0, 0, 0, 0]), T=T,
                                       final_goal=goal, cost=attention_cost, lr=step_size, von_mises_parameter=vm_param,
                                       exponential_parameter=exp_param, continuous_attention=continuous_attention,
                                       exo_cost=exo_cost, step_with_model=add_noise,
                                       verbose=False)

    # keep taking steps until reaching the final state
    for i in range(10):
        _, _, _ = macro_agent.step(stop_t=1)

    # compute the cost
    all_exo = macro_agent.agent.all_exogenous
    s_final = macro_agent.env.endogenous_state
    distance_cost = s_final.dot(Qf.mv(s_final))
    exogenous_cost = np.sum([x.dot(R.mv(x)) for x in all_exo])
    all_costs.append(distance_cost + exogenous_cost)

    return distance_cost + exogenous_cost


def compute_lqr_score(init_endogenous, exp_param, vm_param):
    # get the optimal action sequence
    optimal_agent = OptimalAgent(A, B, Q, Qf, R, 10, init_endogenous)
    action_sequence = optimal_agent.get_actions()
    mw = Microworld(A, B, init_endogenous, exp_param, vm_param)
    # apple each action
    for i in range(10):
        mw.step(action_sequence[i])

    endogenous = mw.endogenous_state
    # compute the final cost
    distance_cost = endogenous.dot(Qf.mv(endogenous))
    exogenous_cost = np.sum([x.dot(R.mv(x)) for x in action_sequence])

    return float(distance_cost + exogenous_cost)


n_runs = 10  # number of noisy runs when generating data

# decide on the correct set of parameters for generating the data
cluster_breakdown = [6, 4]
input_folder = '../../data/fitting_results'
output_folder = "../../data/situations"

if __name__ == '__main__':

    df_clusters = pd.read_csv(f'{input_folder}/pilot_clusters.csv')

    clusters = [(row['attention_cost'], row['exp_param'], row['vm_param']) for _, row in df_clusters[3:].iterrows()]

    # generate a random situation from U[-100, 100]^5
    situation = ((torch.rand(5, dtype=torch.float64) - 0.5) * 500).round()

    # compute the cost at the starting situation
    starting_cost = int(situation.dot(Qf.mv(situation)))

    # generate the data from the sparse max agent
    sparse_max_states = []
    for i in range(n_runs):
        if i < cluster_breakdown[0]:
            row_num = 0
        else:
            row_num = 1

        row = df_clusters.iloc[row_num]

        attention_cost = row['attention_cost']
        step_size = row['step_size']
        exp_param = row['exp_param']
        vm_param = row['vm_param']

        sparse_max_states.append(generate_sparsemax_data(continuous_attention, goal, situation, attention_cost,
                                                         step_size, exp_param, vm_param))

    print(len(sparse_max_states))

    # compute the cost of the sparse hill-climbing agent in this situation (with the generating parameters)
    sparse_max_costs = []
    for i in range(2):

        row = df_clusters.iloc[i]

        attention_cost = row['attention_cost']
        step_size = row['step_size']
        exp_param = row['exp_param']
        vm_param = row['vm_param']

        sparse_max_costs.append(compute_sparsemax_score(continuous_attention, goal, situation, attention_cost,
                                                        step_size, exp_param, vm_param, False))
    sparse_max_cost_determ = np.mean(sparse_max_costs)

    # compute the cost with noise (using the generating parameters again)
    sparse_max_costs_noisy_all = []
    for i in range(n_runs):
        if i < cluster_breakdown[0]:
            row_num = 0
        else:
            row_num = 1

        row = df_clusters.iloc[row_num]

        attention_cost = row['attention_cost']
        step_size = row['step_size']
        exp_param = row['exp_param']
        vm_param = row['vm_param']

        sparse_max_costs_noisy_all.append(compute_sparsemax_score(continuous_attention, goal, situation, attention_cost,
                                                                  step_size, exp_param, vm_param, True))

    sparse_max_cost_noisy = np.mean(sparse_max_costs_noisy_all)

    # get the log likelihoods of each of the sparsemax-generated states for each run of the model
    llhs_sparse_max_sparsemax_generated = []
    llhs_optimal_sparsemax_generated = []
    llhs_nm2_sparsemax_generated = []
    for run in sparse_max_states:
        print(f"sparse max run {i}")
        llhs_sparse_max_sparsemax_generated.append(compute_sparsemax_log_likelihood(run, continuous_attention, goal,
                                                                                    init_endogenous=situation,
                                                                                    clusters=clusters))
        llhs_optimal_sparsemax_generated.append(compute_lqr_log_likelihood(run, situation, exp_param, vm_param))
        llhs_nm2_sparsemax_generated.append(compute_nm2_log_likelihood(run, situation, clusters))

    # compute the statse under the LQR model
    lqr_states = generate_lqr_data(situation, exp_param, vm_param)

    lqr_cost = compute_lqr_score(situation, exp_param, vm_param)

    # get the log-likelihoods of the sparse-max generated states under each model
    llhs_sparse_max_lqr_generated = []
    llhs_optimal_lqr_generated = []
    llhs_nm2_lqr_generated = []
    for run in lqr_states:
        print(f"lqr run {i}")
        llhs_sparse_max_lqr_generated.append(compute_sparsemax_log_likelihood(run, continuous_attention, goal,
                                                                              init_endogenous=situation,
                                                                              clusters=clusters))
        llhs_optimal_lqr_generated.append(compute_lqr_log_likelihood(run, situation, exp_param, vm_param))
        llhs_nm2_lqr_generated.append(compute_nm2_log_likelihood(run, situation, clusters))

    # generate the null model 2 scores
    nm2_states = []
    nm2_costs = []
    for i in range(n_runs):
        if i < cluster_breakdown[0]:
            row_num = 0
        else:
            row_num = 1

        row = df_clusters.iloc[row_num]

        attention_cost = row['attention_cost']
        step_size = row['step_size']
        exp_param = row['exp_param']
        vm_param = row['vm_param']

        nm2_states.append(generate_nm2_data(situation, exp_param, vm_param, add_noise=True))
        last_state = nm2_states[-1][-1]
        nm2_costs.append(last_state.dot(last_state))

    nm2_cost_noisy = np.mean(nm2_costs)

    # compute the determinisitic null model 2 score
    last_state_determ = generate_nm2_data(situation, 0, 0, add_noise=False)[-1]
    nm2_cost_determ = last_state_determ.dot(last_state_determ)

    # compute the log likelihoods of the nm2 data under different models
    llhs_sparse_max_nm2_generated = []
    llhs_optimal_nm2_generated = []
    llhs_nm2_nm2_generated = []
    for run in nm2_states:
        print(f"nm2 run {i}")
        llhs_sparse_max_nm2_generated.append(compute_sparsemax_log_likelihood(run, continuous_attention, goal,
                                                                              init_endogenous=situation,
                                                                              clusters=clusters))
        llhs_optimal_nm2_generated.append(compute_lqr_log_likelihood(run, situation, exp_param, vm_param))
        llhs_nm2_nm2_generated.append(compute_nm2_log_likelihood(run, situation, clusters))

    # turn the rows into a dataframe, then save it to a file.
    row = {'situation': str(situation.tolist()),
           'llh_sparsemax_to_lqr': np.mean(llhs_optimal_sparsemax_generated),
           'llh_sparsemax_to_sparsemax': np.mean(llhs_sparse_max_sparsemax_generated),
           'llh_sparsemax_to_nm2': np.mean(llhs_nm2_sparsemax_generated),
           'llh_lqr_to_sparsemax': np.mean(llhs_sparse_max_lqr_generated),
           'llh_lqr_to_lqr': np.mean(llhs_optimal_lqr_generated),
           'llh_lqr_to_nm2': np.mean(llhs_nm2_lqr_generated),
           'llh_nm2_to_sparsemax': np.mean(llhs_sparse_max_nm2_generated),
           'llh_nm2_to_lqr': np.mean(llhs_optimal_nm2_generated),
           'llh_nm2_to_nm2': np.mean(llhs_nm2_nm2_generated),
           'sparsemax_score_deterministic': sparse_max_cost_determ,
           'sparsemax_score_noisy': sparse_max_cost_noisy,
           'lqr_score': lqr_cost,
           'nm2_score_deterministic': nm2_cost_determ,
           'nm2_score_noisy': nm2_cost_noisy,
           'starting_cost': starting_cost}
    df = pd.DataFrame(row, index=[0])
    df.to_csv(f"{output_folder}/situations_with_llhs_{sys.argv[1]}.csv")
