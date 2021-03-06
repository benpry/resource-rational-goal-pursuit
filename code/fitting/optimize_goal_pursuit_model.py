"""
This file optimizes the parameters of the goal pursuit model for each human participant using Bayesian optimization.

Note: this is only designed to fit the models to the data from experiments without subgoals.
"""
import sys
import torch
import numpy as np
from collections import defaultdict
import pandas as pd
from bayes_opt import BayesianOptimization

# import helper functions
sys.path.append("../main")
from helper_functions_fitting import make_individual_cost_function, make_individual_cost_function_null_1,\
    make_individual_cost_function_null_2, convert_to_agent_situations

# configure the parameter ranges and environment parameters
exp_range = (0.001, 1.)
vm_range = (0., 10.)
step_size_range = (0., 1.5)
exo_cost = 0.01
np.random.seed(416)
torch.manual_seed(647)
opt_iter = 500  # number of optimization iterations

def do_bayesian_optimization(agent_type, data, pp_id, goals):
    """
    Perform Bayesian optimization using the data from the participant specified by pp_id
    """

    # set up a cost function based on the selected model type
    if agent_type == 'null_model_1':
        cost_function = make_individual_cost_function_null_1(human_data=data, pp_id=pp_id, goals=goals)
        pbounds = {'n': (1., 5.), 'b': (0., 50.), 'exp_param': exp_range, 'vm_param': vm_range}
        n_params = 4
        probe_points = [{'n': 1., 'b': 0., 'exp_param': 0.03, 'vm_param': 2.}]

    elif agent_type == 'null_model_2':
        cost_function = make_individual_cost_function_null_2(human_data=data, pp_id=pp_id, goals=goals)
        pbounds = {'exp_param': exp_range, 'vm_param': vm_range}
        n_params = 2
        probe_points = [{'exp_param': 0.03, 'vm_param': 2.}]

    elif agent_type == 'sparse_max_discrete':
        cost_function = make_individual_cost_function(human_data=completed_data, pp_id=pp_id, goals=goals,
                                                      continuous_attention=False, agent_type=agent_type,
                                                      exo_cost=exo_cost)
        pbounds = {'attention_cost': (0., 30.), 'step_size': step_size_range, 'exp_param': exp_range,
                   'vm_param': vm_range}
        n_params = 4
        probe_points = [{'attention_cost': 30., 'step_size': 0., 'exp_param': 0.03, 'vm_param': 2.}]

    elif agent_type == 'sparse_max_continuous':
        cost_function = make_individual_cost_function(human_data=completed_data, pp_id=pp_id, goals=goals,
                                                      continuous_attention=True, agent_type=agent_type,
                                                      exo_cost=exo_cost)
        pbounds = {'attention_cost': (0., 30.), 'exp_param': exp_range,
                   'step_size': step_size_range, 'vm_param': vm_range}
        n_params = 4
        probe_points = [{'attention_cost': 30., 'step_size': 0., 'exp_param': 0.03, 'vm_param': 2.}]

    elif agent_type == 'hill_climbing':
        cost_function = make_individual_cost_function(human_data=completed_data, pp_id=pp_id, goals=goals,
                                                      continuous_attention=True, agent_type=agent_type,
                                                      exo_cost=exo_cost)
        pbounds = {'step_size': step_size_range, 'exp_param': exp_range, 'vm_param': vm_range}
        n_params = 3
        probe_points = [{'step_size': 0., 'exp_param': 0.03, 'vm_param': 2.}]

    elif agent_type == 'sparse_lqr':
        # Sparse LQR Model
        cost_function = make_individual_cost_function(human_data=completed_data, pp_id=pp_id, goals=goals,
                                                      agent_type=agent_type, exo_cost=exo_cost)
        pbounds = {'exp_param': exp_range, 'vm_param': vm_range, 'attention_cost': (0., 300.)}
        probe_points = [{'attention_cost': 300., 'exp_param': 0.03, 'vm_param': 2.}]
        n_params = 3

    else:
        # LQR Model
        cost_function = make_individual_cost_function(human_data=completed_data, pp_id=pp_id, goals=goals,
                                                      agent_type=agent_type, exo_cost=exo_cost)
        pbounds = {'exp_param': exp_range, 'vm_param': vm_range}
        probe_points = [{'exp_param': 0.03, 'vm_param': 2.}]
        n_params = 2

    # set up Bayesian optimization
    optimizer = BayesianOptimization(
        f=cost_function,
        pbounds=pbounds,
        random_state=1,
    )
    # add the test point(s) to probe
    for point in probe_points:
        optimizer.probe(
            params=point,
            lazy=True
        )
    # maximize the log-likelihood with Bayesian optimization
    optimizer.maximize(
        init_points=opt_iter,
        n_iter=opt_iter
    )

    return (optimizer.max, n_params)


# configure the paths to the input and output files
path_root = "../../data/experimental_data"
data_path = path_root + "/experiment_actions.csv"
conditions_path = path_root + "/experiment_conditions.csv"
ppids_path = path_root + "/experiment_ppids.csv"
output_folder = "../../data/fitting_results/individual"
agent_types = ['lqr', 'sparse_lqr', 'sparse_max_discrete', 'sparse_max_continuous', 'null_model_1',
               'null_model_2', 'hill_climbing']

if __name__ == "__main__":

    # read the data from files
    pp_nrs = pd.read_csv(ppids_path)['id']
    completed_data = pd.read_csv(data_path)
    all_conditions = pd.read_csv(conditions_path)
    goals = convert_to_agent_situations(all_conditions)
    n_pps = len(pp_nrs)

    # select the agent type and participant based on the first argument
    it = int(sys.argv[1])
    at = int(np.floor(it / n_pps))
    select_pp = it % n_pps
    pp_id = pp_nrs[select_pp]
    agent_type = agent_types[at]

    print('AGENT-TYPE', agent_type)
    print('pp_id:', pp_id)
    fitted_agent_pp_data = defaultdict(list)

    pp_data = completed_data[completed_data['pp_id'] == pp_id]
    if len(pp_data) == 0:
        raise RuntimeError(f"unrecognized pp_id: {pp_id}")

    # run the Bayesian optimization
    opt_max, n_params = do_bayesian_optimization(agent_type, completed_data, pp_id, goals)

    # retrieve parameters of the maximum
    for key in opt_max['params']:
        fitted_agent_pp_data[key].append(float(opt_max['params'][key]))

    # get the log likelihood
    log_likelihood = float(opt_max['target'])

    # save the participant id, log-likelihood, AIC, and agent type
    fitted_agent_pp_data['pp_id'].append(pp_id)
    fitted_agent_pp_data['ll'].append(log_likelihood)
    fitted_agent_pp_data['AIC'].append(2 * n_params - 2 * log_likelihood)
    fitted_agent_pp_data['agent_type'].append(agent_type)

    # save the results to CSV
    fitted_agent_pp_data = pd.DataFrame(fitted_agent_pp_data)
    print('writing to {}/fitted_model_{}_pp_nr_{}.csv'.format(output_folder, at, select_pp))
    fitted_agent_pp_data.to_csv('{}/fitted_model_{}_pp_nr_{}.csv'.format(output_folder, at, select_pp))
