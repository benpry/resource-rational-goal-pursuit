"""
This file runs a specified model on input situations and gets its score
"""
import torch
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict
import sys
sys.path.append("../main")
from MicroworldMacroAgent import MicroworldMacroAgent
from linear_quadratic_regulator import SparseLQRAgent
from Microworld_experiment import Microworld
from helper_functions_fitting import null_model

# set up the general parameters of the environment
Q = torch.zeros((5, 5), dtype=torch.float64)
Qf = torch.diag(torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float64))
goal = torch.tensor([[0., 0., 0., 0., 0.], [1, 1, 1, 1, 1]], dtype=torch.float64)
A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                  [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]], dtype=torch.float64)
B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]],
                 dtype=torch.float64)
init_exogenous = torch.tensor([0., 0., 0., 0.], dtype=torch.float64)
T = 10

np.random.seed(500)
torch.manual_seed(1000)

# initialize filepaths and hyperparameters
params_filepath = '../../data/fitting_results/best_fitting_models.csv'
situations_filepath = '../../data/experimental_data/experiment_conditions.csv'
exp_param = 0.1
vm_param = 40
n_noisy = 10
output_folder = "../../data/input_cost_analysis"
qualitative_output_folder = "../../data/qualitative_data"

# add the arguments to the parser
parser = argparse.ArgumentParser()
parser.add_argument("--exo_cost_mult", type=int, default=-1)
parser.add_argument('--no_noise', action="store_true", default=False)
parser.add_argument('--save_qual', action="store_true", default=False)

if __name__ == "__main__":

    # get the exogenous cost and whether to use noise from the arguments
    args = parser.parse_args()

    # select the right exogenous cost
    if args.exo_cost_mult == -1:
        exo_cost = 0.01
    else:
        all_exo_costs = 10 ** (np.linspace(-4, 0, 100))
        exo_cost = all_exo_costs[args.exo_cost_mult]

    R = exo_cost * torch.diag(torch.ones((4,), dtype=torch.float64))
    noise = not args.no_noise
    save_qual = args.save_qual

    # get all the situations
    df_condition = pd.read_csv(situations_filepath)
    situations = df_condition['initial_endogenous']

    # read the best-fitting models and parameters for each participant
    df_params = pd.read_csv(params_filepath)

    # initialize dictionaries to store all the exogenous inputs and performance samples
    all_model_exo = defaultdict(list)
    all_model_performance_samples = defaultdict(list)
    df_all_runs = pd.DataFrame()

    # run on each situation
    for situation in tqdm(situations[:30]):
        # convert the situation to tensor format
        situation = torch.tensor(literal_eval(situation), dtype=torch.float64)

        # run for each participant, using the best-fitting model and parameters for that participant
        for index, participant in df_params.iterrows():
            agent_type = participant["agent_type"]
            step_size = float(participant["step_size"])
            attention_cost = float(participant["attention_cost"])

            for _ in range(n_noisy):
                if agent_type in ("hill_climbing", "sparse_max_discrete", "sparse_max_continuous"):
                    continuous_attention = False if agent_type == 'sparse_max_discrete' else True
                    # initialize and run the agent
                    macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=situation,
                                                       subgoal_dimensions=[0, 1, 2, 3, 4],
                                                       init_exogenous=init_exogenous, T=T, final_goal=goal,
                                                       cost=attention_cost, lr=step_size,
                                                       von_mises_parameter=vm_param,
                                                       exponential_parameter=exp_param,
                                                       continuous_attention=continuous_attention, exo_cost=exo_cost,
                                                       step_with_model=noise, verbose=False)

                    for i in range(10):
                        _, _, _ = macro_agent.step(stop_t=1)

                    # compute the cost and store the exogenous variables
                    all_exo = macro_agent.agent.all_exogenous
                    s_final = macro_agent.env.endogenous_state

                elif agent_type == 'sparse_lqr':
                    performance_samples = []
                    microworld = Microworld(A=A, B=B, init=situation, exponential_parameter=exp_param,
                                            von_mises_parameter=vm_param)

                    all_exo = []
                    for i in range(10):
                        sparse_lqr_agent = SparseLQRAgent(A, B, Q, Qf, R, T-i, microworld.endogenous_state,
                                                          attention_cost=attention_cost*((T-i)/T))
                        opt_sequence = sparse_lqr_agent.get_actions()
                        microworld.step_with_model(opt_sequence[0], noise=noise)
                        all_exo.append(opt_sequence[0])

                    # compute the cost and store the exogenous variables
                    s_final = microworld.endogenous_state

                elif agent_type == 'null_model_2':
                    microworld = Microworld(A=A, B=B, init=situation, exponential_parameter=exp_param,
                                            von_mises_parameter=vm_param)

                    # run null model 2 on the microworld
                    for i in range(10):
                        microworld.step_with_model(torch.zeros(B.shape[1], dtype=torch.float64), noise=noise)

                    # compute the cost and store the exogenous variables
                    s_final = microworld.endogenous_state
                    all_exo = [torch.tensor([0., 0., 0., 0.], dtype=torch.float64) for _ in range(10)]

                elif agent_type == 'null_model_1':
                    n = int(np.round(participant['n']))
                    b = float(participant['b'])

                    microworld = Microworld(A=A, B=B, init=situation, exponential_parameter=exp_param,
                                            von_mises_parameter=vm_param)

                    # run null model 1 on the microworld
                    all_exo = []
                    for i in range(10):
                        action = null_model(n, b, microworld.endogenous_state, torch.zeros(A.shape[0],
                                                                                           dtype=torch.float64))
                        action = torch.tensor(action, dtype=torch.float64)
                        microworld.step_with_model(action, noise=noise)
                        all_exo.append(action)

                    # compute the cost and store the exogenous variables
                    s_final = microworld.endogenous_state

                elif agent_type == "lqr":  # we run the lqr on its own in a separate file
                    continue

                else:
                    raise RuntimeError("unrecognized agent type")

                # compute the cost attained in this run
                distance_cost = s_final.dot(Qf.mv(s_final))
                exogenous_cost = np.sum([x.dot(R.mv(x)) for x in all_exo])
                total_cost = np.sqrt(distance_cost + exogenous_cost).item()

                # save the cost and exogenous variables
                all_model_performance_samples[agent_type].append(total_cost)
                all_model_exo[agent_type].extend(all_exo)
                df_all_runs = df_all_runs.append(
                    pd.DataFrame({"situation": str(situation.tolist()), "model": agent_type,
                                  "performance": total_cost}, index=[0]), ignore_index=True)

                if not noise:
                    break

    # save the dataframe of all runs
    if save_qual:
        df_all_runs.to_csv(f"{qualitative_output_folder}/all_model_runs_on_situations_canonical.csv")
    else:
        df_all_runs.to_csv(f"{output_folder}/all_model_runs_on_situations_exo={exo_cost}.csv")

    # save the qualitative data if the option to do so was on
    if save_qual:
        if noise:
            with open(f"{qualitative_output_folder}/all_scores_by_model.p", "wb") as fp:
                pickle.dump(all_model_performance_samples, fp)

            with open(f"{qualitative_output_folder}/all_exo_by_model.p", "wb") as fp:
                pickle.dump(all_model_exo, fp)
        else:
            with open(f"{qualitative_output_folder}/all_scores_by_model_no_noise.p", "wb") as fp:
                pickle.dump(all_model_performance_samples, fp)

            with open(f"{qualitative_output_folder}/all_exo_by_model_no_noise.p", "wb") as fp:
                pickle.dump(all_model_exo, fp)
