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
Q = torch.zeros((5, 5))
Qf = torch.diag(torch.tensor([1., 1., 1., 1., 1.]))
goal = torch.tensor([[0., 0., 0., 0., 0.], [1, 1, 1, 1, 1]])
A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                  [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]])
B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]])
init_exogenous = [0., 0., 0., 0.]
use_exo_cost = True
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

parser = argparse.ArgumentParser()
parser.add_argument("--exo_cost_mult", type=int, default=-1)
parser.add_argument('--no_noise', action="store_true", default=False)
parser.add_argument('--save_qual', action="store_true", default=False)

if __name__ == "__main__":

    # get the exogenous cost and whether to use noise from the arguments
    args = parser.parse_args()

    if args.exo_cost_mult == -1:
        exo_cost = 0.01
    else:
        all_exo_costs = 10 ** (np.linspace(-4, 0, 100))
        exo_cost = all_exo_costs[args.exo_cost_mult]

    R = exo_cost * torch.diag(torch.ones((4,)))
    noise = not args.no_noise
    save_qual = args.save_qual

    # get all the conditions
    df_condition = pd.read_csv(situations_filepath)
    situations = df_condition['initial_endogenous']

    # read the parameters from a file
    df_params = pd.read_csv(params_filepath)

    all_model_exo = defaultdict(list)
    all_model_performance_samples = defaultdict(list)
    df_all_runs = pd.DataFrame()
    model_performances_all_situations = []

    # run on each situation
    for situation in tqdm(situations[:30]):
        situation = torch.tensor(literal_eval(situation))

        # run once per participant, using the best-fitting model and paramters for that participant
        for index, participant in df_params.iterrows():
            agent_type =  participant["agent_type"]
            step_size = float(participant["step_size"])
            attention_cost = float(participant["attention_cost"])

            if agent_type in ("hill_climbing", "sparse_max_discrete", "sparse_max_continuous"):
                continuous_attention = False if agent_type == 'sparse_max_discrete' else True

                for _ in range(n_noisy):
                    # initialize and run the agent
                    macro_agent = MicroworldMacroAgent(A=A, B=B, init_endogenous=situation,
                                                       subgoal_dimensions=[0, 1, 2, 3, 4],
                                                       nr_subgoals=0, init_exogenous=torch.tensor([0., 0., 0., 0.]),
                                                       T=T, final_goal=goal, clamp=25, agent_class=agent_type,
                                                       cost=attention_cost, lr=step_size,
                                                       von_mises_parameter=vm_param,
                                                       exponential_parameter=exp_param,
                                                       continuous_attention=continuous_attention,
                                                       use_exo_cost=use_exo_cost, exo_cost=exo_cost,
                                                       step_with_model=noise, verbose=False)

                    for i in range(10):
                        _, _, _ = macro_agent.step(stop_t=1)

                    # compute the cost and store the exogenous variables
                    all_exo = macro_agent.agent.all_exogenous
                    all_model_exo[agent_type].extend([x.tolist() for x in all_exo])
                    s_final = macro_agent.env.endogenous_state
                    distance_cost = s_final.dot(Qf.mv(s_final))
                    exogenous_cost = np.sum([x.dot(R.mv(x)) for x in all_exo])

                    total_cost = np.sqrt(distance_cost + exogenous_cost).item()
                    all_model_performance_samples[agent_type].append(total_cost)
                    df_all_runs = df_all_runs.append(
                        pd.DataFrame({"situation": str(situation.tolist()), "model": agent_type,
                                      "performance": total_cost}, index=[0]), ignore_index=True)

                    if not noise:
                        break

            elif agent_type == 'sparse_lqr':
                performance_samples = []
                for _ in range(n_noisy):
                    microworld = Microworld(A=A, B=B, init=situation, exponential_parameter=exp_param,
                                            von_mises_parameter=vm_param)

                    all_actions = []
                    for i in range(10):
                        sparse_lqr_agent = SparseLQRAgent(A, B, Q, Qf, R, T-i, microworld.endogenous_state,
                                                          attention_cost=attention_cost*((T-i)/T))
                        opt_sequence = sparse_lqr_agent.get_actions()
                        microworld.step_with_model(opt_sequence[0], noise=noise)
                        all_actions.append(opt_sequence[0])

                    all_model_exo[agent_type].extend([x.tolist() for x in all_actions])

                    # compute the cost and store the exogenous variables
                    all_model_exo[agent_type].extend([x.tolist() for x in all_actions])
                    s_final = microworld.endogenous_state
                    distance_cost = s_final.dot(Qf.mv(s_final))
                    exogenous_cost = np.sum([x.dot(R.mv(x)) for x in all_actions])

                    total_cost = np.sqrt(distance_cost + exogenous_cost).item()
                    all_model_performance_samples[agent_type].append(total_cost)
                    df_all_runs = df_all_runs.append(
                        pd.DataFrame({"situation": str(situation.tolist()), "model": agent_type,
                                      "performance": total_cost}, index=[0]), ignore_index=True)

            elif agent_type == 'null_model_2':
                for _ in range(n_noisy):
                    microworld = Microworld(A=A, B=B, init=situation, exponential_parameter=exp_param,
                                            von_mises_parameter=vm_param)

                    # run null model 2 on the microworld
                    for i in range(10):
                        microworld.step_with_model(torch.zeros(B.shape[1]), noise=noise)

                    # compute the cost and store the exogenous variables
                    s_final = microworld.endogenous_state
                    distance_cost = s_final.dot(Qf.mv(s_final))
                    exogenous_cost = 0.
                    all_model_exo[agent_type].extend([[0., 0., 0., 0.] for _ in range(10)])

                    total_cost = np.sqrt(distance_cost + exogenous_cost).item()
                    all_model_performance_samples[agent_type].append(total_cost)
                    df_all_runs = df_all_runs.append(
                        pd.DataFrame({"situation": str(situation.tolist()), "model": agent_type,
                                      "performance": total_cost}, index=[0]), ignore_index=True)

                    if not noise:
                        break

            elif agent_type == 'null_model_1':
                n = int(participant['n'])
                b = float(participant['b'])

                for _ in range(n_noisy):
                    microworld = Microworld(A=A, B=B, init=situation, exponential_parameter=exp_param,
                                            von_mises_parameter=vm_param)

                    # run null model 1 on the microworld
                    all_exo = []
                    for i in range(10):
                        action = null_model(n, b, microworld.endogenous_state, torch.zeros(A.shape[0]))
                        action = torch.tensor(action)
                        microworld.step_with_model(action, noise=noise)
                        all_exo.append(action)

                    # compute the cost and store the exogenous variables
                    s_final = microworld.endogenous_state
                    distance_cost = s_final.dot(Qf.mv(s_final))
                    exogenous_cost = np.sum([x.dot(R.mv(x)) for x in all_exo])
                    all_model_exo[agent_type].extend(all_exo)

                    total_cost = np.sqrt(distance_cost + exogenous_cost).item()
                    all_model_performance_samples[agent_type].append(total_cost)
                    df_all_runs = df_all_runs.append(
                        pd.DataFrame({"situation": str(situation.tolist()), "model": agent_type,
                                      "performance": total_cost}, index=[0]), ignore_index=True)

                    if not noise:
                        break

            else:
                raise RuntimeError("unrecognized agent type")

    # save the dataframe of all runs
    if save_qual:
        df_all_runs.to_csv(f"{qualitative_output_folder}/all_model_runs_on_situations_canonical.csv")
    else:
        df_all_runs.to_csv(f"{output_folder}/all_model_runs_on_situations_exo={exo_cost}.csv")

    # save the qualitative data if the option to do so was on
    if save_qual:
        if noise:
            with open(f"{qualitative_output_folder}/individual_model_all_scores.p", "wb") as fp:
                pickle.dump(all_model_performance_samples, fp)

            with open(f"{qualitative_output_folder}/all_individual_exogenous_variables.p", "wb") as fp:
                pickle.dump(all_model_exo, fp)
        else:
            with open(f"{qualitative_output_folder}/individual_model_all_scores_no_noise.p", "wb") as fp:
                pickle.dump(all_model_performance_samples, fp)

            with open(f"{qualitative_output_folder}/all_individual_exogenous_variables_no_noise.p", "wb") as fp:
                pickle.dump(all_model_exo, fp)
