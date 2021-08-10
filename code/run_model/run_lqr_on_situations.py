"""
This file runs a specified model on input situations and gets its score
"""
import torch
import numpy as np
import pandas as pd
import pickle
from ast import literal_eval
from collections import defaultdict
import sys
sys.path.append('../main')
from linear_quadratic_regulator import OptimalAgent
from Microworld_experiment import Microworld

# set up the general parameters of the environment
Q = torch.zeros((5, 5), dtype=torch.float64)
Qf = torch.diag(torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float64))
R = torch.diag(torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=torch.float64))
goal = torch.tensor([[0., 0., 0., 0., 0.], [1, 1, 1, 1, 1]], dtype=torch.float64)
A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                  [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]], dtype=torch.float64)
B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]],
                 dtype=torch.float64)
init_exogenous = [0., 0., 0., 0.]
exo_cost = 0.01
T = 10

# define the filepaths for the best-fitting parameters and situations
situations_filepath = '../../data/experimental_data/experiment_conditions.csv'

OUTPUT_FOLDER = "../../data/qualitative_data"

if __name__ == "__main__":

    # read the situations and parameters
    df_condition = pd.read_csv(situations_filepath)
    situations = df_condition['initial_endogenous']

    # initialize lists for performances and exogenous inputs
    all_lqr_exo = []
    all_lqr_performances = []
    lqr_rows = []

    for situation in situations:
        situation = torch.tensor(literal_eval(situation), dtype=torch.float64)

        # configure a microworld and agent
        microworld = Microworld(A=A, B=B, init=situation)
        lqr_agent = OptimalAgent(A, B, Q, Qf, R, T, microworld.endogenous_state)

        # take each action computed by the agent
        all_actions = lqr_agent.get_actions()
        for action in all_actions:
            microworld.step(action)

        # save the exogenous actions and the cost
        all_lqr_exo.extend([x.tolist() for x in all_actions])
        s_final = microworld.endogenous_state
        distance_cost = s_final.dot(Qf.mv(s_final))
        exogenous_cost = np.sum([x.dot(R.mv(x)) for x in all_actions])
        all_lqr_performances.append(np.sqrt(distance_cost + exogenous_cost).item())
        lqr_rows.append({"situation": situation, "lqr_score": np.sqrt(distance_cost + exogenous_cost).item()})

    # save the performance data
    pd.DataFrame(lqr_rows).to_csv(f"{OUTPUT_FOLDER}/lqr_all_situations.csv", index=False)

    # save all the scores and exogenous variables to pickle
    with open(f"{OUTPUT_FOLDER}/all_lqr_scores.p", "wb") as fp:
        pickle.dump(all_lqr_performances, fp)

    with open(f"{OUTPUT_FOLDER}/all_lqr_exo.p", "wb") as fp:
        pickle.dump(all_lqr_exo, fp)
