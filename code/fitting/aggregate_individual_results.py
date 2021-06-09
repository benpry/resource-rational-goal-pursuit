"""
This code compiles the results from the individual csv files produced by each of the individual fitting instances
"""
import pandas as pd
import os

# specify the input and output folders
input_folder_name = '../../data/fitting_results/individual'
output_folder_name = '../../data/e3_fitting_results'

if __name__ == "__main__":
    agent_names = ['lqr', 'sparse_lqr', 'sparse_max_discrete', 'sparse_max_continuous', 'null_model_1',
                   'null_model_2', 'hill_climbing']
    for agent_type in range(7):
        # get the agent name and initialize an empty dataframe
        agent_name = agent_names[agent_type]
        df = pd.DataFrame()

        # read the file for each participant
        for pp_number in range(111):
            file_path = f'{input_folder_name}/fitted_model_{agent_type}_pp_nr_{pp_number}_weighting_{0.4}.csv'

            if os.path.isfile(file_path):
                df_ind = pd.read_csv(file_path)
                df = df.append(df_ind)
            else:
                print(f"missing: {input_folder_name}/fitted_model_{agent_type}_pp_nr_{pp_number}_weighting_{0.4}.csv")

        df.to_csv(f"{output_folder_name}/fitting_results_{agent_name}.csv")
