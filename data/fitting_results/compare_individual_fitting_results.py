import pandas as pd
from collections import defaultdict

if __name__ == "__main__":

    n_mismatches = 0
    mms_by_agent = defaultdict(int)
    for agent_num in range(7):
        for pp_num in range(111):

            df_1 = pd.read_csv(f"./individual/fitted_model_{agent_num}_pp_nr_{pp_num}.csv")
            df_2 = pd.read_csv(f"./individual_v2/fitted_model_{agent_num}_pp_nr_{pp_num}.csv")

            mm_so_far = False
            for col in df_1.columns:
                if df_1[col][0] != df_2[col][0]:
                    print(agent_num, pp_num)
                    print(f"MISMATCH! {col}: {df_1[col][0]} - {df_2[col][0]}")
                    if not mm_so_far:
                        n_mismatches += 1
                        mms_by_agent[agent_num] += 1
                    mm_so_far = True

    print(f"num mismatches: {n_mismatches}")
    print(mms_by_agent)
