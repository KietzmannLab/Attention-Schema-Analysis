import matplotlib.pyplot as plt

# numpy for handeling matrix operations
import pandas as pd

def load(dir):
    df = pd.read_csv(dir)
    return df

checkpoint_dir = "/share/klab/sthorat/lpi_tf_agents/Cookie-Attention-Tracking/src/cookie_attention_tracking/replicated_code/"

file_names = ["results_agent_with_as_no_ar_200units_1_epochs.csv",
                "results_agent_with_as_no_ar_500units_1_epochs.csv",
                "results_agent_with_as200units_1_epochs.csv",
                "results_agent_with_as500units_1_epochs.csv"
                ]

for file_name in file_names:
    df = load(checkpoint_dir + file_name)
    print(file_name)
    for col in df.columns:
        print(df[col])
