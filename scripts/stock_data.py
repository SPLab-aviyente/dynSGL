from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np

from sklearn import preprocessing

import project_path
from src import graphlearning

PROJECT_DIR = Path(__file__).parents[1]
INPUT_DIR = Path(PROJECT_DIR, "data", "inputs")
OUTPUT_DIR = Path(PROJECT_DIR, "data", "outputs")

# Read Data
data_file = Path(INPUT_DIR, "stock_data.csv")
metadata_file = Path(INPUT_DIR, "stock_metadata.csv")

data = pd.read_csv(data_file, index_col = 0, low_memory=False, header=[0, 1])
data = data["Adj Close"]
data.index = pd.to_datetime(data.index)

# Calculate returns and split quarterly
day_returns = np.log1p(data.pct_change()).dropna(axis=0, how="all").dropna(axis=1, how="any")
day_returns = day_returns["2020-01-01":"2021-01-01"]

name = "covid"
w = 10
s = 5
day_returns_split = []
col_names = []
for i in range(0, len(day_returns)-w+1, s):
    start = i 
    end = start + w
    curr_returns = day_returns.iloc[start:end, :]
    col_names.append(curr_returns.index[0])
    day_returns_split.append(preprocessing.scale(curr_returns.to_numpy().T, axis=1))

# Learn the dynamic graph
desired_density = 0.1
desired_similarity = 0.9

v, _ = graphlearning.learn_a_dynamic_signed_graph(day_returns_split, 
                                                  desired_density, 
                                                  desired_similarity)

# Save
learned_graph = {
    "Node1": [n1 for n1, _ in combinations(day_returns.columns.to_list(), 2)], 
    "Node2": [n2 for _, n2 in combinations(day_returns.columns.to_list(), 2)]
}
for i in range(len(day_returns_split)):
    learned_graph[col_names[i]] = np.squeeze(v["+"][i] - v["-"][i])

learned_graph = pd.DataFrame(learned_graph)

save_dir = "data/outputs/stock"

Path(save_dir).mkdir(parents=True, exist_ok=True)

save_name = f"{name}_rolling_{w}_{s}_dynSGL_{desired_density:.2f}_{desired_similarity:.2f}.csv"

learned_graph.to_csv(Path(save_dir, save_name))

v = {"+": [], "-": []}
for t in range(len(day_returns_split)):
    if t==0:
        w_pos, w_neg, params_st = graphlearning.learn_a_static_signed_graph(
            day_returns_split[t], desired_density, desired_density
        )
    else:
        w_pos, w_neg, params_st = graphlearning.learn_a_static_signed_graph(
            day_returns_split[t], desired_density, desired_density, 
            alpha_pos=params_st["alpha_pos"],
            alpha_neg = params_st["alpha_neg"]
        )

    v["+"].append(w_pos)
    v["-"].append(w_neg)

# Save
learned_graph = {
    "Node1": [n1 for n1, _ in combinations(day_returns.columns.to_list(), 2)], 
    "Node2": [n2 for _, n2 in combinations(day_returns.columns.to_list(), 2)]
}
for i in range(len(day_returns_split)):
    learned_graph[col_names[i]] = np.squeeze(v["+"][i] - v["-"][i])

learned_graph = pd.DataFrame(learned_graph)

save_dir = "data/outputs/stock"

Path(save_dir).mkdir(parents=True, exist_ok=True)

save_name = f"{name}_rolling_{w}_{s}_SGL_{desired_density:.2f}.csv"

learned_graph.to_csv(Path(save_dir, save_name))