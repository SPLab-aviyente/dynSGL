import warnings
import copy
import os
import argparse

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn import preprocessing

import project_path
from src import data
from src import graphlearning

warnings.simplefilter(action="ignore", category=FutureWarning)

def calc_recon_error(w_gt, w_hat):
    return np.linalg.norm(w_gt-w_hat)/np.linalg.norm(w_gt)

parser = argparse.ArgumentParser()
parser.add_argument("--graphmodel", dest="graph_model", default=None,
                    help="Graph model to use to generate the data")
parser.add_argument("--nsignals", dest="n_signals", default=10,
                    help="Number of signals in each time window")
args = parser.parse_args()

graph_model = args.graph_model
if graph_model is None:
    graph_model = "er"

graph_param = 0.15 if graph_model == "er" else 8

# Random model to use to generate base graph and its parameter
if graph_model == "er":
    graph_generator = data.graphs.gen_er_graph
elif graph_model == "ba":
    graph_generator = data.graphs.gen_ba_graph

n_nodes = 100 # number of nodes
n_times = 10
n_signals = [int(args.n_signals)]*n_times # number of signals at each time point

# amount of the perturbation across time 
perturbation = 0.1

desired_density = 0.1
desired_similarity = 0.75

n_runs = 20

# output
results = {}
for method in ["SGL", "dynSGL", "dynGL"]:
    results[method] = {
        "Method": [],
        "Graph Parameter": [graph_param]*n_runs,
        "F1": [],
        "ReconError": [],
    }

for r in range(n_runs):

    ##### GENERATE DATA #####
    w_gt = [] # ground-truth graphs
    Xv = []

    # Generate a graph whose perturbation will be used for dynamic graph generation
    G = graph_generator(n_nodes, graph_param, seed=32*r)
    data.graphs.assign_signs(G, 0.5, seed=34*r)

    n_swaps = int(G.number_of_edges()*perturbation)
    for s in range(n_times):

        Gt = copy.deepcopy(G)
        i = 1
        while True:
            data.edge_swap.topo_und_degree_preserved(Gt, n_swap=n_swaps, seed=32*r+s+30*i)
            if nx.is_connected(Gt):
                break
            else:
                i += 1

        # Generate data
        X = data.signed.gen_signals_from_signed_graph(
            Gt, n_signals[s], seed=32*r+73*s, filter="Gaussian", noise_amount=0.1
        )
        preprocessing.scale(X, axis=1, copy=False)
        Xv.append(X)

        w_gt.append(nx.to_numpy_array(Gt, weight="sign")[np.triu_indices(n_nodes, k=1)])

        G = Gt

    if r == 0:
        v, params_dyn = graphlearning.learn_a_dynamic_signed_graph(Xv, desired_density, desired_similarity)
    else: 
        v, _ = graphlearning.learn_a_dynamic_signed_graph(Xv, desired_density, desired_similarity, 
                                                          alpha=params_dyn["alpha"],
                                                          beta=params_dyn["beta"])

    f1 = 0
    recon_error = 0
    for t in range(n_times):

        w_gt_pos = w_gt[t].copy()
        w_gt_pos[w_gt_pos < 0] = 0

        w_gt_neg = w_gt[t].copy()
        w_gt_neg[w_gt_neg > 0] = 0
        w_gt_neg *= -1

        f1 += f1_score(w_gt_pos, np.squeeze(v["+"][t]>0))/2
        f1 += f1_score(w_gt_neg, np.squeeze(v["-"][t]>0))/2
        recon_error += calc_recon_error(n_nodes*w_gt_pos/np.sum(w_gt_pos), np.squeeze(v["+"][t]))/2
        recon_error += calc_recon_error(n_nodes*w_gt_neg/np.sum(w_gt_neg), np.squeeze(v["-"][t]))/2

    f1 /= n_times
    recon_error /= n_times
    results["dynSGL"]["Method"].append("dynSGL")
    results["dynSGL"]["F1"].append(f1)
    results["dynSGL"]["ReconError"].append(recon_error)

    # DYNAMIC UNSIGNED RESULTS
    f1 = 0
    recon_error = 0
    for t in range(n_times):

        w_gt_pos = w_gt[t].copy()
        w_gt_pos[w_gt_pos < 0] = 0

        w_gt_neg = w_gt[t].copy()
        w_gt_neg[w_gt_neg > 0] = 0
        w_gt_neg *= -1

        f1 += f1_score(w_gt_pos, np.squeeze(v["+"][t]>0))/2
        f1 += f1_score(w_gt_neg, np.squeeze(v["+"][t]>0))/2
        recon_error += calc_recon_error(n_nodes*w_gt_pos/np.sum(w_gt_pos), np.squeeze(v["+"][t]))/2
        recon_error += calc_recon_error(n_nodes*w_gt_neg/np.sum(w_gt_neg), np.squeeze(v["+"][t]))/2

    f1 /= n_times
    recon_error /= n_times
    results["dynGL"]["Method"].append("dynGL")
    results["dynGL"]["F1"].append(f1)
    results["dynGL"]["ReconError"].append(recon_error)

    ##### STATIC GRAPH LEARNING #####

    f1 = 0
    recon_error = 0
    for t in range(n_times):
        if r==0 and t==0:
            w_pos, w_neg, params_st = graphlearning.learn_a_static_signed_graph(
                Xv[t], desired_density, desired_density
            )
        else:
            w_pos, w_neg, params = graphlearning.learn_a_static_signed_graph(
                Xv[t], desired_density, desired_density, alpha_pos=params_st["alpha_pos"],
                alpha_neg = params_st["alpha_neg"]
            )

        w_gt_pos = w_gt[t].copy()
        w_gt_pos[w_gt_pos < 0] = 0

        w_gt_neg = w_gt[t].copy()
        w_gt_neg[w_gt_neg > 0] = 0
        w_gt_neg *= -1

        f1 += f1_score(w_gt_pos, np.squeeze(w_pos>0))/2
        f1 += f1_score(w_gt_neg, np.squeeze(w_neg>0))/2
        recon_error += calc_recon_error(n_nodes*w_gt_pos/np.sum(w_gt_pos), np.squeeze(w_pos))/2
        recon_error += calc_recon_error(n_nodes*w_gt_neg/np.sum(w_gt_neg), np.squeeze(w_neg))/2
                        

    f1 /= n_times
    recon_error /= n_times
    results["SGL"]["Method"].append("SGL")
    results["SGL"]["F1"].append(f1)
    results["SGL"]["ReconError"].append(recon_error)

save_dir = "data/outputs/experiment1/{}_graph_param_{:.3f}_nruns_{:d}_nsignals_{:d}".format(
    graph_model, graph_param, n_runs, int(args.n_signals)
)

Path(save_dir).mkdir(parents=True, exist_ok=True)

for method in ["SGL", "dynSGL", "dynGL"]:
    results_df = pd.DataFrame(results[method])
    results_df.rename_axis("Run")

    save_name = "{}.csv".format(method)

    results_df.to_csv(os.path.join(save_dir, save_name))
