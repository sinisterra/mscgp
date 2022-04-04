# %%
from math import comb
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from semantic_group_discovery import find_clique_cover
from networkx.algorithms import bipartite

dfch = pd.read_csv("./assoc_full.csv")

dfch2 = dfch.query("significant == True and cramer >= 0.078")
distinct_cramer = sorted(list(dfch2["cramer"].unique()))
len(set(dfch["a1"].unique()).union(set(dfch["a2"].unique())))
# %%
adj = pd.pivot_table(dfch2, index="a1", columns="a2", values="cramer").fillna(0)
gr = nx.from_pandas_adjacency(adj)
# %%
def compute_weights(g):
    for n in g.nodes:
        n_edges = g.edges([n])
        weight_sum = 0
        for e in n_edges:
            edge_weight = g[e[0]][e[1]]["weight"]
            weight_sum += edge_weight

        g.nodes[n]["weight"] = int(weight_sum * 1000)

    return g


# %%
gr = compute_weights(gr)
while True:
    (max_clique, _) = nx.max_weight_clique(gr)
    if len(max_clique) == 0:
        break
    print(max_clique)
    gr.remove_nodes_from(max_clique)

# %%
