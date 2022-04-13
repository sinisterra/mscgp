# %%
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from semantic_group_discovery import find_clique_cover
from networkx.algorithms import bipartite
from scipy.stats import skew, kurtosis
from tabulate import tabulate

dfch = pd.read_csv("./assoc_full.csv").query("a1 != 'index' and a2 != 'index'")

dfch["key"] = dfch.apply(lambda r: tuple(sorted([r["a1"], r["a2"]])), axis=1)
dfdedup = dfch.drop_duplicates(subset=["key"])
print(f"Total de aristas: {len(dfdedup)}")

dfch2 = dfch.query("significant == True")
distinct_cramer = sorted(list(dfch2["cramer"].unique()))
len(set(dfch["a1"].unique()).union(set(dfch["a2"].unique())))
# %%
adj = pd.pivot_table(dfch2, index="a1", columns="a2", values="cramer").fillna(0)
gr = nx.from_pandas_adjacency(adj)

gr_full = nx.from_pandas_adjacency(
    pd.pivot_table(dfch, index="a1", columns="a2", values="cramer").fillna(0)
)
# %%
len(gr_full.edges), len(gr.edges)
# %%

dsets = []
cut = None
cut_point = None
for c in distinct_cramer:
    without = dfch2.query(f"cramer >= {c}")
    g = nx.Graph()
    for (i, r) in without.iterrows():
        g.add_edge(r["a1"], r["a2"])

    if nx.has_bridges(g):
        print(len(list(nx.connected_components(g))), list(nx.bridges(g)), c)
        cut = g
        cut_point = c
        break
# %%
cut_point
# %%
len(gr.edges)

# %%
len(cut.edges)
# %%
nx.max_weight_matching(gr)
# %%
write_dot(nx.maximum_spanning_tree(cut), "./cutoff_max_st.dot")
# %%
import itertools

between_pairs = {}
total_attributes = {}
sources = {}
targets = {}
cc = find_clique_cover(cut)[0]
ccvs = cc.keys()

for (sg, sg_elems) in cc.items():
    subg = nx.maximum_spanning_tree(cut.subgraph(sg_elems))
    print(sg, subg)

for (as1, as2) in itertools.combinations(ccvs, 2):
    s1 = cc[as1]
    s2 = cc[as2]
    key = (as1, as2)
    # between_pairs[key] = 0
    for (s, t) in itertools.product(s1, s2):
        if cut.has_edge(s, t):
            sources[key] = sources.get(key, set()).union(set([s]))
            targets[key] = targets.get(key, set()).union(set([t]))
            between_pairs[key] = between_pairs.get(key, 0) + 1

sts = {}
for k in [*sources.keys(), *targets.keys()]:
    sk = sources[k]
    tk = targets[k]
    sts[k] = min(len(sk), len(tk))

for (i, ((s, t), v)) in enumerate(sts.items()):
    print(f"| {i+1} | {s} | {t} | {v} |")
# %%
g_spanning = nx.Graph()
for ((s, t), v) in between_pairs.items():

    g_spanning.add_edge(s, t, weight=v, label=v)

write_dot(g_spanning, "./spanning.dot")
# %%
g_max_spanning = nx.maximum_spanning_tree(g_spanning)
write_dot(g_max_spanning, "./spanning_max.dot")
print(g_max_spanning.edges)
# %%
len(g_max_spanning.edges)

# %%
len(g_spanning.edges)

# %%
nx.max_weight_matching(g_spanning)

# %%
for (k, v) in cc.items():
    subg = cut.subgraph(v)
    if len(subg.nodes) > 1:
        print(k, nx.min_edge_cover(subg))

# %%
dfdedup

# %%
print(f"Asimetría: {skew(dfdedup['cramer'])}\n Curtosis: {kurtosis(dfdedup['cramer'])}")


# %%
print(f"Total de aristas eliminadas = {len(dfdedup.query('cramer < 0.08'))}")

6  # %%

# %%
print(
    tabulate(
        pd.read_csv("./sg_groups.csv"),
        headers="keys",
        showindex="never",
        tablefmt="github",
    )
)

# %%
