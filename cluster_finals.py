# %%
from typing import overload
from h11 import Data
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.cluster.vq import vq
import seaborn as sns
import itertools
from networkx.drawing.nx_pydot import write_dot
from semantic_group_discovery import find_clique_cover
from scipy.stats import zscore


df = pd.read_csv("./results/1643756164/finals.csv")

df = df.query("significant == True and tp > 0 and level == 1")

df["selectors"] = df["rule"].apply(lambda v: eval(v))
# df["selectors"] = df["selectors"].apply(
#     lambda s: set([*[t[0] for t in s[0]], *[t[0] for t in s[1]]])
# )
df["selectors"] = df["selectors"].apply(lambda s: set([*s[0], *s[1]]))
df[["selectors"]]


def jaccard_distance(A, B):
    # Find symmetric difference of two sets
    nominator = A.symmetric_difference(B)

    # Find union of two sets
    denominator = A.union(B)

    # Take the ratio of sizes
    distance = len(nominator) / len(denominator)

    return distance


# %%
df = df.query("sg_pair == 0")
acc = []
for (a, b) in itertools.combinations(df.index, 2):
    sels_a = df.loc[a, "selectors"]
    sels_b = df.loc[b, "selectors"]
    r_a = df.loc[a, "repr"]
    r_b = df.loc[b, "repr"]
    overlap = len(sels_a.intersection(sels_b)) / min(len(sels_a), len(sels_b))
    acc.append(
        {
            "a": r_a,
            "b": r_b,
            "overlap": len(sels_a.intersection(sels_b)),
            "preferred": r_a if len(sels_a) >= len(sels_b) else r_b,
        }
    )
df_acc = pd.DataFrame(acc)
df_acc


# %%
df_acc["too_similar"] = df_acc["overlap"] >= 2
df_acc["overlap_zscore"] = zscore(df_acc["overlap"])
df_acc_sim = df_acc[df_acc["too_similar"]]
df_acc_sim
# %%
import networkx as nx

g = nx.Graph()

for (i, r) in df_acc_sim.iterrows():
    g.add_edge(
        r["a"], r["b"], label=r["overlap"], weight=r["overlap"],
    )

write_dot(g, "./similarity.dot")
d = pd.Series(nx.pagerank(g)).sort_values(ascending=False)
sns.displot(d, rug=True)
# %%

# %%
print(len(g.nodes()))

#%%
cc = find_clique_cover(g)[0]
longest_rules = []
shortest_rules = []
for (k, gr) in cc.items():
    elems = [(int(df[df["repr"] == r].head()["length"]), r) for r in gr]
    sel_min = min(elems, key=lambda s: s[0],)
    sel_max = max(elems, key=lambda s: s[0],)

    row_selected = df[df["repr"] == sel_max[1]]
    longest_rules.append(row_selected)
    shortest_rules.append(row_selected)

pd.concat(longest_rules).to_csv("longest_selected.csv", index=False)
pd.concat(shortest_rules).to_csv("shortest_selected.csv", index=False)

# %%
len(cc.keys()) / len(df)
# %%
