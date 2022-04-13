# %%
import pandas as pd
import networkx as nx
import itertools
from networkx.drawing.nx_pydot import write_dot
from tabulate import tabulate
import seaborn as sns

groups = pd.read_csv("./sg_groups.csv")
groups

groups["Attributes"] = groups["Attributes"].apply(lambda v: eval(v))
# %%
df_assoc = pd.read_csv("./assoc_full.csv").query("cramer >= 0.08")
df_assoc
df_assoc["key"] = df_assoc.apply(lambda r: tuple(sorted((r["a1"], r["a2"]))), axis=1)

attributes_by_group = {}
colors = [
    "#e57373",
    "#7986cb",
    "#fff176",
    "#aed581",
    "#ba68c8",
    "#64b5f6",
    "#ffb300",
    "#4db6ac",
    "#f06292",
    "#9575cd",
    "#ff8a65",
    "#a1887f",
    "#90a4ae",
    "#ffcdd2",
    "#f8bbd0",
    "#e1bee7",
    "#d1c4e9",
    "#c5cae9",
    "#bbdefb",
    "#b3e5fc",
    "#b2ebf2",
    "#b2dfdb",
    "#c8e6c9",
    "#dcedc8",
    "#fff9c4",
    "#ffecb3",
    "#ffe0b2",
    "#ffccbc",
    "#d7ccc8",
    "#f5f5f5",
    "#cfd8dc",
]
g = nx.Graph()
group_for_attribute = {}
for (i, r) in groups.iterrows():
    for a in r["Attributes"]:
        group_for_attribute[a] = r["Group ID"]
        g.add_node(a, color=colors[r["Group ID"]])
group_for_attribute

df_assoc["g_a1"] = df_assoc["a1"].apply(lambda r: group_for_attribute[r])
df_assoc["g_a2"] = df_assoc["a2"].apply(lambda r: group_for_attribute[r])
# %%
for (i, r) in df_assoc.iterrows():
    g.add_edge(
        r["a1"],
        r["a2"],
        weight=r["cramer"],
        penwidth=5,
        style="dashed",
        color="maroon",
    )
write_dot(g, "./cramer_significant.dot")

g_max_st = nx.maximum_spanning_tree(g)
g_max_st_c = nx.maximum_spanning_tree(g)
write_dot(g_max_st, "./cramer_mst.dot")

for (i, r) in groups.iterrows():
    for (a, b) in itertools.combinations(r["Attributes"], 2):
        k = tuple(sorted((a, b)))
        v = df_assoc[df_assoc["key"] == k]["cramer"].values[0]
        g_max_st.add_edge(a, b, penwidth=1 + (v * 4), color="black", style="solid")
    # for a in r["Attributes"]:
    #     g.add_node(a, color=colors[r["Group ID"]])


write_dot(g_max_st, "./cramer.dot")

# %%
keys_in_mst = []
without_dups = df_assoc
for u, v, a in g_max_st_c.edges(data=True):
    k = tuple(sorted([u, v]))
    keys_in_mst.append(k)
    # print(k)
    # df.at[""]
    # df_assoc[df_assoc["key"] == tuple(sorted([u, v]))]["in_mst"] = True


without_dups["in_mst"] = without_dups["key"].apply(lambda r: r in keys_in_mst)

in_mst = without_dups[without_dups["in_mst"]]
d = {}
import numpy as np

# ct = pd.crosstab(in_mst["g_a1"], in_mst["g_a2"]).replace()
# for i in range(0, 7):
#     ct[i] = np.where(ct[i] > 0, 1, 0)

# ct = ct.replace({1: "x", 0: ""})
# print(tabulate(ct, headers="keys", tablefmt="latex"))
# %%
# sns.heatmap(ct, cmap="YlGnBu")
# %%
