# %%
import pandas as pd
import networkx as nx
import itertools
from networkx.drawing.nx_pydot import write_dot

groups = pd.read_csv("./sg_groups.csv")
groups

groups["Attributes"] = groups["Attributes"].apply(lambda v: eval(v))
# %%
df_assoc = pd.read_csv("./assoc_full.csv")
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
    "#f0f4c3",
    "#fff9c4",
    "#ffecb3",
    "#ffe0b2",
    "#ffccbc",
    "#d7ccc8",
    "#f5f5f5",
    "#cfd8dc",
]
g = nx.Graph()
for (i, r) in groups.iterrows():
    for a in r["Attributes"]:
        g.add_node(a, color=colors[r["Group ID"]])
# %%
for (i, r) in df_assoc.iterrows():
    g.add_edge(
        r["a1"],
        r["a2"],
        weight=r["cramer"],
        penwidth=5,
        style="dashed",
        color="#d1c4e9",
    )

g_max_st = nx.maximum_spanning_tree(g)

for (i, r) in groups.iterrows():
    for (a, b) in itertools.combinations(r["Attributes"], 2):
        k = tuple(sorted((a, b)))
        v = df_assoc[df_assoc["key"] == k]["cramer"].values[0]
        g_max_st.add_edge(a, b, penwidth=1 + (v * 3), color="black", style="solid")
    # for a in r["Attributes"]:
    #     g.add_node(a, color=colors[r["Group ID"]])


write_dot(g_max_st, "./cramer.dot")

