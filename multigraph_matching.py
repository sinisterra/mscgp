# %%
import networkx as nx
import pandas as pd
import itertools

file_key = "1645637655"

df = (
    # pd.read_csv(f"./results/{file_key}/0/0/uniques.csv")
    pd.read_csv(f"two_by_two_0.csv")
    .query("significant == True")
    .reset_index(drop=True)
)

df["rule"] = df["rule"].apply(lambda r: eval(r))

df

g = nx.Graph()
# %%
df = df.sort_values("markedness", ascending=False).reset_index(drop=True)
source_set = set()
target_set = set()
for (i, r) in df.iterrows():
    (a, c) = r["rule"]
    for (s, t) in itertools.product(a, c):
        source_set = source_set.union(set([s]))
        target_set = target_set.union(set([t]))
        if g.has_edge(s, t):
            if r["markedness"] > g[s][t]["weight"]:
                g.add_edge(
                    s, t, label=i, source=s, target=t, weight=r["markedness"], rule=r
                )
        else:
            g.add_edge(
                s, t, label=i, source=s, target=t, weight=r["markedness"], rule=r
            )

g_max_weight = sorted(nx.max_weight_matching(g))
g_max_weight
# %%
indices = [g[s][t]["label"] for (s, t) in g_max_weight]
sources = [g[s][t]["source"] for (s, t) in g_max_weight]
targets = [g[s][t]["target"] for (s, t) in g_max_weight]

indices
# %%
selection = df.iloc[indices]
selection["sources"] = sources
selection["targets"] = targets
selection
# %%
selection.to_csv("g_max_weight.csv", index=False)

# %%
nx.max_weight_matching(g)

# %%
