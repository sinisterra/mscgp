# %%
import networkx as nx
from rule_repr import repr_selectors
import pandas as pd
import itertools
from nds import nds
from semantic_group_discovery import find_clique_cover
from tabulate import tabulate

file_key = "1645662822"

df = (
    # pd.read_csv(f"./results/{file_key}/0/0/uniques.csv")
    pd.read_csv("./results/1646339050/finals.csv")
    # .query("significant == True")
    .reset_index(drop=True)
)


df["rule"] = df["rule"].apply(lambda r: eval(r))
print(len(df))
# %%
df = nds(df, ["absolute_risk", "full_absolute_risk_rev"], ["max", "max"])
measure = "absolute_risk"
optimize = "max"
df = df.sort_values(measure, ascending=True).reset_index(drop=True)

# %%
pairs = {}
rules = {}
selector_set = []
set_values = {}
g = nx.Graph()
selector_weights = {}
for (i, r) in df.iterrows():
    (a, c) = r["rule"]
    selectors = tuple(set([*a, *c]))
    selector_name = repr_selectors(selectors)
    set_values[selectors] = r[measure]
    selector_set.append(set(selectors))
    # print(selector_name, r[measure])
    g.add_node(selector_name, row=r)
    selector_weights[selector_name] = r[measure]

    # (a2, c2) = r2["rule"]
    # selectors1 = set([*a1, *c1])
    # selectors2 = set([*a2, *c2])
combs = list(itertools.combinations(selector_set, 2))
print(len(combs))
for (i, (a, b)) in enumerate(combs):
    # print(i / len(combs))
    intr = a.intersection(b)
    if len(intr) > 0:
        g.add_edge(repr_selectors(tuple(a)), repr_selectors(tuple(b)), intr=intr)

# %%
gc = nx.complement(g)
for (e, weight) in selector_weights.items():
    gc.add_node(
        e,
        weight=1000000 - int((1 / weight) * 1000)
        if optimize == "min"
        else int(weight * 1000),
    )

# %%
# (elems, w) = nx.max_weight_clique(gc)

elems = []
ws = []
print("Begin finding cliques...")
while True:
    (it_elems, w) = nx.max_weight_clique(
        gc.subgraph([e for e in gc.nodes if e not in elems])
    )
    ws += [w]
    elems += it_elems

    if len(it_elems) == 0:
        break

    dfsel = pd.DataFrame([g.nodes[e]["row"] for e in it_elems])
    print(
        tabulate(
            dfsel[
                ["repr", "level", "absolute_risk", "full_absolute_risk_rev", "sg_pair"]
            ],
            headers="keys",
            tablefmt="",
            showindex=False,
        )
    )

    # if len(ws) == 1:
    #     break

print(ws)

# if len(elems) > 0:

#     dfsel.to_csv("./clique_selection.csv", index=False)
#     print(dfsel[["repr", measure, "absolute_risk", "absolute_risk_abs"]])
# else:
#     print("No max weight clique found")


# %%
