# %%
import pandas as pd
import itertools
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import numpy as np
from nds import nds

from clickhouse_io import get_selectors


# ./results/1643847207/finals.csv
# ./results/1643832341/finals.csv

key = "1643996637"
criteria = [
    # "absolute_risk",
    # "relative_risk",
    "ppv",
    "npv",
    "markedness"
    # "full_prevalence",
]
maxmin = ["max", "max", "max"]
df = pd.concat(
    [
        nds(
            pd.read_csv(f"./results/{key}/0/0/uniques.csv").query(
                "significant == True"
            ),
            criteria,
            maxmin,
        ),
        nds(
            pd.read_csv(f"./results/{key}/0/1/uniques.csv").query(
                "significant == True"
            ),
            criteria,
            maxmin,
        ),
    ]
).sort_values("level", ascending=True)
# df = nds(df, ["absolute_risk", "relative_risk", "support"], ["max", "max", "max"])

sels = get_selectors("arm.quimica")


semantic_groups = {
    0: ["W", "Fe", "V", "Co", "Mo", "Ni", "CoMo", "Al2O3", "hds"],
    1: ["time", "temperatura", "reuse", "funcion_de_bondad", "oxidative", "O_S"],
}
# semantic_groups = {
#     0:["oxidative", "hds", "temperatura", "funcion_de_bondad", "Mo", "Al2O3"],
#     1: ["time", "O_S", "CoMo", "ionic_liquid", "Ni"]
# }
sg0 = semantic_groups[0]
sg1 = semantic_groups[1]


def format_selector(s):
    (a, v) = s
    return f"[{a}={v}]"


sg0_sels = []
sg1_sels = []
for sl in sg0:
    sg0_sels += [format_selector(s) for s in sels[sl]]

for sl in sg1:
    sg1_sels += [format_selector(s) for s in sels[sl]]


# .query(
#     "sg_pair == 0 or sg_pair == 1"
# )
df["rule"] = df["rule"].apply(lambda v: eval(v))
m = "level"
direction = "min"

pairs = []
for (i, r) in df.iterrows():
    [antecedent, consequent] = r["rule"]
    for (a, c) in itertools.product(antecedent, consequent):
        pair = {}
        pair["rule"] = r["repr"]
        pair["source"] = format_selector(a)
        pair["target"] = format_selector(c)
        pair[m] = r[m]
        pair["length"] = r["length"]
        pairs.append(pair)

df_pairs = pd.DataFrame(pairs)
df_pairs = df_pairs.sort_values(by=[m, "length"], ascending=(direction == "min", False))

sg0_sels = df_pairs["source"].unique()
sg1_sels = df_pairs["target"].unique()

# %%
df_pairs["source"].value_counts(normalize=True)

# %%
df_pairs["target"].value_counts(normalize=True)
# %%

df_best_pairs = df_pairs.drop_duplicates(subset=["source", "target"])
# %
sg0_prefs = dict()
sg1_prefs = dict()
for sel in sg0_sels:
    ranked = (
        df_best_pairs[df_best_pairs["source"] == sel][m]
        .rank(method="first", ascending=direction == "min")
        .astype(int)
    )
    targets = []
    for (i, rank) in ranked.items():
        elem = df_best_pairs.loc[i, "target"]
        sg0_prefs[sel] = sg0_prefs.get(sel, []) + [elem]

for sel in sg1_sels:
    ranked = (
        df_best_pairs[df_best_pairs["target"] == sel][m]
        .rank(method="first", ascending=direction == "min")
        .astype(int)
    )
    targets = []
    for (i, rank) in ranked.items():
        elem = df_best_pairs.loc[i, "source"]
        sg1_prefs[sel] = sg1_prefs.get(sel, []) + [elem]
# %%
#
sg0_prefs
sg0_caps = {}
sg0_in_prefs = list(sg0_prefs.keys())
sg1_in_prefs = list(sg1_prefs.keys())

sg0_preferred_targets = np.ceil(
    df_best_pairs[df_best_pairs["source"].isin(sg0_in_prefs)]["target"].value_counts(
        normalize=True
    )
    * len(sg0_in_prefs)
).astype(int)


sg1_preferred_targets = np.ceil(
    df_best_pairs[df_best_pairs["source"].isin(sg1_in_prefs)]["target"].value_counts(
        normalize=True
    )
    * len(sg1_in_prefs)
).astype(int)

# %%
dict(sg0_preferred_targets)

# %%
sg1_preferred_targets

# %%
from matching.games import HospitalResident


matched = HospitalResident.create_from_dictionaries(
    sg0_prefs, sg1_prefs, dict(sg0_preferred_targets)
).solve(optimal="hospital")
matched

# matched2 = HospitalResident.create_from_dictionaries(
#     sg1_prefs, sg0_prefs, dict(sg1_preferred_targets)
# ).solve(optimal="resident")

# matched = {*matched, *matched2}
# %%
acc = []
for (target, ss) in matched.items():
    for (s, t) in itertools.product(ss, [target]):
        r = list(df_best_pairs.query(f"source == '{s}' and target == '{t}'")["rule"])[0]

        # r = df_best_pairs[
        #     (df_best_pairs["source"] == s) & (df_best_pairs["target"] == t)
        # ]["rule"]

        selection = df[df["repr"] == str(r)].copy()

        selection["source"] = s
        selection["target"] = t
        acc.append(selection)

selected_rules = (
    pd.concat(acc)
    .drop_duplicates(subset="repr")[["repr", "level", "sg_pair", *criteria]]
    .sort_values("level")
)
selected_rules
# %%
list(dict(selected_rules["repr"]).values())
# %%
