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

key = "1644344515"
criteria = [
    "absolute_risk",
    "relative_risk",
    "support",
    "confidence",
    "full_prevalence",
]
maxmin = ["max", "max", "max", "max", "max"]
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
).sort_values("absolute_risk", ascending=False)
# df = nds(df, ["absolute_risk", "relative_risk", "support"], ["max", "max", "max"])

sels = get_selectors("arm.quimica")


semantic_groups = {
    0: ["DIABETES", "HIPERTENSION", "TABAQUISMO", "SEXO", "EDAD"],
    1: ["NEUMONIA", "DEFUNCION", "INTUBADO"],
}
# semantic_groups = {
#     0:["oxidative", "hds", "temperatura", "funcion_de_bondad", "Mo", "Al2O3"],
#     1: ["time", "O_S", "CoMo", "ionic_liquid", "Ni"]
# }
sg0 = semantic_groups[0]
sg1 = semantic_groups[1]

# .query(
#     "sg_pair == 0 or sg_pair == 1"
# )
df["rule"] = df["rule"].apply(lambda v: eval(v))
m = "level"
direction = "min"


# %%
pairs = {}
pairs_length = {}
rule_selected = {}
for (i, r) in df.iterrows():
    rule = r["rule"]
    for k in itertools.product(*rule):
        k = (tuple(k[0]), tuple(k[1]))
        if "max" == direction:
            should_replace = r[m] >= pairs.get(tuple(k), r[m])
            if should_replace:
                r_val = pairs.get(k, None)
                pairs[k] = max(pairs.get(k, r[m]), r[m])
                r_len = pairs_length.get(k, 0)
                pairs_length[k] = max(pairs_length.get(k, r["length"]), r["length"])
                next_len = r["length"]
                if r_val == r[m]:
                    if r["length"] > pairs_length[k]:
                        print(
                            f"Replace ({r_val}, {r_len}) with ({r[m]}, {r['length']})"
                        )
                        rule_selected[k] = r["repr"]
                else:
                    rule_selected[k] = r["repr"]

        if "min" == direction:
            should_replace = r[m] <= pairs.get(tuple(k), r[m])
            if should_replace:
                r_val = pairs.get(k, None)
                pairs[k] = min(pairs.get(k, r[m]), r[m])
                r_len = pairs_length.get(k, 0)
                pairs_length[k] = max(pairs_length.get(k, r["length"]), r["length"])
                next_len = r["length"]
                if r_val == r[m]:
                    if r["length"] > pairs_length[k]:
                        print(
                            f"Replace ({r_val}, {r_len}) with ({r[m]}, {r['length']})"
                        )
                        rule_selected[k] = r["repr"]
                else:
                    rule_selected[k] = r["repr"]
pairs

# %%
s = set(rule_selected.values())
len(s)

# %%
g = nx.DiGraph()
as_triplets = []
for (k, v) in pairs.items():
    l1 = f"[{k[0][0]}={k[0][1]}]"
    l2 = f"[{k[1][0]}={k[1][1]}]"
    if "PAIS" not in l1 and "PAIS" not in l2:
        as_triplets.append(
            {
                "source": k[0],
                "target": k[1],
                m: v,
                "rule": rule_selected[k],
                "rule_length": len([*rule_selected[k][0], *rule_selected[k][1]]),
            }
        )
        g.add_edge(
            l1, l2, label=v, weight=v,
        )

write_dot(g, "viz.dot")


# %%
df_triplets = (
    pd.DataFrame(as_triplets)
    # .query(f"{m} > 0")
    # .sort_values([m, "rule_length"], ascending=(False, False))
)
df_triplets["direction"] = "none"
tr_s = []
tr_t = []
for s in df_triplets["source"].unique():
    df_triplets_s = (
        df_triplets[df_triplets["source"] == s].reset_index(drop=True).iloc[0:1]
    )
    df_triplets_s["direction"] = "source"

    tr_s.append(df_triplets_s)

for t in df_triplets["target"].unique():
    df_triplets_t = (
        df_triplets[df_triplets["target"] == t].reset_index(drop=True).iloc[0:1]
    )
    df_triplets_t["direction"] = "target"
    tr_t.append(df_triplets_t)

df_triplets_s = pd.concat(tr_s).drop_duplicates(subset=["source", "target"])
df_triplets_t = pd.concat(tr_t).drop_duplicates(subset=["source", "target"])
# df_triplets.drop_duplicates(subset="target",).reset_index(drop=True)
selected_rules = df[
    df["repr"].isin(
        set(df_triplets_s["rule"].unique()).union(df_triplets_t["rule"].unique())
    )
]

selected_rules.to_csv("selection.csv", index=False)
# %%
g_max = nx.DiGraph()


g_sources = nx.DiGraph()
g_targets = nx.DiGraph()
for (i, r) in df_triplets_s.iterrows():
    s = r["source"]
    t = r["target"]
    l1 = f"[{s[0]}={s[1]}]"
    l2 = f"[{t[0]}={t[1]}]"
    g_sources.add_edge(l1, l2, weight=r[m])
    g_max.add_edge(
        l1,
        l2,
        weight=r[m],
        label=r[m],
        color="#6082B6",
        # color="green",
        penwidth=round(np.interp(r[m], [0, 1], [1, 8]), 3),
    )

for (i, r) in df_triplets_t.iterrows():
    s = r["source"]
    t = r["target"]
    l1 = f"[{s[0]}={s[1]}]"
    l2 = f"[{t[0]}={t[1]}]"
    g_targets.add_edge(l1, l2, weight=r[m])
    g_max.add_edge(
        l1,
        l2,
        weight=r[m],
        label=r[m],
        color="#6082B6",
        # color="cyan" if g_max.has_edge(l1, l2) else "red",
        penwidth=round(np.interp(r[m], [0, 1], [1, 8]), 3),
    )

# for (i, r) in selected_rules.iterrows():
#     rule = r["rule"]
#     # connect antecedents
#     for s in rule[0]:
#         l1 = f"[{s[0]}={s[1]}]"
#         # l2 = f"[{t[0]}={t[1]}]"
#         g_max.add_edge(l1, f"r{i}", color="black")

#     for s in rule[1]:
#         l1 = f"[{s[0]}={s[1]}]"
#         # l2 = f"[{t[0]}={t[1]}]"
#         g_max.add_edge(f"r{i}", l1, color="red")
#         # g_max.add_edge(l2, l1, color="cyan")


# %%
len(df_triplets_s), len(df_triplets_t)

# %%
bins = 11
pageranked = pd.cut(
    pd.Series(nx.pagerank(g_max)), bins=bins, labels=False, duplicates="drop"
).sort_values(ascending=False)
for (a, b) in pageranked.items():
    g_max.add_node(
        a,
        # color=f"#6082B6{str(hex(int(np.interp(b, [0,bins - 1], [int(0.05 * 255), int(1 * 255)]))))[2:].zfill(2)}",
    )
pageranked
# %%
write_dot(g_max, "maxes.dot")

# %%
df_sources = nx.to_pandas_adjacency(g_sources)
df_targets = nx.to_pandas_adjacency(g_targets)


g0 = [[f"[{a}={v}]" for (a, v) in sels[s]] for s in sg0]
g1 = [[f"[{a}={v}]" for (a, v) in sels[s]] for s in sg1]

g0 = [item for sublist in g0 for item in sublist]
g1 = [item for sublist in g1 for item in sublist]
adjm = nx.to_pandas_adjacency(g)
adj_sr = adjm.loc[
    [c for c in g0 if c in adjm.index], [c for c in g1 if c in adjm.columns]
].transpose()
adj_rs = adjm.loc[
    [c for c in g1 if c in adjm.index], [c for c in g0 if c in adjm.columns]
].transpose()

#  %%
resident_preferences = {}
for c in adj_sr.columns:

    resident_preferences[c] = list(
        adj_sr[adj_sr[c] > 0][c]
        .rank(method="first", ascending=False if direction == "max" else True)
        .sort_values()
        .keys()
    )
resident_preferences
# %%
hospital_preferences = {}
for c in adj_rs.columns:
    hospital_preferences[c] = list(
        adj_rs[adj_rs[c] > 0][c]
        .rank(method="first", ascending=False if direction == "max" else True)
        .sort_values()
        .keys()
    )

    for (a, b) in itertools.product([c], hospital_preferences[c]):
        # verificar que b esté en a
        b_prefs = resident_preferences.get(b, [])
        if a not in b_prefs:
            resident_preferences[b] = b_prefs + [a]

for c in adj_sr.columns:
    for (a, b) in itertools.product([c], resident_preferences[c]):
        # verificar que b esté en a
        b_prefs = hospital_preferences.get(b, [])
        if a not in b_prefs:
            hospital_preferences[b] = b_prefs + [a]
        # print(a, b, a in b_prefs)


# fix_preferences_changed = [
#     ("[temperatura=[9.08, 9.4]]", "[ionic_liquid=1]"),
#     ("[time=[5.65, 8.48]]", "[oxidative=1]"),
# ]

# for (a, b) in fix_preferences_changed:
#     hospital_preferences[a] = list(set(hospital_preferences.get(a, []) + [b]))

# hospital_preferences["[reuse=[2.8, 10.0]]"] = hospital_preferences.get(
#     "[reuse=[2.8, 10.0]]", []
# ) + ["[W=1]"]
hospital_preferences

# %%
cap = 1
hospital_capacities = {}
for h in hospital_preferences.keys():
    hospital_capacities[h] = cap

resident_capacities = {}
for h in resident_preferences.keys():
    resident_capacities[h] = cap
# %%
from matching.games import HospitalResident


matched = HospitalResident.create_from_dictionaries(
    resident_preferences, hospital_preferences, hospital_capacities
).solve(optimal="hospital")
matched

# %%
matched_2 = HospitalResident.create_from_dictionaries(
    hospital_preferences, resident_preferences, resident_capacities
).solve(optimal="hospital")
matched_2
# %%
df_triplets_selectors = df_triplets.copy()
df_triplets_selectors["source"] = df_triplets_selectors["source"].apply(
    lambda t: f"[{t[0]}={t[1]}]"
)
df_triplets_selectors["target"] = df_triplets_selectors["target"].apply(
    lambda t: f"[{t[0]}={t[1]}]"
)

matched = {**matched, **matched_2}
rules_acc = []
for (t, a_s) in matched.items():
    for s in a_s:
        rules_acc.append(
            df_triplets_selectors.query(f"source == '{s}' and target == '{t}'")
        )
uniq_rules = pd.concat(rules_acc)
g_matched = nx.DiGraph()

for (i, r) in uniq_rules.iterrows():
    g_matched.add_edge(r["source"], r["target"], label=r[m])

uniques = uniq_rules.drop_duplicates("rule")["rule"].unique()
(uniques, len(uniques))

write_dot(g_matched, "matched.dot")

# %%
uniq_rules
# %%
df[df["repr"].isin(uniques)][
    ["repr", "level", "sg_pair"]
    + [
        *criteria,
        "support",
        "confidence",
        "cer",
        "eer",
        "full_prevalence",
        "full_support",
    ]
].to_csv("matched_rules.csv", index=False)
# %%

