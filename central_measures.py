# %%
import pandas as pd
from scipy.stats import skew
from tabulate import tabulate

executions = [
    (1649267135, "support", 0, "support"),
    (1649267768, "confidence", 0, "confidence"),
    (1649268422, "lift", 0, "lift"),
    (1649256939, "aptitude", 0, "support, confidence, lift",),
    (1649259080, "absolute_risk", 0, "absolute_risk"),
    (1649274174, "r_absolute_risk", 0, "absolute risk (reciprocal)"),
    (1649258089, "aptitude", 0, "biconditional"),
    (1649274314, "susceptibility", 0, "susceptibility"),
    ("1649255045", "paf", 0, "impact"),
    (1649257332, "aptitude", 0, "susceptibility, impact"),
    # ....
    (1649340590, "support", 1, "support"),
    (1649339417, "confidence", 1, "confidence"),
    (1649341028, "lift", 1, "lift"),
    (1649344358, "aptitude", 1, "support, confidence, lift"),
    (1649359734, "susceptibility", 1, "susceptibility"),
    (1649360199, "paf", 1, "impact"),
    (1649344657, "aptitude", 1, "susceptibility, impact"),
    (1649361080, "absolute_risk", 1, "absolute_risk"),
    (1649361386, "r_absolute_risk", 1, "absolute_risk (reciprocal)"),
    (1649344977, "aptitude", 1, "biconditional"),
    # ....
    (1649361796, "support", 2, "support"),
    (1649362020, "confidence", 2, "confidence"),
    (1649363037, "lift", 2, "lift"),
    (1649346487, "aptitude", 2, "support, confidence, lift"),
    (1649362281, "absolute_risk", 2, "absolute_risk"),
    (1649366813, "r_absolute_risk", 2, "absolute_risk (reciprocal)"),
    (1649345806, "aptitude", 2, "biconditional"),
    (1649346846, "aptitude", 2, "susceptibility, impact"),
    # ...
    (1649800896, "support", 3, "support"),
    (1649801322, "confidence", 3, "confidence"),
    (1649797273, "lift", 3, "lift"),
    (1649796513, "aptitude", 3, "support, confidence, lift"),
    (1649806228, "absolute_risk", 3, "absolute_risk"),
    (1649807478, "r_absolute_risk", 3, "absolute_risk (reciprocal)"),
    (1649796823, "aptitude", 3, "biconditional"),
    (1649808534, "susceptibility", 3, "susceptibility"),
    (1649809587, "paf", 3, "impact"),
    (1649796123, "aptitude", 3, "susceptibility, impact"),
    # ...
    (1649794145, "aptitude", 4, "support, confidence, lift"),
    (1649795060, "aptitude", 4, "biconditional"),
    (1649795463, "aptitude", 4, "susceptibility, impact"),
]

acc_bests = []
acc_scenarios = []
acc_aptitude = []
all_results = []
for (key, measure, scenario, label) in executions:
    acc = []
    df = pd.read_csv(f"./results/{key}/selection.csv")

    for s in df["seed"].unique():
        df_run = df.query(f"seed == {s}").sort_values(
            [measure, "full_support"], ascending=[False, False]
        )
        acc_aptitude.append(
            {"label": label, "scenario": scenario, **df_run[measure].describe()}
        )
        row_best = df_run[df_run[measure] == df_run[measure].max()]
        r = row_best[measure].values[0]
        rule = row_best["repr"].values[0]
        acc.append({"run": s, "scenario": scenario, "best": r, "rule": rule})
    dfacc = pd.DataFrame(acc)
    dfacc["scenario"] = scenario
    dfacc["experiment"] = label

    d = dict(dfacc["rule"].value_counts())

    uniques = dfacc.drop_duplicates(subset=["rule", "scenario"]).copy()
    uniques["count"] = uniques["rule"].apply(lambda v: d[v])
    acc_bests.append(uniques)

    dbest = dfacc["best"].describe().to_dict()
    dbest_selected = dfacc[dfacc["best"] == dbest["max"]]["rule"].values[0]
    dbest["scenario"] = scenario
    dbest["measure"] = measure
    dbest["label"] = label
    dbest["run_id"] = key
    dbest["rule"] = dbest_selected
    acc_scenarios.append(dbest)
    # d = df_run[measure].describe().to_dict()
    # acc.append({"run": s, "scenario": scenario, **d})

scenarios = pd.DataFrame(acc_scenarios)
scenarios
# %%
for scenario in scenarios["scenario"].unique():
    dfs = scenarios.query(f"scenario == {scenario}")
    dfs.rename(columns={"50%": "median"}, inplace=True)
    print(
        tabulate(
            dfs[["label", "max", "median", "min", "mean", "std"]].round(4),
            showindex=False,
            headers="keys",
            tablefmt="github",
        )
    )
    print("...\n")
# %%
ab = pd.concat(acc_bests)
ab = ab[["scenario", "rule", "experiment", "best", "count"]].sort_values(
    ["scenario", "experiment", "best", "count"], ascending=[True, True, False, False]
)

print(tabulate(ab, headers="keys", showindex="never", tablefmt="github"))
# %%
ab.to_csv("./best_for_runs.csv", index=False)

# %%
# df_aptitudes = pd.DataFrame(acc_aptitude)
# df_means = (
#     df_aptitudes.groupby(["scenario", "label"])
#     .mean()[["max", "50%", "min", "mean", "std",]]
#     .round(3)
#     .reset_index()
# )
# df_means = df_means.rename(columns={"50%": "median"})
# print(tabulate(df_means, headers="keys", showindex="never", tablefmt="github"))
# %%
import itertools

acc_desc = []
length_desc = []
for (k, m, sc, label) in executions:
    df_experiment = pd.read_csv(f"./results/{k}/selection.csv")

    desc = dict(df_experiment[m].describe().round(4))
    ls = df_experiment["repr"].apply(lambda r: len([s for s in r if s == "["])).round(4)
    length_d = dict(ls.describe())
    length_d["measure"] = m
    length_d["scenario"] = sc
    length_d["label"] = label
    length_d["skewness"] = skew(ls)
    length_desc.append(length_d)
    desc["measure"] = m
    desc["scenario"] = sc
    desc["label"] = label
    desc["skewness"] = skew(df_experiment[m])
    acc_desc.append(desc)
# %%
df_length_desc = pd.DataFrame(length_desc).round(4)
df_length_desc = df_length_desc[
    ["scenario", "label", "max", "50%", "min", "mean", "std", "skewness"]
]
print(tabulate(df_length_desc, headers="keys", showindex="never", tablefmt="github"))
# print(m, sc, df_experiment[m])

# %%
df_acc_desc = pd.DataFrame(acc_desc).round(4)
df_acc_desc = df_acc_desc[
    ["scenario", "label", "max", "50%", "min", "mean", "std", "skewness"]
]
print(tabulate(df_acc_desc, headers="keys", showindex="never", tablefmt="github"))
