# %%
import pandas as pd
from scipy.stats import skew
from tabulate import tabulate

executions = [
    (1651023959, "support", 0, "support"),
    (1651024715, "confidence", 0, "confidence"),
    (1651025471, "lift", 0, "lift"),
    (1651027301, "absolute_risk", 0, "absolute_risk"),
    (1651027512, "r_absolute_risk", 0, "r_absolute_risk"),
    (1651027992, "susceptibility", 0, "susceptibility"),
    (1651028250, "paf", 0, "paf"),
    (1651028625, "aptitude", 0, "support, confidence, lift"),
    (1651028882, "aptitude", 0, "absolute_risk, r_absolute_risk"),
    (1651029575, "aptitude", 0, "susceptibility, paf"),
    (1651022257, "support", 1, "support"),
    (1651029882, "confidence", 1, "confidence"),
    (1651036019, "lift", 1, "lift"),
    (1651030892, "absolute_risk", 1, "absolute_risk"),
    (1651030925, "r_absolute_risk", 1, "r_absolute_risk"),
    (1651031556, "susceptibility", 1, "susceptibility"),
    (1651031676, "paf", 1, "paf"),
    (1651010656, "aptitude", 1, "support, confidence, lift"),
    (1651011064, "aptitude", 1, "absolute_risk, r_absolute_risk"),
    (1651011459, "aptitude", 1, "susceptibility, paf"),
    (1651031996, "support", 2, "support"),
    (1651032004, "confidence", 2, "confidence"),
    (1651066495, "lift", 2, "lift"),
    (1651036908, "absolute_risk", 2, "absolute_risk"),
    (1651069129, "r_absolute_risk", 2, "r_absolute_risk"),
    (1651036926, "susceptibility", 2, "susceptibility"),
    (1651071358, "paf", 2, "paf"),
    (1651011998, "aptitude", 2, "support, confidence, lift"),
    (1651012376, "aptitude", 2, "absolute_risk, r_absolute_risk"),
    (1651012894, "aptitude", 2, "susceptibility, paf"),
    (1651072883, "support", 10, "support"),
    (1651073619, "confidence", 10, "confidence"),
    (1651079007, "lift", 10, "lift"),
    (1651076541, "absolute_risk", 10, "absolute_risk"),
    (1651089192, "r_absolute_risk", 10, "r_absolute_risk"),
    (1651095713, "susceptibility", 10, "susceptibility"),
    (1651097782, "paf", 10, "paf"),
    (1651013353, "aptitude", 10, "support, confidence, lift"),
    (1651014019, "aptitude", 10, "absolute_risk, r_absolute_risk"),
    (1651016618, "aptitude", 10, "susceptibility, paf"),
    (1651100347, "support", 11, "support"),
    (1651100774, "confidence", 11, "confidence"),
    (1651104951, "lift", 11, "lift"),
    (1651112053, "absolute_risk", 11, "absolute_risk"),
    (1651113199, "r_absolute_risk", 11, "r_absolute_risk"),
    (1651114875, "susceptibility", 11, "susceptibility"),
    (1651115863, "paf", 11, "paf"),
    (1651017675, "aptitude", 11, "support, confidence, lift"),
    (1651017686, "aptitude", 11, "absolute_risk, r_absolute_risk"),
    (1651017703, "aptitude", 11, "susceptibility, paf"),
    (1651153558, "support", 12, "support"),
    (1651154007, "confidence", 12, "confidence"),
    (1651157792, "lift", 12, "lift"),
    (1651166727, "absolute_risk", 12, "absolute_risk"),
    (1651169336, "r_absolute_risk", 12, "r_absolute_risk"),
    (1651171208, "susceptibility", 12, "susceptibility"),
    (1651171225, "paf", 12, "paf"),
    (1651018918, "aptitude", 12, "support, confidence, lift"),
    (1651018933, "aptitude", 12, "absolute_risk, r_absolute_risk"),
    (1651018943, "aptitude", 12, "susceptibility, paf"),
]

acc_bests = []
acc_scenarios = []
acc_aptitude = []
all_results = []

acc_total_generations = []
for (key, measure, scenario, label) in executions:
    acc_exp = []
    for exp in range(20):
        try:
            df_exp_generations = pd.read_csv(
                f"./results/{key}/{exp}/0/total_generations.csv"
            )
            df_exp_generations["experiment_id"] = exp
            df_exp_generations["scenario"] = scenario
            df_exp_generations["label"] = label
            acc_exp.append(df_exp_generations)
        except Exception:
            pass
    acc_total_generations += acc_exp

df_acc_tg = pd.concat(acc_total_generations)
total_generations_tendency = []
for ((sc, lbl), g) in df_acc_tg.groupby(["scenario", "label"]):
    dsc = dict(g["total_generations"].describe())
    dsc["scenario"] = sc
    dsc["label"] = lbl
    total_generations_tendency.append(dsc)

print(
    "TOTAL GENERATIONS WHEN STOP CRITERIA WAS REACHED \n"
    + tabulate(
        pd.DataFrame(total_generations_tendency)[
            ["scenario", "label", "min", "50%", "max", "mean", "std"]
        ],
        headers="keys",
        showindex="never",
        tablefmt="github",
    )
)

mss = [
    "support",
    "confidence",
    "lift",
    "aptitude",
    "absolute_risk",
    "r_absolute_risk",
    "susceptibility",
    "paf",
    "full_support",
]

label_problem_mapping = {
    "support": 1,
    "confidence": 2,
    "lift": 3,
    "support, confidence, lift": 4,
    "absolute_risk": 5,
    "r_absolute_risk": 6,
    "absolute_risk, r_absolute_risk": 7,
    "susceptibility": 8,
    "paf": 9,
    "susceptibility, paf": 10,
}
scenario_id_mapping = {0: "A", 1: "B", 2: "C", 10: "D", 11: "E", 12: "F"}

for (key, measure, scenario, label) in executions:
    acc = []
    df = pd.read_csv(f"./results/{key}/selection.csv")
    exp_name = f"{scenario_id_mapping[scenario]}{label_problem_mapping[label]}"
    df.to_csv(f"./mono_results/{exp_name}.csv", index=False)

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
        rs = {"run": s, "scenario": scenario, "best": r, "rule": rule}
        for e in mss:
            if e in df_run.columns:
                rs[e] = row_best[e].values[0]
        acc.append(rs)

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
label_names = {
    "support": "Soporte",
    "confidence": "Confianza",
    "lift": "Ascenso",
    "absolute_risk": "Efecto causal",
    "r_absolute_risk": "Efecto causal (recíproca)",
    "susceptibility": "Susceptibilidad",
    "paf": "Impacto",
    "support, confidence, lift": "Soporte, confianza, ascenso (media geométrica)",
    "absolute_risk, r_absolute_risk": "Efectos causales (media geométrica)",
    "susceptibility, paf": "Susceptibilidad, impacto (media geométrica)",
}
order = [
    "support",
    "confidence",
    "lift",
    "support, confidence, lift",
    "absolute_risk",
    "r_absolute_risk",
    "absolute_risk, r_absolute_risk",
    "susceptibility",
    "paf",
    "susceptibility, paf",
]
scenario_letters = {0: "A", 1: "B", 2: "C", 10: "D", 11: "E", 12: "F"}
_acc = {}
for scenario in scenarios["scenario"].unique():
    scenario_letter = scenario_letters.get(scenario)
    dfs = scenarios.query(f"scenario == {scenario}").copy()
    dfs["order"] = dfs["label"].apply(lambda v: order.index(v))
    dfs["problema"] = dfs["order"] + 1
    dfs["scenario_letter"] = scenario_letter
    dfs["key"] = dfs.apply(lambda r: f"{r['scenario_letter']}{r['problema']}", axis=1)
    dfs = dfs.sort_values("order")

    dfs = (
        dfs[["mean", "std"]]
        .fillna(0)
        .round(3)
        .replace({"label": label_names})
        .reset_index(drop=True)
    )

    for (i, r) in dfs.iterrows():
        acc_s = _acc.get(i, {})
        acc_s[f"{scenario_letter}_mean"] = r["mean"]
        acc_s[f"{scenario_letter}_std"] = r["std"]
        # _acc[i] = _accf"{scenario_letter}_mean"] = r["mean"]
        # _acc[f"{scenario_letter}_std"] = r["std"]
        _acc[i] = acc_s
    print(_acc)
    dfs.rename(
        columns={
            "50%": "median",
            "label": "Problema",
            "key": "Experimento",
            "mean": "$\mu$",
            "std": "$\sigma$",
        },
        inplace=True,
    )
    # print(f"Scenario {scenario}")
    print("\n\n")
    print(tabulate(dfs, showindex=False, headers="keys", tablefmt="github",))
    print(
        f"\n\nTable: Resultados del escenario {scenario_letter} {{#tbl:exc_scenario_{scenario_letter}}}\n"
    )

print(
    tabulate(
        pd.DataFrame(_acc.values()), showindex=True, headers="keys", tablefmt="latex",
    )
)

# %%
ab = pd.concat(acc_bests)
ab = ab[["scenario", "rule", "experiment", "best", "count", *mss]].sort_values(
    ["scenario", "experiment", "best", "count"], ascending=[True, True, False, False]
)

print(len(ab))
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

    desc = dict(df_experiment[m].describe().round(3))
    ls = df_experiment["repr"].apply(lambda r: len([s for s in r if s == "["])).round(3)
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
df_length_desc = pd.DataFrame(length_desc).round(3)
df_length_desc = df_length_desc[
    ["scenario", "label", "max", "50%", "min", "mean", "std", "skewness"]
]
# print(tabulate(df_length_desc, headers="keys", showindex="never", tablefmt="github"))
# print(m, sc, df_experiment[m])

# %%
df_acc_desc = pd.DataFrame(acc_desc).round(3)
df_acc_desc = df_acc_desc[
    ["scenario", "label", "max", "50%", "min", "mean", "std", "skewness"]
]
# print(tabulate(df_acc_desc, headers="keys", showindex="never", tablefmt="github"))
