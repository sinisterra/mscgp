# %%
import pandas as pd
from tabulate import tabulate

results = [
    # key, scenario, problem
    (1651764528, "A", 1),
    (1652364172, "A", 2),
    (1651765192, "A", 3),
    (1651765572, "B", 1),
    (1651766446, "B", 2),
    (1651767095, "B", 3),
    (1651767416, "C", 1),
    (1651768182, "C", 2),
    (1651768875, "C", 3),
    (1651769382, "D", 1),
    (1651788498, "D", 2),
    (1651791823, "D", 3),
    (1651792641, "E", 1),
    (1651794132, "E", 2),
    (1651794770, "E", 3),
    (1651803174, "F", 1),
    (1651950668, "F", 2),
    (1651952496, "F", 3),
]

# por cada resultado, extraer...
# la longitud del frente de pareto
# los extremos: la regla de mayor aptitud en cada extremo

problems = {
    1: ["support", "confidence", "lift"],
    2: ["absolute_risk", "r_absolute_risk"],
    3: ["susceptibility", "paf"],
}

acc_lengths = []
acc_extremes = []
for (key, scenario, problem) in results:
    df = pd.read_csv(f"./results/{key}/selection.csv").query("level == 1")
    problem_measures = problems[problem]

    for (i, s) in enumerate(df["seed"].unique()):
        run_df = df.query(f"seed == {s}")
        front = {
            "scenario": scenario,
            "problem": problem,
            "length": len(run_df),
            "run": i + 1,
        }
        acc_lengths.append(front)
        for m in [*problem_measures, "aptitude"]:
            df_max = run_df[run_df[m] == run_df[m].max()].copy()
            df_max["measure"] = m
            df_max["scenario"] = scenario
            df_max["problem"] = problem
            acc_extremes.append(df_max)

    # eliminar duplicados de (scenario, problem, rule)

df_lengths = pd.DataFrame(acc_lengths)
acc_df_lengths = []
for ((sc, pr), g) in df_lengths.groupby(["scenario", "problem"]):
    acc_df_lengths.append(
        {"scenario": sc, "problem": pr, **dict(g["length"].describe().round(3))}
    )

df_lns = pd.DataFrame(acc_df_lengths)

acc_sc = {}
for (k, g) in df_lns.groupby(["scenario", "problem"]):
    (scn, pr) = k
    entry = acc_sc.get(scn, {"scenario": scn})
    for m in ["mean", "std"]:
        entry[f"p{pr}_{m}"] = g[m].values[0]
    acc_sc[scn] = entry
    # print(g[].values[0])
print(
    tabulate(
        pd.DataFrame(list(acc_sc.values())),
        headers="keys",
        showindex=False,
        tablefmt="latex",
    )
)

print(tabulate(df_lns, headers="keys", showindex=False))
df_extremes = pd.concat(acc_extremes)

extreme_centrals = []
for ((sc, pr, m), g) in df_extremes.groupby(["scenario", "problem", "measure"]):
    d = dict(g[m].describe().round(3))
    extreme_centrals.append({"scenario": sc, "problem": pr, "measure": m, **d})

df_extreme_centrals = pd.DataFrame(extreme_centrals)
# print(tabulate(df_extreme_centrals, headers="keys", showindex=False))
import itertools

tendency = ["mean", "std"]
for (problem, g) in df_extreme_centrals.groupby(["problem"]):
    acc_problem = {}
    p_measures = problems[problem] + ["aptitude"]
    for (p, t) in itertools.product(p_measures, tendency,):
        for sc in g["scenario"].unique():
            val = g.query(f"measure == '{p}' and scenario == '{sc}'")[t].values[0]
            # print(sc, problem, t, p, val)
            key = f"{p}_{t}"
            scn = acc_problem.get(sc, {"scenario": sc})
            scn[key] = val
            acc_problem[sc] = scn
            # acc_problem.append({"scenario": sc, key: val})
    print(
        tabulate(
            pd.DataFrame(list(acc_problem.values())),
            headers="keys",
            showindex="never",
            tablefmt="latex",
        )
    )
    print(f"\n\nTable: Problema {problem} {{#tbl:resultados_multi_{problem}}}\n\n")
