# %%
import pandas as pd
from nds import nds
from tabulate import tabulate
from scipy.stats import gmean
import pareto


acc = []
level = 1

df = pd.read_csv("./uniques.csv").query("nnt > 0 and significant == True")
df

df["paf_pop"] = (df["absolute_risk_abs"] / df["full_total"]).round(4)

of_cols_ = ["prevalence", "relative_risk"]
directions = ["min", "max"]

of_cols = [list(df.columns).index(i) + 1 for i in of_cols_]
of_cols

for (o, d) in zip(of_cols_, directions):
    if d == "max":
        df[o] = -df[o]


def t(v):
    return 1 / (1 + v)


until_level = 10
while len(df) > 0:
    if level > until_level:
        df["level"] = None
        acc.append(df)
        break

    nondominated = pareto.eps_sort([list(df.itertuples(True))], of_cols,)

    df_pareto = pd.DataFrame.from_records(
        nondominated, columns=["Index"] + list(df.columns.values)
    )
    # for o in of_cols_:
    #     df_pareto[o] = -df_pareto[o]
    for (o, d) in zip(of_cols_, directions):
        if d == "max":
            df_pareto[o] = -df_pareto[o]

    df_pareto = df_pareto.sort_values(
        # "gmean", ascending=False
        of_cols_,
        ascending=([d == "min" for d in directions]),
    )

    in_pareto = list(df_pareto["Index"].unique())
    df = df[~(df.index.isin(in_pareto))]
    df_pareto["level"] = level
    acc.append(df_pareto)

    level += 1

all_pareto = pd.concat(acc)
all_pareto = all_pareto[
    [
        "repr",
        "level",
        *of_cols_,
        *[c for c in all_pareto.columns if c not in ["level", "repr", *of_cols_]],
    ]
]
all_pareto = all_pareto.drop("Index", 1)
all_pareto.to_csv("./pareto.csv", index=False)


# %%
