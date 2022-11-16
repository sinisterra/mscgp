# %%
import pandas as pd
from tabulate import tabulate

df = pd.read_csv("./all_wave_results.csv")
df
# %%
experiment_fns = {
    1: ["support", "confidence", "lift"],
    2: ["absolute_risk", "r_absolute_risk"],
    3: ["susceptibility", "paf"],
}

dfm = pd.melt(df, id_vars=["repr", "scenario", "problem", "wave"], var_name="measure")
dfm["wave"] = dfm["wave"].replace({"A": "T"})
dfm

# %%
# %%
# for (sc,  tbl) in dfm.groupby("scenario"):
#     print(tbl.pivot_table(index="repr", columns=["scenario", "problem", "measure", "wave"]))
# %%
def sort_rule(x):
    print(x)
    return x


for ((sc, pr), tbl) in dfm.groupby(["scenario", "problem"]):
    # for (pr, tbl) in tbl_sc.groupby("problem"):
    output_tbl = tbl
    # output_tbl = tbl[tbl["measure"].isin(experiment_fns[pr])]
    output_tbl["measure"] = pd.Categorical(
        output_tbl["measure"],
        [
            "full_support",
            "confidence",
            "lift",
            "absolute_risk",
            "r_absolute_risk",
            "susceptibility",
            "paf",
        ],
    )

    output_tbl = output_tbl.sort_values("measure")
    output_tbl = output_tbl.pivot_table(
        index="repr", columns=["scenario", "problem", "measure", "wave"]
    )
    print(output_tbl.dropna())
    output_tbl.dropna().to_csv(f"./wave_tables/{sc}{pr}.csv")
    # .dropna()
    #     .to_csv(f"./wave_tables/{sc}{pr}.csv", index=False),
    # )


# %%
