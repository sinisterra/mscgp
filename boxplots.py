# %%
import itertools
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

run_id = "results/1637360805"
total_runs = 10
total_sgs = 34

# %%
# Accumulate convergence results for each semantic_group
acc = []
for sg in range(total_sgs):
    for r in range(total_runs):
        convergence = pd.read_csv(f"{run_id}/{r}/{sg}/convergence.csv")
        convergence["run"] = r
        convergence["sg_pair"] = sg
        convergence["inverse"] = sg % 2 != 0
        acc.append(convergence)

df_acc = pd.concat(acc)
df_acc
# %%
ignored_measures = ["Generation", "run", "sg_pair", "inverse"]
os.makedirs(f"{run_id}/convergence", exist_ok=True)
for (i, measure) in itertools.product(
    range(0, total_sgs // 2), [c for c in df_acc.columns if c not in ignored_measures]
):
    gr = i * 2
    grn = gr + 1
    ax = sns.catplot(
        data=df_acc.query(f"sg_pair == {gr} or sg_pair == {grn}"),
        # col="sg_pair",
        # col_wrap=1,
        x="Generation",
        y=measure,
        kind="box",
        hue="inverse",
        aspect=21 / 9,
        # width=16,
        # height=9,
    )
    plt.title(f"Semantic groups {gr} and {grn} [{measure}]")
    plt.savefig(f"{run_id}/convergence/{gr}_{grn}_{measure}.jpg")

# %%
