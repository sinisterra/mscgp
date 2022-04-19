# %%
import pygmo as pg
import pandas as pd


df = pd.read_csv("./results/1650330482/selection.csv")

vs = ["absolute_risk", "r_absolute_risk"]
pareto = df.query("level == 1")[vs]
# for v in vs:
#     pareto[v] = -pareto[v]

pareto = pareto.values.tolist()
pareto
# %%
hv = pg.hypervolume(pareto)
hv.compute(
    [1.1, 1.1,]
)

# %%
