# %%
import seaborn as sns
import matplotlib.pyplot as plt
from nds import nds

import pandas as pd

r = 1648577771
df = pd.read_csv("./results/1648578793/selection.csv")
# .query(
#     "(sg_pair == 2 or sg_pair == 3)"
# )
# df.drop("level", axis=1)
df = nds(df, ["absolute_risk", "r_absolute_risk"], ["max", "max"])
df["pareto"] = df["level"] == 1
df = df.sort_values("pareto")
# %%
sns.relplot(
    data=df,
    x="absolute_risk",
    y="r_absolute_risk",
    size="absolute_risk_abs",
    hue="pareto",
)
# %%
df.query("level == 1")[["repr"]]

# %%
