# %%
import pandas as pd
from nds import nds

df = pd.read_csv("./rules.csv").query("significant == True and full_tp >= 10000")
df = df[df.apply(lambda r: "SECTOR" in r["repr"], axis=1)]

r = nds(df, ("markedness", "full_tp"), ("max", "max"), until_level=1).query(
    "level == 1"
)
r.to_csv("./nds.csv", index=False)
r[["repr", "markedness", "full_tp"]]
