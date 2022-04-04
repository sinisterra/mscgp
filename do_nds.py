# %%
import pandas as pd
from nds import nds

df = pd.read_csv("./rules_full_3.csv")

dfnds = nds(df, ["absolute_risk", "relative_risk"], ["max", "max"], until_level=10,)[
    [
        "repr",
        "cer",
        "eer",
        "absolute_risk",
        "relative_risk",
        "support",
        "confidence",
        "lift",
        "full_tp",
        "prevalence",
        "full_prevalence",
        "level",
    ]
]
dfnds.to_csv("./full_nds.csv", index=False)
dfnds
