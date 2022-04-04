# %%
import pandas as pd
from clickhouse_io import (
    c_evaluate_assoc_measures,
    c_get_confusion_matrix,
    get_selectors,
)
from measures import evaluate_confusion_matrix, apply_evaluation
from nds import nds

# %%
get_selectors("arm.covid")

# %%
ev = apply_evaluation(
    "arm.covid", ((("CLASIFICACION_FINAL", 1),), (("DEFUNCION", 1),),),
)
dfev = pd.DataFrame([ev])
dfev[["repr", "cer", "eer", "markedness", "tp", "relative_risk"]]

# %%
finals = pd.read_csv("./finals_unique.csv")
sels = []
for p in range(0, 34):
    e = p
    # n = e + 1
    n = e

    p_finals = finals.query(f"sg_pair == {e} or sg_pair == {n}")
    sel = nds(p_finals, ("markedness", "cer"), ("max", "min"), until_level=1,).query(
        "level == 1"
    )
    sel["gp"] = f"{e}, {n}"
    sels.append(sel)
df_sels = pd.concat(sels)
df_sels.to_csv("./selection.csv", index=False)
df_sels
# %%
