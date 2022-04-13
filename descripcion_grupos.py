# %%
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
from tabulate import tabulate

df = pd.read_csv("./assoc_full.csv")
df = df[["a1", "a2", "cramer", "significant"]]

df["key"] = df.apply(lambda r: tuple(sorted((r["a1"], r["a2"]))), axis=1)
df["removed"] = df["cramer"] < 0.08
df = df.drop_duplicates("key").query("significant == True")

# %%
dfg = pd.read_csv("./sg_groups.csv")
dfg["Attributes"] = dfg["Attributes"].apply(
    lambda v: v.replace('"', "").replace("[", "").replace("]", "").replace("'", "`")
)
print(
    tabulate(
        dfg[["Group ID", "Attributes"]],
        headers="keys",
        showindex="False",
        tablefmt="github",
    )
)
# %%
# sns.set(rc={"figure.figsize": (16, 9)})
plt.figure(figsize=(11, 6))
ax = sns.histplot(data=df, x="cramer", bins=20, color="#7887c2")
ax.set(xlabel="V de Cramér", ylabel="Frecuencia")
plt.axvline(0.08, 1, 0, color="red")
plt.savefig("./charts/dist_cramer.pdf", dpi=300, bbox_inches="tight")
# %%
df["cramer"].describe().round(5)

# %%
zscore(df["cramer"]).describe().round(5)
# %%
sns.ecdfplot(df["cramer"])

# %%
sns.displot(data=df.query("removed == False"), x="cramer", bins=20, stat="probability")

# %%
sns.displot(data=df.query("removed == True"), x="cramer", bins=20, stat="probability")

# %%
sns.displot(data=df.query("removed == False"), x="cramer", bins=20, stat="probability")
