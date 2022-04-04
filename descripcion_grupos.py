# %%
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt

df = pd.read_csv("./assoc_full.csv")
df = df[["a1", "a2", "cramer", "significant"]]

df["key"] = df.apply(lambda r: tuple(sorted((r["a1"], r["a2"]))), axis=1)
df["removed"] = df["cramer"] < 0.08
df = df.drop_duplicates("key").query("significant == True")

# %%
sns.histplot(data=df, x="cramer", bins=20)
plt.axvline(0.08, 1, 0, color="red")
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
