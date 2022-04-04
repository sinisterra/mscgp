# %%
from xml.etree.ElementInclude import include
import pandas as pd
import seaborn as sns

df = pd.read_csv("./etl/quimica.csv")
df

# %%
df["reuse"].describe()

# %%
sns.displot(df["reuse"], bins=5, rug=True)

# %%
sns.countplot(pd.qcut(df["reuse"], 5, duplicates="drop"))

# %%
sns.countplot(pd.cut(df["reuse"], bins=5, include_lowest=True, duplicates="drop"))

# %%
sns.countplot(df["Mo"])

# %%
