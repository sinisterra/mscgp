# %%
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import girvan_newman
from networkx.drawing.nx_pydot import write_dot
import seaborn as sns
from clickhouse_io import get_selectors

colors = ["#dc1c13", "#ea4c46", "#f07470", "#f1959b", "#f6bdc0"]

df = pd.read_csv("./adjacencies.csv")
df = df.set_index("a1")
df

# %%
bins = 5
g = nx.from_pandas_adjacency(df)
series = pd.Series(nx.pagerank(g)).sort_values(ascending=False)
series
# %%
sns.displot(series, rug=True, kind="hist")
# %%
bin_disc = pd.qcut(series, bins, labels=False, duplicates="drop")
acc_bin = {}
for (a, b) in bin_disc.items():
    acc_bin[b] = acc_bin.get(b, []) + [a]
    g.add_node(
        a,
        color=f"#6082B6{str(hex(int(np.interp(b, [0,bins - 1], [int(0.05 * 255), int(1 * 255)]))))[2:].zfill(2)}",
        label=f"{a} ({b})",
    )
acc_bin

# %%
quant_disc = pd.qcut(series, bins, labels=False, duplicates="drop")
acc_quant = {}
for (a, b) in quant_disc.items():
    acc_quant[b] = acc_quant.get(b, []) + [a]
acc_quant

# %%
write_dot(g, "pagerank.dot")
