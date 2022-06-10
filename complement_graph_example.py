# %%
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from semantic_group_discovery import find_clique_cover, colors

size = 10
g = nx.barabasi_albert_graph(size, 3, seed=0)
# g = nx.binomial_graph(size, 0.3, seed=42)

gc = nx.complement(g)
write_dot(nx.complement(g), f"./ba_{size}_c.dot")

cc = find_clique_cover(g)[0]
cc

gwc = g.copy()
# for (u, v) in g.edges():

#     gwc[u][v]["color"] = "transparent"

for (u, v) in gc.edges():
    gwc.add_edge(u, v)
    gwc[u][v]["color"] = "transparent"
    gwc[u][v]["style"] = "dashed"
write_dot(gwc, "./ba_10.dot")


for (k, v) in cc.items():
    for e in v:
        g.nodes[e]["color"] = colors[k]
        gc.nodes[e]["color"] = colors[k]

gcol = g.copy()

for (u, v) in gc.edges():
    gcol.add_edge(u, v)
    gcol[u][v]["color"] = "transparent"

gcolc = gc.copy()


for (u, v) in g.edges():
    gcolc.add_edge(u, v)
    gcolc[u][v]["color"] = "transparent"

write_dot(gcol, f"./ba_{size}_colored.dot")
write_dot(gcolc, f"./ba_{size}_compc.dot")

# %%
