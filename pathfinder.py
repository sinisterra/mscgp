from itertools import product
import math
import networkx as nx


def pathfinder(gr, r):
    if r == 0:
        return gr
    nodes = gr.nodes()

    pfnet_graph = nx.Graph()

    w = {}
    for (i, j) in product(nodes, nodes):
        edge = gr.get_edge_data(i, j, None)
        if edge is not None:
            w[(i, j)] = 1 / edge["weight"]
        else:
            w[(i, j)] = math.inf
    # for k in nodes:
    #     for (i, j) in product(nodes, nodes):
    d = w.copy()

    for (i, j, k) in product(nodes, nodes, nodes):
        dij = d.get((i, j))
        dik = d.get((i, k))
        dkj = d.get((k, j))

        d[(i, j)] = min(dij, ((dik ** r) + (dkj ** r)) ** (1 / r))

    for (i, j) in product(nodes, nodes):
        if d[(i, j)] == w[(i, j)]:
            if w[(i, j)] != math.inf:
                pfnet_graph.add_edge(
                    i,
                    j,
                    **{
                        "weight": round(1 / w[(i, j)], 3),
                        "label": round(1 / w[(i, j)], 3),
                    },
                )

    pfnet_graph.add_nodes_from(gr.nodes(data=True))
    return pfnet_graph
