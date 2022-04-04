from networkx.algorithms.operators.unary import reverse
from pathfinder import pathfinder
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
import networkx as nx
import time
from clickhouse_io import get_attributes, get_crosstab

# from itertools import combinations
from networkx.drawing.nx_pydot import write_dot
from itertools import combinations, product
import multiprocessing

from functools import partial, reduce

colors = [
    "#e57373",
    "#7986cb",
    "#fff176",
    "#aed581",
    "#ba68c8",
    "#64b5f6",
    "#ffb300",
    "#4db6ac",
    "#f06292",
    "#9575cd",
    "#ff8a65",
    "#a1887f",
    "#90a4ae",
    "#ffcdd2",
    "#f8bbd0",
    "#e1bee7",
    "#d1c4e9",
    "#c5cae9",
    "#bbdefb",
    "#b3e5fc",
    "#b2ebf2",
    "#b2dfdb",
    "#c8e6c9",
    "#dcedc8",
    "#f0f4c3",
    "#fff9c4",
    "#ffecb3",
    "#ffe0b2",
    "#ffccbc",
    "#d7ccc8",
    "#f5f5f5",
    "#cfd8dc",
]


style_options = {}
style_options[
    "semantic"
] = """node [style="rounded,filled" shape="rect" fontname="IBM Plex Sans"]
edge [fontname="IBM Plex Sans"]
layout="twopi"
overlap=f
outputorder=edgesfirst"""

style_options[
    "network"
] = """node [style="rounded,filled" shape="rect" fontname="IBM Plex Sans"]
edge [fontname="IBM Plex Sans"]
layout="neato"
overlap=f
outputorder=edgesfirst"""


def map_attribute_pairs(map_input, df, p_value_threshold, assoc_threshold, N):
    (i, (a1, a2)) = map_input
    # df = pd.read_pickle(df_path)

    ct = get_crosstab(df, a1, a2)
    (chi2, p, dof, ex) = chi2_contingency(ct, correction=True)
    # k = min(len(ct.columns), len(ct))
    cramer = association(ct.values.tolist(), method="cramer", correction=True)
    # cramer = 0 if k <= 1 else (chi2 / (N * (k - 1))) ** 0.5
    significant = p <= p_value_threshold
    measures = {
        # "chi2": chi2,
        "cramer": cramer,
        "pValue": p,
        "significant": significant,
    }
    elem = {"a1": a1, "a2": a2, **measures}
    print("\t".join((str(s) for s in (i, round(cramer, 3), a1, a2))))

    return (
        # elems
        [{"a1": a1, "a2": a2, **measures}, {"a2": a1, "a1": a2, **measures}],
        # pair_list
        [elem],
    )


def dict_as_values(d):
    acc = {}
    for (k, v) in d.items():
        acc[v] = acc.get(v, []) + [k]
    return acc


def build_attribute_network(df, assoc_threshold=0.1, p_value_threshold=0.1, workers=4):

    columns = get_attributes(df)
    columns = [
        c
        for c in columns
        if c
        not in [
            "index",
            "PAIS_ORIGEN",
            "PAIS_NACIONALIDAD",
            "INTERVAL_FECHA_SINTOMAS_FECHA_DEF",
            "INTERVAL_FECHA_INGRESO_FECHA_DEF",
            "INTERVAL_FECHA_SINTOMAS_FECHA_INGRESO",
        ]
    ]
    pairs = list(combinations(columns, 2))
    # elems = []
    # pair_list = []
    N = len(df)
    print(
        f"Building attribute network, assoc threshold is {assoc_threshold}, p-value <= {p_value_threshold}"
    )
    print(
        f"Evaluating significance of {len(pairs)} pairs ({len(columns)} attributes)..."
    )

    def reduce_attribute_pairs(p1, p2):
        acc = []
        for (a1, a2) in zip(p1, p2):
            acc.append(a1 + a2)
        return acc

    # df.to_pickle("./df.pkl")

    elems = []
    pair_list = []
    start = time.time()

    with multiprocessing.Pool(workers) as pool:
        mapped = pool.map(
            partial(
                map_attribute_pairs,
                df=df,
                N=N,
                p_value_threshold=p_value_threshold,
                assoc_threshold=assoc_threshold,
            ),
            enumerate(pairs),
        )
        (elems, pair_list) = reduce(reduce_attribute_pairs, mapped)
    # for (i, (a1, a2)) in enumerate(pairs):
    #     ct = pd.crosstab(df[a1], df[a2])
    #     (chi2, p, dof, ex) = chi2_contingency(ct, correction=True)
    #     k = min(len(ct.columns), len(ct))
    #     cramer = 0 if k <= 1 else (chi2 / (N * (k - 1))) ** 0.5
    #     significant = p < p_value_threshold and cramer >= assoc_threshold
    #     measures = {
    #         # "chi2": chi2,
    #         "cramer": cramer,
    #         "pValue": p,
    #         "significant": significant,
    #     }
    #     elem = {"a1": a1, "a2": a2, **measures}

    #     elems.append({"a1": a1, "a2": a2, **measures})
    #     elems.append({"a2": a1, "a1": a2, **measures})
    #     pair_list.append(elem)

    end = time.time()
    print(end - start)

    dfelems = pd.DataFrame(elems)
    attrs = set(dfelems["a1"].unique()).union(set(dfelems["a2"].unique()))
    dfelems.round(3).to_csv("./assoc_full.csv")
    dfch2 = (
        dfelems.sort_values("cramer", ascending=False)
        .query(f"significant == True and cramer >= {assoc_threshold}")
        .round(3)
    )
    adj = pd.pivot_table(dfch2, index="a1", columns="a2", values="cramer").fillna(0)

    for c in columns:
        if c not in adj.columns:
            adj[c] = 0

    adj = adj.transpose()
    for c in columns:
        if c not in adj.columns:
            adj[c] = 0

    adj = adj.transpose()
    adj = adj.fillna(0)

    adj.to_csv("./adjacencies.csv")
    # return a networkx instance
    g = nx.from_pandas_adjacency(adj)
    for a in attrs:
        g.add_node(a)

    print(
        f"Adjacency Graph with {len(g.nodes())} attributes and {len(g.edges())} edges."
    )
    return g


def apply_colors(g, clique_cover):
    g = g.copy()
    color_assignation = {}
    clique_assignation = {}
    clique_ids = []
    for node in g.nodes():
        clique_id = clique_cover[node]
        color = colors[clique_id % len(colors)]
        color_assignation[node] = color
        clique_assignation[node] = clique_id
        clique_ids = (
            [*clique_ids, clique_id] if clique_id not in clique_ids else clique_ids
        )

    nx.set_node_attributes(g, color_assignation, name="fillcolor")
    nx.set_node_attributes(g, clique_assignation, name="clique")
    nx.set_edge_attributes(g, nx.get_edge_attributes(g, "weight"), name="label")

    return g


def find_clique_cover(graph):
    print(f"Finding a clique cover for {len(graph.nodes())} attributes...")
    # return dictionary of (group_id, [attributes])
    clique_cover = nx.greedy_color(nx.complement(graph), strategy="DSATUR")

    print(
        f"Clique cover found: size = {1 + max([int(c) for c in clique_cover.values()])}"
    )
    acc_print = []

    as_vals = dict_as_values(clique_cover)
    sorted_as_vals = {}
    for (k, v) in enumerate(sorted(as_vals.values(), key=len, reverse=True)):
        sorted_as_vals[k] = v
        acc_print.append({"Group ID": k, "Attributes": v, "Group Size": len(v)})
    df_acc_print = pd.DataFrame(acc_print).sort_values("Group Size", ascending=False)
    print(df_acc_print)
    df_acc_print.to_csv(f"./sg_groups.csv", index=False)

    return (sorted_as_vals, clique_cover)


def draw_semantic_map(g, pfg, dict_cc, clique_cover):
    g = apply_colors(g.copy(), clique_cover)
    pfg = apply_colors(pfg.copy(), clique_cover)
    clique_pairs = combinations(list(dict_cc.keys()), 2)
    hgraph = nx.Graph()

    write_dot(g, "./graph.dot")
    include_styles("./graph.dot", "network")

    write_dot(pfg, "./scaling.dot")
    include_styles("./scaling.dot", "network")

    for (s, t) in clique_pairs:
        a1 = dict_cc[s]
        a2 = dict_cc[t]

        edges_in_pair = 0
        node_edge_set = set()
        for a, b in product(a1, a2):
            if pfg.has_edge(a, b):
                node_edge_set = node_edge_set.union(set([a, b]))
                edges_in_pair += 1

        if edges_in_pair > 0:
            edges_in_pair = edges_in_pair
            # hgraph.add_edge(s, t, label=len(node_edge_set), weight=len(node_edge_set))
            hgraph.add_edge(s, t, weight=len(node_edge_set))

    sgs = hgraph.copy()
    write_dot(sgs, "./intersections.dot")
    sgs.add_node("_", fontcolor="transparent", color="transparent")

    sgs.add_nodes_from(g.nodes(data=True))
    for (k, nodes) in dict_cc.items():
        sgs.add_node(k)
        sgs.add_edge("_", k, color="transparent")
        for n in nodes:
            sgs.add_node(n)
            sgs.add_edge(k, n)
    write_dot(sgs, "./semantic.dot")
    include_styles("./semantic.dot", "semantic")


def find_group_pairs(groups, graph, pfnet_r=1):
    before_pfg = []
    all_sgs = []
    print(f"Simplifying the network, PFNET r = {pfnet_r}")
    pfg = nx.maximum_spanning_tree(graph)
    print(
        f"Simplification done, {len(pfg.edges())} out of {len(graph.edges())} edges were conserved ({len(graph.edges()) - len(pfg.edges())} removed)"
    )

    bfg_links = []
    after_bfg_links = []
    sgs_to_mine = []
    for (a1, a2) in combinations(list(groups.keys()), 2):

        v_a1 = set(groups[a1])
        v_a2 = set(groups[a2])

        for (e1, e2) in product(v_a1, v_a2):

            if graph.has_edge(e1, e2):
                bfg_links.append((a1, a2))

            if pfg.has_edge(e1, e2):
                all_sgs.append((set(v_a1), set(v_a2)))
                all_sgs.append((set(v_a2), set(v_a1)))
                after_bfg_links.append((a1, a2,))
                sgs_to_mine.append({"Group 1": v_a1, "Group 2": v_a2})
                sgs_to_mine.append({"Group 1": v_a2, "Group 2": v_a1})

                break

    print(
        f"Links after simplification: {len(set(after_bfg_links))} out of {len(set(bfg_links))} links remain. ({len(set(bfg_links)) - len(set(after_bfg_links))} were removed)."
    )

    print("\n\n==========")
    print(f"Pairs to mine: {len(all_sgs)}")
    print("==========\n\n")
    print(pd.DataFrame(sgs_to_mine))
    print("==========\n\n")
    pd.DataFrame(sgs_to_mine).to_csv("./sgs_to_mine.csv",)
    return (all_sgs, pfg)


def include_styles(output, kind):
    styles = style_options.get(kind)
    with open(output, "r") as f:
        content = f.read()
        content_list = content.splitlines()
        [head, *rest] = content_list
        with open(output, "w") as fw:
            for e in [head, styles, *rest]:
                fw.write(e + "\n")
        f.close()


def semantic_group_discovery(
    dataset,
    assoc_threshold=0,
    pfnet_threshold=10,
    p_value_threshold=0.000001,
    workers=4,
):
    print("Semantic group discovery...")

    graph = build_attribute_network(
        dataset,
        assoc_threshold=assoc_threshold,
        p_value_threshold=p_value_threshold,
        workers=workers,
    )

    (groups, clique_cover) = find_clique_cover(graph)

    (group_pairs, pfnet) = find_group_pairs(groups, graph, pfnet_r=pfnet_threshold)

    print("Drawing semantic map...")
    draw_semantic_map(graph, pfnet, groups, clique_cover)

    return (groups, group_pairs)
