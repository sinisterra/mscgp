from pickletools import optimize
from dsl import Context
import networkx as nx
import itertools
import numpy as np
import pandas as pd
from nds import nds
from rule_repr import repr_selectors
from networkx.drawing.nx_pydot import write_dot
from pysat.examples.hitman import Hitman
from tabulate import tabulate
import math
from scipy.stats import poisson, gmean


def solve(X, Y, solution=[]):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()


def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def do_diversity_op(ctx: Context, state, population):

    df = population.copy().query("tp > 0 and significant == True")

    is_mono = len(ctx.measures) == 1
    if is_mono:
        df = df.sort_values(ctx.measures[0], ascending=(ctx.optimize[0] == "min",))
    else:
        df = nds(df, criteria=ctx.measures, maxmin=ctx.optimize).sort_values(
            ["level", "aptitude", "length"], ascending=(True, False, False)
        )

    measure = ctx.measures[0] if len(ctx.measures) == 1 else "level"
    direction = "min" if len(ctx.measures) > 1 else ctx.optimize[0]

    selector_set = []
    set_values = {}
    g = nx.Graph()
    selector_weights = {}
    selector_lengths = {}
    aptitude_weights = {}
    hset = {}
    weights = {}
    rules = {}
    X = set()
    Y = {}
    compare = min if not is_mono else (min if ctx.optimize[0] == "min" else max)

    antecedents = {}
    consequents = {}
    selector_sets = {}

    for (i, r) in df.iterrows():
        (a, c) = r["rule"]

        selectors = tuple(
            set(
                [
                    *(a if ctx.cover_mode in ["a", "all"] else []),
                    *(c if ctx.cover_mode in ["c", "all"] else []),
                ]
            )
        )
        selector_name = repr_selectors(selectors)

        set_values[selectors] = r[measure]
        selector_set.append(set(selectors))

        current_weight = selector_weights.get(selector_name)
        entry_weight = r[measure]
        next_weight = compare(entry_weight, current_weight or entry_weight)
        selector_weights[selector_name] = next_weight
        selector_lengths[selector_name] = len(selectors)

        if current_weight is None or next_weight != current_weight:
            selector_sets[selector_name] = set(selectors)
            antecedents[selector_name] = set(a)
            consequents[selector_name] = set(c)
            aptitude_weights[selector_name] = r["aptitude"]
            g.add_node(selector_name, row=r)

        # if current_weight is None:
        #     g.add_node(selector_name, row=r)
        #     selector_weights[selector_name] = compare(r[measure], current_weight)
        # else:

        # if current_weight is None:
        #     g.add_node(selector_name, row=r)

        # if next_weight != current_weight:
        #     g.add_node(selector_name, row=r)

        X = X.union(set(selectors))
        Y[i] = list(set(selectors))

        weights[i] = r[measure]
        rules[i] = r
        for s in selectors:
            hset[s] = hset.get(s, []) + [i]

    selector_list = itertools.combinations(enumerate(selector_set), 2)
    g_empty_intersections = nx.Graph()
    for ((i1, s1), (i2, s2)) in selector_list:
        if len(s1.intersection(s2)) == 0:

            g_empty_intersections.add_node(
                i1, label=selector_weights[repr_selectors(s1)]
            )
            g_empty_intersections.add_node(
                i2, label=selector_weights[repr_selectors(s2)]
            )
            g_empty_intersections.add_edge(i1, i2)

    for (a, b) in itertools.combinations(selector_set, 2):
        intr = a.intersection(b)
        if len(intr) > 0:
            g.add_edge(
                repr_selectors(tuple(a)),
                repr_selectors(tuple(b)),
                intr=intr,
                label=repr_selectors(tuple(intr)),
            )

    gc = nx.complement(g)

    # realizar un ordenamiento no-dominado de
    # optimizar la medida , maximizar la longitud del consecuente,
    # if not is_mono:

    for (e, weight) in selector_weights.items():
        e_antecedents = antecedents.get(e)
        e_consequents = consequents.get(e)
        aptitude_weight = aptitude_weights.get(e)
        length = selector_lengths.get(e)
        # length_weight = 1 - (1 / length)
        # length_weight = poisson.cdf(length, mu=2)
        # length_weight = poisson.cdf(length, mu=2)
        length_weight = 1

        # weight = weight_as_int

        # make the weight large if it's a minimization problem

        if not is_mono:
            # print(weight, (1 / (1 + math.exp(-(1 / weight)))))
            # exp_weight = 1 / (1 + math.exp(-1 / weight))
            length_weight = length
            # arc_tan_weight = 1 - (np.arctan(weight) / (np.pi / 2))
            # antecedent_weight = 1 - (1 / (1 + len(e_antecedents)))
            antecedent_weight = 1
            assigned_weight = int(
                1000 * (1 / weight) * antecedent_weight * aptitude_weight
            )
            # arc_tan_length_weight = np.arctan(length) / (np.pi / 2)
            # print((length, length_weight), (weight, arc_tan_weight))
            # assigned_weight = 1000 * (1 / weight)
        else:
            # length_weight = poisson.cdf(length, mu=4)
            # consequents_weight = (0.2 if len(e_consequents) == 1 else 0) + (
            #     1 - (1 / len(e_consequents))
            # )
            consequents_weight = 1
            # print(
            #     len(e_consequents),
            #     (np.arctan(len(e_consequents)) / (np.pi / 2)),
            #     consequents_weight,
            # )
            weight = int(weight * 1000)
            assigned_weight = int(1000000 - weight) if direction == "min" else weight
            assigned_weight = int(assigned_weight)

        gc.add_node(
            e, weight=assigned_weight,
        )

    # X = {j: set() for j in X}
    # for i in Y:
    #     for j in Y[i]:
    #         X[j].add(i)

    best_cover = None
    # sols = list(solve(X, Y))
    # best_cover_val = float("-inf") if compare == max else float("inf")
    # cover_size = 0
    # for s in sols:
    #     cover_df = pd.DataFrame([rules[i] for i in s])
    #     val = cover_df[measure].sum()
    #     next_val = compare(val, best_cover_val)
    #     if val == next_val:
    #         best_cover = cover_df
    #         best_cover_val = val
    #         cover_size = len(s)

    # if best_cover is not None:
    #     print(len(sols), best_cover_val, cover_size)
    #     best_cover.to_csv("./exact_cover.csv", index=False)
    # sets = list(hset.values())
    # max_aptitude = float("-inf")
    # best_hs = None
    # at_most = 10e3
    # counter = 0
    # min_len = float("-inf")
    # diversity_best = None
    # with Hitman(bootstrap_with=sets, htype="sorted") as hitman:
    #     for hs in hitman.enumerate():
    #         # if len(hs) > min_len and min_len != float("-inf"):
    #         #     break
    #         summation = round(sum([weights[i] for i in hs]), 4)
    #         if summation > max_aptitude:
    #             max_aptitude = summation
    #             best_hs = hs

    #             print(counter, summation, hs, len(hs))
    #             diversity_best = pd.DataFrame([rules[h] for h in best_hs])

    #             diversity_best.to_csv(f"./best_hs.csv", index=False)

    #             min_len = len(hs)
    #         counter += 1
    #         if counter > at_most:
    #             break
    elems = []
    ws = []
    while len(ws) < 1:
        (it_elems, w) = nx.max_weight_clique(
            gc.subgraph([e for e in gc.nodes if e not in elems])
        )
        ws += [w]
        elems += it_elems
        if len(it_elems) == 0:
            break
    # print(ws)

    # (elems, w) = nx.max_weight_clique(gc)
    # (elems2, w2) = nx.max_weight_clique(
    #     gc.subgraph([e for e in gc.nodes if e not in elems])
    # )
    # elems = [*elems, *elems2]
    # (elems3, w3) = nx.max_weight_clique(
    #     gc.subgraph([e for e in gc.nodes if e not in elems])
    # )
    # elems = [*elems, *elems3]
    # print(w1, w2, w3)

    if len(elems) > 0:
        write_dot(gc.subgraph(elems), "./max_weight_clique.dot")
        dfsel = pd.DataFrame([g.nodes[e]["row"] for e in elems])

        g_clique = nx.Graph()
        for (a, b) in itertools.combinations(elems, 2):
            g_clique.add_edge(a, b)

        # write_dot(g_clique, "./empty_intersection.dot")
        # si es mono-objetivo, ordenar por la medida de

        if is_mono:
            dfsel = dfsel.sort_values(by=[measure], ascending=(direction == "min"))
        else:
            dfsel = dfsel.sort_values(
                by=["level", "aptitude", *ctx.measures],
                ascending=(True, False, *[o == "min" for o in ctx.optimize]),
            )

        # print(round(dfsel[measure].sum(), 4))
        # print(
        #     tabulate(
        #         dfsel[["repr", *ctx.measures, *(["level"] if not is_mono else [])]],
        #         headers="keys",
        #         tablefmt="psql",
        #     )
        # )
        # dfsel.to_csv("./clique_selection.csv", index=False)

        return pd.concat(
            [dfsel, *([best_cover] if best_cover is not None else [])]
        ).drop_duplicates("repr")
    else:
        print("No max weight clique found")
        return pd.DataFrame()

