# %%
import plotly.graph_objects as go

import pandas as pd
from tabulate import tabulate
from itertools import combinations

# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
# from mlxtend.frequent_patterns import association_rules
from nds import nds
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import statistics
import itertools
from nds import nds
import plotly.express as px
import time
import os

# scenario, problem, wave, result
results = [
    ("A", 1, 1, 1655217855),
    ("A", 1, 2, 1655217863),
    ("A", 1, 3, 1655219735),
    ("A", 1, 4, 1655219743),
    ("A", 1, "A", 1651764528),
    ("A", 2, 1, 1655222834),
    ("A", 2, 2, 1655217863),
    ("A", 2, 3, 1655224550),
    ("A", 2, 4, 1655224558),
    ("A", 2, "A", 1652364172),
    ("A", 3, 1, 1655226857),
    ("A", 3, 2, 1655226865),
    ("A", 3, 3, 1655228535),
    ("A", 3, 4, 1655228548),
    ("A", 3, "A", 1651765192),
    ("B", 1, 1, 1655168055),
    ("B", 1, 2, 1655168077),
    ("B", 1, 3, 1655168088),
    ("B", 1, 4, 1655168110),
    ("B", 1, "A", 1651765572),
    ("B", 2, 1, 1655150123),
    ("B", 2, 2, 1655152261),
    ("B", 2, 3, 1655152276),
    ("B", 2, 4, 1655151204),
    ("B", 2, "A", 1651766446),
    ("B", 3, 1, 1655172283),
    ("B", 3, 2, 1655172534),
    ("B", 3, 3, 1655173704),
    ("B", 3, 4, 1655173716),
    ("B", 3, "A", 1651767095),
    ("C", 1, 1, 1655311418),
    ("C", 1, 2, 1655311561),
    ("C", 1, 3, 1655313015),
    ("C", 1, 4, 1655313025),
    ("C", 1, "A", 1651767416),
    ("C", 2, 1, 1655321871),
    ("C", 2, 2, 1655325162),
    ("C", 2, 3, 1655326597),
    ("C", 2, 4, 1655328783),
    ("C", 2, "A", 1651768182),
    ("C", 3, 1, 1655329234),
    ("C", 3, 2, 1655329871),
    ("C", 3, 3, 1655330529),
    ("C", 3, 4, 1655331494),
    ("C", 3, "A", 1651768875),
]

acc_waves = []
for (sc, problem, wave, result) in results:
    exp = f"{sc}{problem}"
    result_key = f"{sc}{problem}{wave}"
    # exp_fns = experiment_fns[problem]

    df_result = (
        pd.read_csv(f"./results/{result}/finals.csv")
        .query("level == 1")
        .drop_duplicates(subset=["repr"])
    )
    df_result["scenario"] = sc
    df_result["wave"] = wave
    df_result["problem"] = problem
    df_result = df_result[
        [
            "repr",
            "scenario",
            "wave",
            "problem",
            "support",
            "full_support",
            "confidence",
            "lift",
            "absolute_risk",
            "r_absolute_risk",
            "susceptibility",
            "paf",
        ]
    ]
    outdir = f"./waves/wave_{wave}/"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    acc_waves.append(df_result)
    df_result.to_csv(f"{outdir}wave{wave}_{exp}.csv", index=False)

pd.concat(acc_waves).to_csv("./all_wave_results.csv", index=False)
print("Wave result concat done")

by_experiment = {}
experiment_fns = {
    1: ["full_support", "confidence", "lift"],
    2: ["absolute_risk", "r_absolute_risk"],
    3: ["susceptibility", "paf"],
}


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


pareto_lengths = []
pvts_by_problem = {}

for (sc, problem, wave, result) in results:
    exp = f"{sc}{problem}"
    result_key = f"{sc}{problem}{wave}"
    exp_fns = experiment_fns[problem]

    df_result = pd.read_csv(f"./results/{result}/finals.csv")
    df_result["scenario"] = sc
    df_result["wave"] = wave
    df_result["problem"] = problem

    # if wave != "A":
    for ((level, seed), pareto_front) in df_result.query("level == 1").groupby(
        ["level", "seed"]
    ):
        pareto_lengths.append(
            {
                "scenario": sc,
                "problem": problem,
                "wave": wave,
                "nds_length": len(pareto_front),
            }
        )

    df_pareto = df_result.query("level == 1").drop_duplicates(subset=["repr"])
    df_pareto = df_pareto[["repr", *exp_fns]]
    df_pareto["result_key"] = result_key
    df_pareto["wave"] = wave
    by_experiment[exp] = by_experiment.get(exp, []) + [df_pareto]

plengths = pd.DataFrame(pareto_lengths)
acc_rows = {}
for ((scenario, problem, wave), elems) in plengths.groupby(
    ["scenario", "problem", "wave"]
):
    scp_key = f"{scenario}{problem}"
    dsc = dict(elems["nds_length"].describe())
    for k in ["mean", "std"]:
        measure_key = f"wave_{wave}_{k}"
        row = acc_rows.get(scp_key, {"llave": scp_key})
        row[measure_key] = dsc[k]
        acc_rows[scp_key] = row


print(
    tabulate(
        pd.DataFrame(acc_rows.values()).fillna(0).round(3),
        headers="keys",
        showindex="never",
        tablefmt="github",
    )
)

plengths.to_csv("./wave_pareto_lengths.csv", index=False)

rule_repeats = []
pvt_acc = {}

for (k, v) in by_experiment.items():
    experiment_comparison = []
    all_exp_results = pd.concat(v)
    rules = {}
    scenario = k[0]
    problem = k[1]
    for (exp_key, res) in all_exp_results.groupby(["result_key"]):
        rules[exp_key] = res["repr"].unique()

    rules_in_all_waves = []
    vals_rules = []
    freqs = []
    pvts = []
    for (repr, g_repr) in all_exp_results.groupby(["repr"]):
        l = len(g_repr)
        freqs.append(l)
        if l == 5:
            exp_ms = experiment_fns[int(problem)]
            pvts.append(g_repr.pivot(index="repr", columns="wave", values=exp_ms))

            d = g_repr.describe().loc[["mean", "std"], :]
            d = pd.melt(d.reset_index(), id_vars="index")
            d["rule"] = repr
            vals_rules.append(d)
            # print(d)
            r_all_waves = {
                "experiment": k,
                "rule": repr,
                "count": l,
                # "mean": d["mean"],
                # "std": d["mean"],
            }
            for m in exp_ms:
                r_all_waves[f"{m}_mean"] = g_repr[m].mean()
                r_all_waves[f"{m}_std"] = g_repr[m].std()

            rules_in_all_waves.append(r_all_waves)

    if len(pvts) > 0:
        df_pvts = pd.concat(pvts).round(3)
        df_pvts.columns = ["_".join([str(c) for c in col]) for col in df_pvts.columns]
        df_pvts["scenario"] = scenario
        df_pvts["problem"] = problem
        df_pvts["experiment"] = f"{scenario}{problem}"
        pvt_acc[problem] = pvt_acc.get(problem, []) + [df_pvts]
        # print(tabulate(df_pvts, headers="keys"))

    d = dict(pd.Series(freqs).value_counts())
    d["experiment"] = k
    print(k, "\n", d)
    rule_repeats.append(d)
    print(
        tabulate(
            pd.DataFrame(rules_in_all_waves).round(3), headers="keys", showindex="never"
        )
    )

    if len(vals_rules) > 0:
        repeats = pd.concat(vals_rules)
        maxes = []
        mins = []
        for ((index, variable), g) in repeats.groupby(["index", "variable"]):

            maxes.append(g[g["value"] == g["value"].max()])
            mins.append(g[g["value"] == g["value"].min()])
            # maxes.append(g.iloc[g["value"].idxmin()])

        mx = pd.concat(maxes)
        mx["op"] = "max"
        mn = pd.concat(mins)
        mn["op"] = "min"
        mxmn = pd.concat([mx, mn])
        mxmn["experiment"] = f"{scenario}{problem}"
        print(
            tabulate(
                mxmn.query(
                    "variable != 'full_support' and not (index == 'mean' and op == 'min')"
                ),
                headers="keys",
                showindex="never",
            )
        )

    for (a1, a2) in combinations(rules.keys(), 2):
        s1 = rules[a1]
        s2 = rules[a2]

        experiment_comparison.append(
            {
                "scenario": scenario,
                "problem": problem,
                "w1": a1[2],
                "w2": a2[2],
                "source": a1,
                "target": a2,
                "pair": [a1, a2],
                "jaccard": jaccard(s1, s2),
                "intersection": len(set(s1).intersection(s2)),
            }
        )
    df_exp_comp = pd.DataFrame(experiment_comparison)
    agg = df_exp_comp.pivot(columns="source", index="target")
    similarity_matrix = (agg["jaccard"].fillna(0).round(3) * 100).round(3)
    print(similarity_matrix)

    # g = nx.from_pandas_adjacency(similarity_matrix)
    # print(write_dot(g, f"./scenario{scenario}.dot"))
    # print(agg["intersection"])

    # do arm on the resulting rules
    crosstab = pd.crosstab(all_exp_results["repr"], all_exp_results["result_key"])
    attrs = [c for c in crosstab.columns if c != "repr"]
    crosstab[attrs].to_csv(f"./exp_{k}.csv", index=False)
    # frequent_itemsets = fpgrowth(crosstab[attrs], min_support=0.1, use_colnames=True)
    # rs = association_rules(frequent_itemsets,)
    # if len(rs) == 0:
    #     continue
    # rs["antecedents"] = rs["antecedents"].apply(lambda s: sorted(s))
    # # rs = rs[rs["consequents"].apply(lambda s: {"B2A"} == s)]
    # rs["consequents"] = rs["consequents"].apply(lambda s: sorted(s))
    # rs["repr"] = rs.apply(lambda r: f"{r['antecedents']} -> {r['consequents']}", axis=1)
    # rs["cc"] = rs["confidence"] - rs["consequent support"]
    # rs["cf"] = (rs["confidence"] - rs["consequent support"]) / (
    #     1 - rs["consequent support"]
    # )
    # rs = rs[rs["antecedents"].apply(lambda v: f"{k}A" not in v)]
    # rs = nds(rs, ["confidence", "support"], ["max", "max"]).query("level == 1")
    # print(tabulate(rs, headers="keys",))

df_rule_repeats = pd.DataFrame(rule_repeats)
df_rule_repeats = df_rule_repeats[["experiment", 1, 2, 3, 4, 5]].fillna(0)
df_rule_repeats["Total"] = df_rule_repeats.apply(
    lambda r: sum([r[s] for s in range(1, 5 + 1)]), axis=1
)
df_rule_repeats["Media"] = df_rule_repeats.apply(
    lambda r: statistics.mean(
        list(itertools.chain(*[[s] * int(r[s]) for s in range(1, 5 + 1)]))
    ),
    axis=1,
)

print(
    tabulate(
        df_rule_repeats.round(2), headers="keys", showindex="never", tablefmt="github"
    )
)

for (k, v) in pvt_acc.items():
    l = pd.concat(v).reset_index(drop=False)
    cols = [c for c in l.columns if c not in ["scenario", "problem", "experiment"]]
    cols = ["experiment",] + cols
    l = l[cols]

    print(tabulate(l, headers="keys", showindex="never", tablefmt="github"))

# %%
# import plotly.express as px

# df0 = pd.concat(by_experiment["A2"])
# px.scatter(df0, x="absolute_risk", y="r_absolute_risk", color="wave")


# %%
tbl_nds = {}
paretos_by_wave = {}
for (sc, problem, wave, result) in results:
    exp = f"{sc}{problem}"
    result_key = f"{sc}{problem}{wave}"
    exp_fns = experiment_fns[problem]

    # obtener los frentes de pareto de todas las ejecuciones
    df_result = pd.read_csv(f"./results/{result}/finals.csv").query("level == 1")

    # extraer el frente de pareto entre todos los frentes obtenidos
    df_pareto = nds(df_result, criteria=exp_fns, maxmin=["max" for _ in exp_fns])
    # print(df_pareto["level"].value_counts())
    df_pareto = df_pareto.query("level == 1")
    df_pareto["wave"] = wave
    df_pareto["experiment"] = exp
    tbl_nds[exp] = {**tbl_nds.get(exp, {"Experimento": exp}), wave: len(df_pareto)}
    paretos_by_wave[exp] = paretos_by_wave.get(exp, []) + [df_pareto]

# %%
print(
    tabulate(
        pd.DataFrame(tbl_nds.values()),
        headers="keys",
        showindex="never",
        tablefmt="github",
    )
)

# %%

for w in ["A2", "A2", "B2", "C2", "A3", "B3", "C3"]:
    time.sleep(3)
    df0 = pd.concat(paretos_by_wave[w])
    df0["wave"] = df0["wave"].replace({"A": "T"})
    # df0 = df0.sort_values("wave")
    sc = w[0]
    fns = experiment_fns[int(w[1])]
    fig = px.line(
        df0,
        x=fns[0],
        y=fns[1],
        color="wave",
        hover_data=["repr"],
        # symbol="wave",
        # markers=True,
        labels={
            "wave": "Ola",
            "susceptibility": "Susceptibilidad",
            "paf": "Impacto en la población (AFp)",
            "absolute_risk": "Efecto causal",
            "r_absolute_risk": "Efecto causal (recíproco)",
        },
        title=f"Experimento {w}",
        # range_x=[-0.05, 1.05],
        # range_y=[-0.05, 1.05],
    )
    fig.update_traces(marker={"size": 10, "line": {"width": 1, "color": "black"}})
    # fig.update_layout(
    #     title={
    #         "text": f"Experimento {w}",
    #         "y": 0.9,
    #         "x": 0.1,
    #         "xanchor": "center",
    #         "yanchor": "top",
    #     }
    # )
    fig.write_image(f"./pareto_waves/{w}.pdf")

# %%
for w in ["C1", "B1", "A1"]:

    df0 = pd.concat(paretos_by_wave[w])
    df0["wave"] = df0["wave"].replace({"A": 5})
    # df0 = df0.query("wave == 1")
    max_val = 0
    for d in ["confidence", "support", "lift"]:
        max_val = max(df0[d].max(), max_val)
    labels = {
            "wave": "Ola",
            "support": "Soporte",
            "confidence": "Confianza",
            "lift": "Ascenso",
        }

    # fig = px.parallel_coordinates(
    #     df0,
    #     color="wave",
    #     dimensions=["confidence", "support", "lift",],
    #     title=f"Experimento {w}",
    #     range_color=[0, max_val],
    #     # color_continuous_scale=px.colors.sequential.YlGnBu,
    #     labels={
    #         "wave": "Ola",
    #         "support": "Soporte",
    #         "confidence": "Confianza",
    #         "lift": "Ascenso",
    #     },
    # )
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df0["wave"],
                showscale=True
            ),
            dimensions=list(
                [
                    {"range": [0, max_val], "label": labels[v], "values": df0[v]}
                    for v in ["support", "confidence", "lift"]
                    # dict(
                    #     range=[0, 8],
                    #     constraintrange=[4, 8],
                    #     label="Sepal Length",
                    #     values=df["sepal_length"],
                    # ),
                    # dict(range=[0, 8], label="Sepal Width", values=df["sepal_width"]),
                    # dict(range=[0, 8], label="Petal Length", values=df["petal_length"]),
                    # dict(range=[0, 8], label="Petal Width", values=df["petal_width"]),
                ]
            ),
        )
    )

    fig.write_image(f"./pareto_waves/{w}.pdf")


# %%

