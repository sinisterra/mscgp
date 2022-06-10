import random
import itertools
import functools
from multiprocessing import Pool
from typing import final
from algorithm import run_algorithm
from filtering_op import do_filtering, do_max_filtering
from diversity_op import do_diversity_op
from measures import is_n_closed, rule_redundancies
from semantic_group_discovery import semantic_group_discovery
import time
import os
import pandas as pd
from dsl import Context, AlgorithmState
import numpy
from scipy.stats import gmean
from nds import nds
from tabulate import tabulate


random.seed(42)
numpy.random.seed(42)


dataset = "arm.covid"
CRAMER_THRESHOLD = 0.08
PATHFINDER_R = 50
P_VALUE_THRESHOLD = 0.001
COVER_MODE = "a"
APTITUDE_FN = (
    # "support",
    # "confidence",
    # "lift",
    "absolute_risk",
    "r_absolute_risk",
    # "susceptibility",
    # "paf",
)
MEASURES = (*APTITUDE_FN,)
refs = {"relative_risk": 1000, "lift": 1000}
OPTIMIZE = tuple(["max" for _ in MEASURES])
REFERENCE_POINTS = tuple([refs.get(t, 1) + 0.1 for t in APTITUDE_FN])
# OPTIMIZE = ("max",)
# OPTIMIZE = (
#     # "min",
#     "max",
#     "max",
# )
WORKERS = 6
SEED = 0
TOTAL_RUNS = 5
run_start = int(time.time())
run_path = f"./results/{run_start}"
os.makedirs(run_path, exist_ok=True)


yes_no_variables = [
    # "DEFUNCION",
    # "TOMA_MUESTRA_LAB",
    # "TOMA_MUESTRA_ANTIGENO",
    "NEUMONIA",
    "EMBARAZO",
    "HABLA_LENGUA_INDIG",
    "DIABETES",
    "INDIGENA",
    "EPOC",
    "ASMA",
    "INMUSUPR",
    "HIPERTENSION",
    "OTRA_COM",
    "CARDIOVASCULAR",
    "OBESIDAD",
    "RENAL_CRONICA",
    "TABAQUISMO",
    "OTRO_CASO",
    "MIGRANTE",
    # "DEFUNCION",
]

# yes_no_variables = [
#     "semi_finished_bread",
#     "white_bread",
#     "brown_bread",
#     "instant_coffee",
#     "coffee",
#     "tea",
#     # "Ni",
#     # "Co",
#     # "Mo",
#     # "CoMo",
#     # "hds",
#     # "Al2O3",
#     # "Fe",
#     # "V",
#     # "W",
#     # "oxidative",
# ]

# USE SEMANTIC GROUPS
# (semantic_groups, sg_pairs) = semantic_group_discovery(
#     dataset,
#     assoc_threshold=CRAMER_THRESHOLD,
#     pfnet_threshold=PATHFINDER_R,
#     p_value_threshold=P_VALUE_THRESHOLD,
#     workers=8,
# )

g_location = {"ENTIDAD_UM", "ATN_MISMA_ENTIDAD", "SECTOR"}
g_health_facility = {"INTUBADO", "UCI", "DEFUNCION", "TIPO_PACIENTE"}
g_demographics = {
    "EDAD",
    "SEXO",
}
g_tests = {
    "TOMA_MUESTRA_ANTIGENO",
    "TOMA_MUESTRA_LAB",
    "RESULTADO_ANTIGENO",
    "RESULTADO_LAB",
}
g_diseases = {
    "CLASIFICACION_FINAL",
    "ASMA",
    "CARDIOVASCULAR",
    "DIABETES",
    "EPOC",
    "HIPERTENSION",
    "INMUSUPR",
    "NEUMONIA",
    "OBESIDAD",
    "RENAL_CRONICA",
    "TABAQUISMO",
}
g_a_diseases = {
    "TABAQUISMO",
    "OBESIDAD",
    "RENAL_CRONICA",
    "ASMA",
    "EPOC",
    "DIABETES",
    "INMUSUPR",
    "CLASIFICACION_FINAL",
    "HIPERTENSION",
    "CARDIOVASCULAR",
    "NEUMONIA",
}
g_a_seriousness = {
    "EDAD",
    "UCI",
    "INTUBADO",
    "EPOCH_FECHA_DEF",
    "DEFUNCION",
    "TOMA_MUESTRA_LAB",
    "TOMA_MUESTRA_ANTIGENO",
    "TIPO_PACIENTE",
}
g_a_tests = {
    "RESULTADO_ANTIGENO",
    "RESULTADO_LAB",
    "EPOCH_FECHA_SINTOMAS",
    "VARIANTE",
    "ORIGEN",
    "EPOCH_FECHA_INGRESO",
}

g_a_location = {"ENTIDAD_UM", "ENTIDAD_RES", "ATN_MISMA_ENTIDAD"}

g_a_others = {"NEUMONIA", "OTRA_COM", "INDIGENA", "HABLA_LENGUA_INDIG", "OTRO_CASO"}

g_a_health_facility = {"SECTOR", "NACIONALIDAD", "MIGRANTE"}

g0 = {
    "ASMA",
    "INMUSUPR",
    "TABAQUISMO",
    "OBESIDAD",
    "RENAL_CRONICA",
    "EPOC",
    "CARDIOVASCULAR",
    "DIABETES",
    "HIPERTENSION",
    "CLASIFICACION_FINAL",
}
g1 = {
    "EDAD",
    "EPOCH_FECHA_DEF",
    "DEFUNCION",
    "INTUBADO",
    "TOMA_MUESTRA_ANTIGENO",
    "TOMA_MUESTRA_LAB",
    "UCI",
    "TIPO_PACIENTE",
}
g3 = {
    "ORIGEN",
    "RESULTADO_ANTIGENO",
    "EPOCH_FECHA_INGRESO",
    "EPOCH_FECHA_SINTOMAS",
    "RESULTADO_LAB",
}

sg_pairs = [
    (g_demographics, g_diseases),
    # (g_diseases, g_health_facility),
    # (g_location, g_diseases),
    # (g0, g1),
    # (g0, g3),
    # (g1, g3),
    # (g_health_facility, g_diseases),
    # (
    #     {
    #         "DEFUNCION",
    #         "EDAD",
    #         "EPOCH_FECHA_DEF",
    #         "INTUBADO",
    #         "TIPO_PACIENTE",
    #         "TOMA_MUESTRA_ANTIGENO",
    #         "TOMA_MUESTRA_LAB",
    #         "UCI",
    #     },
    #     {
    #         "RESULTADO_ANTIGENO",
    #         "RESULTADO_LAB",
    #         "ORIGEN",
    #         "EPOCH_FECHA_SINTOMAS",
    #         "EPOCH_FECHA_INGRESO",
    #     },
    # )
    # (g_diseases, g_demographics,)
    # (g_location, g_tests)
    # (g_demographics, g_health_facility),
    # (g_demographics, g_diseases),
    # (g_location, g_health_facility),
    # (g_a_diseases, g_a_seriousness),
    # (g_a_location, g_a_tests),
    # (g_a_others, g_a_seriousness),
]
# acc = []
# for g in sg_pairs:
#     acc += [g, (g[1], g[0])]
# sg_pairs = acc

df_sg_pairs = pd.DataFrame(sg_pairs)
df_sg_pairs.to_csv(f"{run_path}/semantic_group_pairs.csv")

base_constants = {
    "MEASURES": MEASURES,
    "OPTIMIZE": OPTIMIZE,
    "dataset": dataset,
    "run_start": run_start,
    "run_path": run_path,
}


def do_run(args, constants):

    acc = []
    finals = []
    nds_sorted = []
    elites = []
    poph = []

    (seed_data, sgs) = args
    (n_seed, seed) = seed_data
    (cid, (a, b)) = sgs
    i = a
    j = b

    # print(i, j)
    # if not (
    #     (("OBESIDAD" in a) and ("DEFUNCION" in b))
    #     or (("OBESIDAD" in b) and ("DEFUNCION" in a))
    # ):
    #     return pd.DataFrame()

    # random.seed(seed)
    # numpy.random.seed(seed)
    exec_run_path = f"{constants['run_path']}/{n_seed}/{cid}"
    os.makedirs(exec_run_path, exist_ok=True)
    # print(run_path)

    ctx = Context(
        id=cid,
        seed=seed,
        dataframe=dataset,
        exec_run_path=exec_run_path,
        covariates=(),
        pop_size=50,
        stop_condition=("n_gen", 50),
        omit=(),
        antecedent=(1, len(a)),
        consequent=(1, len(b)),
        measures=MEASURES,
        optimize=OPTIMIZE,
        cover_mode=COVER_MODE,
        use_groups=True,
        groups=(tuple(a), tuple(b),),
        aptitude_fn=APTITUDE_FN,
        reference_points=REFERENCE_POINTS,
        selector_restrictions=(
            (tuple(yes_no_variables), ("keep", (1,)),),
            (
                (
                    "EPOCH_FECHA_DEF",
                    "EPOCH_FECHA_SINTOMAS",
                    "EPOCH_FECHA_INGRESO",
                    "INTERVAL_FECHA_SINTOMAS_FECHA_DEF",
                    "INTERVAL_FECHA_INGRESO_FECHA_DEF",
                    "INTERVAL_FECHA_SINTOMAS_FECHA_INGRESO",
                ),
                ("remove", ("?",),),
            ),
            (("RESULTADO_ANTIGENO", "RESULTADO_LAB"), ("remove", (97,))),
            # (("ATN_MISMA_ENTIDAD",), ("keep", (2,))),
            (("INTUBADO", "UCI"), ("keep", (1,))),
            (("DEFUNCION",), ("keep", (1, 2))),
            (("CLASIFICACION_FINAL",), ("keep", (3, 7))),
        ),
    )

    exc: AlgorithmState = run_algorithm(ctx)

    pair_path = exec_run_path

    if exc is not None:

        with open(f"{exec_run_path}/groups.txt", "w") as f:
            f.write(f"{i}\n{j}")
            f.close()

        with open(f"{exec_run_path}/measures.txt", "w") as f:
            f.write(f"MEASURES {ctx.measures}\nOPTIMIZE {ctx.optimize}")
            f.close()

        dfp = exc.history.copy()
        dfp["sg_pair"] = cid
        dfp["a1"] = str(i)
        dfp["a2"] = str(j)
        # dfp["is_n_closed"] = dfp.apply(lambda r: is_n_closed(ctx, r["rule"]), axis=1)
        # dfp["redundant_selectors"] = dfp.apply(
        #     lambda r: rule_redundancies(ctx.dataframe, r["rule"], r["full_support"])[
        #         "repr_redundant_sels"
        #     ],
        #     axis=1,
        # )
        # for (i, r) in list(dfp.iterrows()):
        #     res = rule_redundancies(ctx.dataframe, r["rule"], r["full_support"])
        #     if res["has_redundant_selectors"]:
        #         print(res["repr_redundant_selectors"])
        # for (k, v) in res.items():
        #     if k not in dfp.columns:
        #         dfp[k] = ""
        #     dfp.loc[i, k] = v

        # print(tabulate(dfp[["repr", "absolute_risk", "is_n_closed",]], headers="keys",))

        dfp.to_csv(f"{pair_path}/uniques.csv", index=False)
        df_filtered = do_diversity_op(ctx, exc, do_max_filtering(ctx, exc, dfp))
        df_filtered.to_csv(f"{pair_path}/filtered.csv", index=False)
        acc.append(dfp)
        # nds_pair = nds(dfp, list(ctx.measures), list(ctx.optimize))
        # nds_sorted.append(nds_pair)

        # nds_pair.to_csv(f"{pair_path}/nds.csv", index=False)

        final_population = exc.population.copy()
        # final_population["is_n_closed"] = final_population.apply(
        #     lambda r: is_n_closed(ctx, r["rule"]), axis=1
        # )
        final_population["seed"] = seed
        final_population["sg_pair"] = cid
        final_population["a1"] = str(i)
        final_population["a2"] = str(j)
        # final_population["is_n_closed"] = final_population.apply(
        #     lambda r: is_n_closed(ctx, r["rule"]), axis=1
        # )
        acc_fp = []
        for (i, r) in final_population.iterrows():
            acc_fp.append(
                {**r, **rule_redundancies(ctx.dataframe, r["rule"], r["full_support"],)}
            )
        final_population = pd.DataFrame(acc_fp)
        finals.append(final_population)
        final_population.to_csv(f"{pair_path}/final.csv", index=False)

        df_pop_h = exc.pop_history.copy()
        df_pop_h["sg_pair"] = cid
        df_pop_h["a1"] = str(i)
        df_pop_h["a2"] = str(j)
        # df_pop_h["is_n_closed"] = df_pop_h.apply(
        #     lambda r: is_n_closed(ctx, r["rule"]), axis=1
        # )
        df_pop_h.to_csv(f"{pair_path}/all.csv", index=False)

        df_elites = exc.elites.copy()
        df_elites["sg_pair"] = cid
        df_elites["a1"] = str(i)
        df_elites["a2"] = str(j)
        # df_elites["is_n_closed"] = df_elites.apply(
        #     lambda r: is_n_closed(ctx, r["rule"]), axis=1
        # )
        df_elites.to_csv(f"{pair_path}/elites.csv", index=False)

        poph.append(df_pop_h)
        elites.append(df_elites)

    return final_population
    # return [
    #     dfp,
    #     final_population,
    #     df_elites,
    #     df_pop_h,
    # ]
    # return {
    #     "seed": seed,
    #     "cid": cid,
    #     "sg_a": a,
    #     "sg_b": b,
    #     "elites": pd.DataFrame(),
    #     "history": pd.DataFrame(),
    #     "final": pd.DataFrame(),
    # }


# generate seeds
seeds = range(1, TOTAL_RUNS + 1)
run_inputs = itertools.product(enumerate(seeds), enumerate([p for p in sg_pairs]))

pool = Pool(WORKERS)
mapped = pool.map(functools.partial(do_run, constants=base_constants), run_inputs)

df_finals = pd.concat(mapped)

cols = [
    "repr",
    "sg_pair",
    "seed",
    *(["level"] if len(MEASURES) > 1 else []),
    "aptitude",
    "absolute_risk",
    "r_absolute_risk",
    "af_e",
    "paf",
    "support",
    "confidence",
    "lift",
    "absolute_risk_abs",
    "antecedent_support",
    "consequent_support",
    "relative_risk",
    "susceptibility",
    "paf",
    # "r_confidence",
    # "cer",
    # "r_cer",
    # "tp",
    # "fp",
    # "fn",
    # "tn",
    # "prevalence",
    "full_support",
    # "used_factor",
    "selected_by_filter",
    "diversity",
    "antecedent_size",
    "consequent_size",
    # "is_n_closed",
    # "has_redundant_selectors",
    # "repr_redundant_sels",
]
cols += [e for e in MEASURES if e not in cols]
cols += [e for e in APTITUDE_FN if e not in cols]
# cols += [e for e in ["nec_not_suff", "suff_not_nec", "nec_and_suff"] if e not in cols]
# cols += [
#     e
#     for e in ["high_exp_low_pop", "low_exp_high_pop", "high_exp_high_pop"]
#     if e not in cols
# ]

# df_finals["nec_not_suff"] = df_finals["r_absolute_risk"] / df_finals["absolute_risk"]
# df_finals["suff_not_nec"] = 1 / df_finals["nec_not_suff"]
# df_finals["nec_and_suff"] = df_finals.apply(
#     lambda v: gmean([v["absolute_risk"], v["r_absolute_risk"]]), axis=1
# )
# df_finals["high_exp_low_pop"] = df_finals["af_e"] / df_finals["paf"]
# df_finals["low_exp_high_pop"] = 1 / df_finals["high_exp_low_pop"]
# df_finals["high_exp_high_pop"] = df_finals.apply(
#     lambda v: gmean([v["af_e"], v["paf_total"]]), axis=1
# )
# for e in [
#     "nec_not_suff",
#     "suff_not_nec",
#     "nec_and_suff",
#     "high_exp_low_pop",
#     "low_exp_high_pop",
#     "high_exp_high_pop",
# ]:
#     df_finals[e] = df_finals.apply(
#         lambda v: gmean([v[e], v["aptitude"]]), axis=1
#     ).round(4)

df_finals.to_csv(f"{run_path}/finals.csv", index=False)

selection = df_finals

selection.query("significant == True and tp > 0")[cols].to_csv(
    f"{run_path}/selection.csv", index=False
)
print(f"{run_path}/selection.csv")


# acc = []
# finals = []
# nds_sorted = []
# elites = []
# poph = []
# for l in mapped:
#     for i in l:
#         [dfp, final_population, df_elites, df_pop_h] = l

#         acc.append(dfp)
#         finals.append(final_population)
#         # nds_sorted.append(nds_pair)
#         elites.append(df_elites)
#         poph.append(df_pop_h)

# acc = pd.concat(acc)
# finals = pd.concat(finals)
# nds_sorted = pd.concat(nds_sorted)
# elites = pd.concat(elites)
# poph = pd.concat(poph)


# # %%
# df_elites = elites
# df_elites.to_csv(f"{run_path}/elites.csv", index=False)

# df_all = poph
# df_all.to_csv(f"{run_path}/all.csv", index=False)

# dfacc = df_all.drop_duplicates(subset="repr")
# dfacc.to_csv(f"{run_path}/uniques.csv", index=False)

# df_nds_sorted = nds_sorted
# df_nds_sorted = nds(df_nds_sorted, MEASURES, OPTIMIZE)
# df_nds_sorted.to_csv(f"{run_path}/nds.csv", index=False)
