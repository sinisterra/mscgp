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
COVER_MODE = "c"
APTITUDE_FN = ("susceptibility", "af_e", "absolute_risk")
MEASURES = (
    *APTITUDE_FN,
    # "absolute_risk",
    # "eer"
    # "paf",
    # "absolute_risk",
    # "aptitude",
    # "susceptibility",
)
# OPTIMIZE = tuple(["max" for _ in MEASURES])
OPTIMIZE = ("max", "max", "max")
# OPTIMIZE = (
#     # "min",
#     "max",
#     "max",
# )
WORKERS = 6
SEED = 0
TOTAL_RUNS = 1
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
(semantic_groups, sg_pairs) = semantic_group_discovery(
    dataset,
    assoc_threshold=CRAMER_THRESHOLD,
    pfnet_threshold=PATHFINDER_R,
    p_value_threshold=P_VALUE_THRESHOLD,
    workers=6,
)
# sg_pairs = [(j, i) for (i, j) in sg_pairs]

# pd.DataFrame.from_dict(semantic_groups, orient="index").to_csv(
#     f"{run_path}/groups.csv", index=False
# )
# print(semantic_groups)
# print(sg_pairs)
# pcas = pd.read_csv("./transformed.csv")
# grouping = "label_6"

# for (i, v) in enumerate(pcas[grouping].unique()):
#     g = pcas[pcas[grouping] == v]["source"].unique()
#     semantic_groups[i] = set(g)

# all v all
# semantic_groups = {
#     # 5: ["oxidative", "hds", "temperatura", "funcion_de_bondad", "Mo", "Al2O3"],
#     # 4: ["time", "O_S", "CoMo", "ionic_liquid", "Ni"],
#     0: ["Mo", "Co", "CoMo", "W", "Fe", "V", "Ni"],
#     1: ["oxidative", "extractive", "adsorptive", "porous"]
#     # 3: ["POM", "per_S", "Co", "W", "reuse"],
#     # 2: ["carbon_cat", "MOF", "Tisop", "extractive", "Ti_cat", "score_ox"],
#     # 1: ["HPM", "MO", "adsorptive", "Fe", "MoHet"],
#     # 0: ["porous", "complex_org", "V", "mat_d", "carbon", "SiO2"],
# }

# semantic_groups = {
#     0: ["W", "Fe", "V", "Co", "Mo", "Ni", "CoMo", "Al2O3", "hds"],
#     1: ["time", "temperatura", "reuse", "funcion_de_bondad", "oxidative", "O_S"],
# }

# semantic_groups = {
#     0: ["coffee", "tea", "instant_coffee"],
#     1: ["semi_finished_bread", "white_bread", "brown_bread",],
# }
# semantic_groups = {
#     5: ["oxidative", "hds", "temperatura", "funcion_de_bondad", "Mo", "Al2O3"],
#     4: ["time", "O_S", "CoMo", "ionic_liquid", "Ni"],
# }

# semantic_groups = {
#     5: ["oxidative", "hds", "temperatura", "funcion_de_bondad"],
#     4: ["Mo"],
#     3: ["Al2O3", "time", "O_S", "CoMo", "ionic_liquid", "Ni"],
#     2: ["POM", "per_S", "Co", "W", "reuse", "carbon_cat", "MOF", "Tisop", "extractive"],
#     1: ["Ti_cat", "score_ox"],
#     0: [
#         "HPM",
#         "MO",
#         "adsorptive",
#         "Fe",
#         "MoHet",
#         "porous",
#         "complex_org",
#         "V",
#         "mat_d",
#         "carbon",
#         "SiO2",
#     ],
# }

g_diseases = {
    # "EDAD",
    # "SEXO",
    "HIPERTENSION",
    "ASMA",
    "CARDIOVASCULAR",
    "DIABETES",
    "HIPERTENSION",
    "INMUSUPR",
    "OBESIDAD",
    "OTRA_COM",
    "OTRO_CASO",
    "RENAL_CRONICA",
    "TABAQUISMO",
    # "EMBARAZO",
    "NEUMONIA",
    "EPOC",
    "CLASIFICACION_FINAL"
    # "EDAD",
    # "SEXO",
}
g_intervals = {
    # "EPOCH_FECHA_SINTOMAS",
    # "EPOCH_FECHA_DEF",
    # "EPOCH_FECHA_INGRESO",
    "INTERVAL_FECHA_SINTOMAS_FECHA_INGRESO",
    "INTERVAL_FECHA_INGRESO_FECHA_DEF",
    "INTERVAL_FECHA_SINTOMAS_FECHA_DEF",
}
g_age_sex = {"EDAD", "SEXO"}
g_location = {
    "ENTIDAD_UM",
    # "ENTIDAD_NAC",
    # "ENTIDAD_RES",
    # "ATN_MISMA_ENTIDAD",
}
g_tests = {
    "TOMA_MUESTRA_ANTIGENO",
    "TOMA_MUESTRA_LAB",
    "RESULTADO_LAB",
    "RESULTADO_ANTIGENO",
    "CLASIFICACION_FINAL"
    # "DEFUNCION",
}
g_demographics = {
    # "INDIGENA",
    # "HABLA_LENGUA_INDIG",
    # "MIGRANTE",
    # "NACIONALIDAD",
    "EDAD",
    "SEXO",
}

g_seriousness = {
    "DEFUNCION",
    "INTUBADO",
    "UCI",
    "TIPO_PACIENTE",
    # "CLASIFICACION_FINAL",
}

g_health_facility = {
    "SECTOR",
    # "ORIGEN"
}

g_all = (
    g_age_sex.union(g_intervals)
    .union(g_location)
    .union(g_tests)
    .union(g_seriousness)
    .union(g_health_facility)
    .union(g_diseases)
)

g_all_c = {"NACIONALIDAD"}

# sg_pairs = [
#     (
#         {"Anxiety", "Peer_Pressure", "Genetics", "Allergy",},
#         {
#             "Smoking",
#             "Lung_cancer",
#             "Car_Accident",
#             "Yellow_Fingers",
#             "Coughing",
#             "Fatigue",
#             "Attention_Disorder",
#         },
#     )
# ]

# sg_pairs = [
#     # (g_demographics, g_diseases),
#     (
#         {
#             "HIPERTENSION",
#             "DIABETES",
#             "NEUMONIA"
#             # "SECTOR",
#             # "INTERVAL_FECHA_SINTOMAS_FECHA_INGRESO",
#             # "ENTIDAD_UM",
#             # "ENTIDAD_RES",
#             # "ATN_MISMA_ENTIDAD",
#         },
#         {
#             "DEFUNCION",
#             "VARIANTE"
#             # "UCI",
#             # "TOMA_MUESTRA_LAB",
#             # "TOMA_MUESTRA_ANTIGENO",
#             # "TIPO_PACIENTE",
#             # "INTUBADO",
#             # "INTERVAL_FECHA_SINTOMAS_FECHA_DEF",
#             # "DEFUNCION",
#             # "EDAD",
#             # "DEFUNCION",
#             # "EPOCH_FECHA_DEF",
#             # "INTERVAL_FECHA_INGRESO_FECHA_DEF",
#         },
#     ),
#     # (g_location, g_tests,),
# ]
# sg_pairs += [(j, i) for (i, j) in sg_pairs]

# yes_no_variables = [*sg_pairs[0][0], *sg_pairs[0][1]]
# sg_pairs = [(semantic_groups[0], semantic_groups[1])]

# (semantic_groups, sg_pairs) = semantic_group_discovery(
#     dataset,
#     assoc_threshold=CRAMER_THRESHOLD,
#     pfnet_threshold=PATHFINDER_R,
#     p_value_threshold=P_VALUE_THRESHOLD,
#     workers=6,
# )
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
        dataframe=dataset,
        exec_run_path=exec_run_path,
        covariates=(),
        pop_size=50,
        stop_condition=("n_gen", 30),
        omit=(),
        antecedent=(1, len(a)),
        consequent=(1, len(b)),
        measures=MEASURES,
        optimize=OPTIMIZE,
        cover_mode=COVER_MODE,
        use_groups=True,
        groups=(tuple(a), tuple(b),),
        aptitude_fn=APTITUDE_FN,
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
            (("ATN_MISMA_ENTIDAD",), ("keep", (2,))),
            (("INTUBADO", "UCI"), ("keep", (1,))),
            (("DEFUNCION",), ("keep", (1,))),
            (("CLASIFICACION_FINAL",), ("keep", (3, 6, 7))),
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
seeds = random.sample(range(1, 100), TOTAL_RUNS)
run_inputs = itertools.product(enumerate(seeds), enumerate([p for p in sg_pairs]))

pool = Pool(6)
mapped = pool.map(functools.partial(do_run, constants=base_constants), run_inputs)

df_finals = pd.concat(mapped)

cols = [
    "repr",
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
    "r_confidence",
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
    "sg_pair",
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
