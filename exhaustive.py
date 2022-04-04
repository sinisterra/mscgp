# %%
from clickhouse_io import (
    c_evaluate_assoc_measures,
    compute_selector_support,
    get_attributes,
    get_crosstab,
    get_selectors,
)
from multiprocessing import Pool
import itertools
import functools
import pandas as pd
import seaborn as sns
from measures import do_evaluate_rule
from nds import nds

table = "arm.covid"
# %%
# attributes = ["ASMA", "HIPERTENSION", "TABAQUISMO", "DIABETES"]
attributes = get_attributes(table)
attributes = [a for a in attributes if a not in ["PAIS_NACIONALIDAD", "PAIS_ORIGEN"]]

# %%
melted = pd.read_csv("./melted.csv")
melted["itemset"] = melted["itemset"].apply(lambda v: set(eval(v)))

selectors = get_selectors(table)
all_selectors = list(itertools.chain(*selectors.values()))
# del selectors["PAIS_NACIONALIDAD"]
# del selectors["PAIS_ORIGEN"]


@functools.lru_cache(maxsize=None)
def set_support(p):
    return int(melted[melted["itemset"] == set(p)]["support"])


attr_sets = itertools.chain(
    *[itertools.combinations(attributes, n) for n in range(2, 3)]
)


def evaluate_attribute_set(attribute_set):
    set_selectors = [selectors[a] for a in attribute_set]
    acc = []
    for s in itertools.product(*set_selectors):
        s = set(s)
        [an, cn] = list(s)
        for r in [((an,), (cn,)), ((cn,), (an,))]:
            # res = {
            #     "itemset": s,
            #     "rule": r,
            #     **c_evaluate_assoc_measures(table, r),
            #     # **c_evaluate_causal_measures(table, r),
            # }
            # print(res)
            # ev = do_evaluate_rule(table, r)
            # print(r)
            acc.append(r)
        # acc.append({"itemset": s, "support": compute_selector_support(table, s)})
    return acc


def rule_eval(rule):
    ev = do_evaluate_rule(table, rule)
    print(ev["repr"])
    return ev


# pool = Pool()
mapped = list(map(evaluate_attribute_set, attr_sets))
acc = []

acc_r = []
# eval_rules = itertools.chain(*mapped)
# print(len(list(eval_rules)))
# rules_mapped = list(map(rule_eval, mapped))
for m in mapped:
    for e in m:
        acc_r.append(rule_eval(e))

# %%
evald = pd.DataFrame(acc_r)
evald.to_csv("./evald.csv", index=False)
# %

evald2 = evald.query("significant == True")
nds(evald2, criteria=["absolute_risk", "support"], maxmin=["max", "max"],).sort_values(
    "level"
)[
    [
        "repr",
        "level",
        "absolute_risk",
        "relative_risk",
        "full_prevalence",
        "cer",
        "eer",
        "support",
        "full_support",
        "confidence",
    ]
].to_csv(
    "./pareto_evald.csv", index=False
)

# %%
# def evaluate_pair(pair, table):
#     (a1, a2) = pair
#     ct = get_crosstab(table, *pair)
#     acc = []
#     for (s, t) in itertools.product(ct.index, ct.columns):
#         p = set([(a1, s), (a2, t)])
#         acc.append(
#             {"itemset": p, "support": ct.loc[s, t],}
#         )
#     e = pd.DataFrame(acc)
#     print(a1, a2)
#     return e


# pairs = list(itertools.combinations(attributes, 2))

# pool = Pool()
# mapped = pool.map(functools.partial(evaluate_pair, table=table), pairs)
# melted = pd.concat(mapped)
# melted
# #%%
# melted.to_csv("./melted.csv", index=False)

# # %%
# sns.histplot()
