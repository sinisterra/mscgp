# %%
import itertools
from functools import lru_cache
from itertools import combinations
from multiprocessing import Pool

import pandas as pd

from clickhouse_io import get_attributes, get_selectors
from measures import do_evaluate_rule
from rule_repr import repr_rule, repr_selectors

table = "arm.lucas"
all_selectors = get_selectors(table)
keys = list(all_selectors.keys())
# del all_selectors["PAIS_ORIGEN"]
# del all_selectors["PAIS_NACIONALIDAD"]
# del all_selectors["ENTIDAD_NAC"]
del all_selectors["index"]
# options = ["ENTIDAD_RES", "ENTIDAD_UM", "DEFUNCION", "TIPO_PACIENTE"]
# for k in keys:
#     if k not in options:
#         del all_selectors[k]


def partition(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        # put `first` in its own subset
        yield [[first]] + smaller


def produce_rules(combination):
    assoc_rules = []
    (a_name, b_name) = combination

    # for (a_name, c_name) in itertools.combinations(all_selectors.keys(), 2):
    a_sels = all_selectors[a_name]
    b_sels = all_selectors[b_name]
    #    c_sels = all_selectors[c_name]

    for (a, b) in itertools.product(a_sels, b_sels,):
        for p in partition([a, b,]):
            if len(p) == 2:
                if len(p[0]) > 0 and len(p[0]) > 0:
                    ant = tuple(p[0])
                    cons = tuple(p[1])
                    # print(repr_rule((ant, cons)))
                    assoc_rules.append((ant, cons))
                    assoc_rules.append((cons, ant))
        # print(a, b, c)
        # for (l, r) in itertools.combinations([a, b, c], 2):
        #     pass
        # print(combination, l, r)

    return assoc_rules


def apply_evaluation(inp):
    (i, rule) = inp
    print(i, repr_rule(rule))
    sels = set([rule[0][0][0], rule[1][0][0]])
    return {
        **do_evaluate_rule(table, rule),
        "attributes": sels,
        "antecedent": repr_selectors(rule[0]),
        "consequent": repr_selectors(rule[1]),
    }


pool = Pool()
mapped = list(
    pool.map(produce_rules, itertools.combinations(list(all_selectors.keys()), 2))
)
all_rules = list(itertools.chain(*mapped))
rule_count = len(all_rules)
print(rule_count)

# %%
pool = Pool()
rules_mapped = list(pool.map(apply_evaluation, enumerate(all_rules)))
df_rules = pd.DataFrame(rules_mapped)
df_rules

# %%
print("Writing...")
df_rules.to_pickle("./all_rules.pkl")
df_rules.query(
    "tp > 0 and fp > 0 and tn > 0 and fn > 0 and significant == True"
).to_csv("./sig_rules_l.csv", chunksize=rule_count * 0.1, index=False)
df_rules.to_csv("./rules_l.csv", chunksize=rule_count * 0.1, index=False)
