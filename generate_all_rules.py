# %%
import itertools
from clickhouse_io import get_selectors
from multiprocessing import Pool
from measures import do_evaluate_rule
import pandas as pd

table = "arm.covid"
selectors = {}
antecedent = [""]
consequent = [
    "HIPERTENSION",
    "DIABETES",
    # "CLASIFICACION_FINAL",
    # "EPOC",
    # "NEUMONIA",
    # "TABAQUISMO",
    # "ASMA",
    "CARDIOVASCULAR",
    # "INMUSUPR",
    "OBESIDAD",
    # "RENAL_CRONICA",
    # "TABAQUISMO",
]

attrs = get_selectors(table)

for e in consequent:
    attrs[e] = [(a, b) for (a, b) in attrs[e] if b in [2]]


antecedents_with_none = itertools.product(*[attrs[a] + ["none"] for a in antecedent])

antecedents = [
    s if "none" not in s else tuple(filter(lambda v: v != "none", s))
    for s in antecedents_with_none
]
antecedents = [a for a in antecedents if a != ()]
antecedents

consequents_with_none = itertools.product(*[attrs[a] + ["none"] for a in consequent])

consequents = [
    s if "none" not in s else tuple(filter(lambda v: v != "none", s))
    for s in consequents_with_none
]
consequents = [a for a in consequents if a != ()]
consequents
# %%
rules_d = itertools.product(antecedents, consequents)
# rules_r = itertools.product(consequents, antecedents)


def eval_rule(rule):
    print(rule)
    return do_evaluate_rule(table, rule)


pool = Pool()
mapped = pool.map(eval_rule, itertools.chain(rules_d,))
evald = pd.DataFrame(mapped)

evald
print(evald)

# %%
evald.to_csv("all_group.csv", index=False)
