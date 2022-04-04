# %%
import pandas as pd
from clickhouse_io import get_selectors, compute_selector_support
from itertools import permutations, product, combinations
from measures import do_evaluate_rule
from multiprocessing import Pool


table = "arm.quimica"
all_selectors = get_selectors(table)
attributes = all_selectors.keys()

# generar todos los itemsets
def generate_ac(l):
    comb = list(product(["a", "c"], repeat=l))
    return [c for c in comb if "a" in c and "c" in c]


acs = {}

acc = []
for itemset_length in range(4, 4 + 1):
    # if len(acc) >= 1000:
    #     break
    combs = generate_ac(itemset_length)
    for attribute_set in combinations(attributes, itemset_length):
        with_selectors = [all_selectors[s] for s in attribute_set]
        for selector_product in product(*with_selectors):
            # if compute_selector_support(table, selector_product) > 0:
            #     print(selector_product)
            for pos_comb in combs:
                rule_elements = {}
                for (symbol, selector) in zip(pos_comb, selector_product):
                    rule_elements[symbol] = rule_elements.get(symbol, []) + [selector]

                acc.append((tuple(rule_elements["a"]), tuple(rule_elements["c"]),))


def with_table(t):
    (i, r) = t
    evald = do_evaluate_rule(table, r)
    # print(i, evald["repr"])
    return evald


print(len(acc))
pool = Pool(6)
mapped = pool.map(with_table, enumerate(acc))
pd.DataFrame(list(mapped)).query("tp > 0 and significant == True").to_csv(
    "./rules_full.csv", index=False, chunksize=len(acc) // 10
)
# %%
