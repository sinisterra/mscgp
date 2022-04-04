# %%
import pandas as pd
from nds import nds
from rule_repr import repr_selectors


file_key = "1645740575"

df = (
    pd.read_csv(f"./results/{file_key}/0/0/all.csv")
    .query("generation == 100")
    .query("significant == True")
    .reset_index(drop=True)
)

df["rule"] = df["rule"].apply(lambda r: eval(r))

# %%
df = nds(df, ["absolute_risk", "absolute_risk_abs"], ["max", "max"])
measure = "level"
optimize = "min"
df = df.sort_values(measure, ascending=True).reset_index(drop=True)

# %%
pairs = {}
rules = {}
selector_set = []
set_values = {}
# g = nx.Graph()
selector_weights = {}
X = set()
available_selectors = set()
Y = {}
hset = {}
weights = {}
for (i, r) in df.iterrows():
    (a, c) = r["rule"]
    selectors = tuple(set([*a, *c]))
    selector_name = repr_selectors(selectors)
    set_values[selectors] = r[measure]
    selector_set.append(set(selectors))
    X = X.union(set(selectors))
    Y[selector_name] = list(set(selectors))
    for s in selectors:
        hset[s] = hset.get(s, []) + [i]
    weights[i] = r[measure]
    # print(selector_name, r[measure])
    # g.add_node(selector_name, row=r)
    selector_weights[selectors] = r[measure]

sets = list(hset.values())
sets

# %%
from pysat.examples.hitman import Hitman

max_aptitude = float("inf")
best_hs = None
at_most = 1e6
counter = 0
with Hitman(bootstrap_with=sets, htype="sorted") as hitman:

    for hs in hitman.enumerate():
        summation = round(sum([weights[i] for i in hs]), 4)
        if summation < max_aptitude:
            max_aptitude = summation
            best_hs = hs
            print(counter, summation, hs, len(hs))
            df.iloc[best_hs].to_csv(f"./best_hs.csv", index=False)

        counter += 1
        if counter > at_most:
            break

(max_aptitude, best_hs)

# %%

# %%
