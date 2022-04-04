# %%
import pandas as pd
import numpy as np
import pareto
from trampoline import trampoline


def nds(df, criteria: list, maxmin: list, until_level=None):
    df = df.copy().reset_index(drop=True)
    criteria = list(criteria)
    maxmin = list(maxmin)

    # get the indices of the objective columns
    # 1 is added because it is necessary to include the Index in the original dataframe for later use
    objective_columns = [list(df.columns).index(i) + 1 for i in criteria]

    # pareto.eds minimizes by default, but remember max f(x) == min -f(x)
    # so a change to the sign of the objective column is needed to maximize it instead
    for (o, d) in zip(criteria, maxmin):
        if d == "max":
            df[o] = -df[o]

    acc = []  # dataframe accumulation
    level = 1  # nds level counter
    # until_level = 10
    while len(df) > 0:
        if until_level is not None:
            if level > until_level:
                df["level"] = 1000
                acc.append(df)
                break

        # df.itertuples is set to True, thus including the index
        # Perform nds
        nondominated = pareto.eps_sort([list(df.itertuples(True))], objective_columns)

        df_pareto = pd.DataFrame.from_records(
            nondominated, columns=["Index"] + list(df.columns.values)
        )

        # sort records
        df_pareto = df_pareto.sort_values(
            criteria, ascending=([d == "min" for d in maxmin])
        )
        # tag nondominated points with their level
        df_pareto["level"] = level

        # select the indices from the nondominated points
        in_pareto = list(df_pareto["Index"].unique())
        # remove points from the original dataframe
        df = df[~(df.index.isin(in_pareto))]
        # collect
        acc.append(df_pareto)

        # increment the level by one in preparation for the next iteration
        level += 1
        # if len(df) == 0:
        #     break
    if len(acc) == 0:
        return pd.DataFrame()
    all_pareto = pd.concat(acc)
    all_pareto = all_pareto[
        [
            "repr",
            "level",
            *criteria,
            *[c for c in all_pareto.columns if c not in ["level", "repr", *criteria]],
        ]
    ]
    for (o, d) in zip(criteria, maxmin):
        if d == "max":
            all_pareto[o] = -all_pareto[o]
    all_pareto = all_pareto.drop("Index", axis=1)

    return all_pareto


# previous iteration of nds sorting (recursive with trampolining, but broken because the order of the criteria changes the output, when it shouldn't)
def nds_(df, criteria: list, maxmin: list, until_level=None):
    criteria = list(criteria)
    maxmin = list(maxmin)
    df = df.copy()
    df["pareto"] = False
    df["level"] = 0
    df = df.sort_values(criteria[0], ascending=maxmin[0] == "min")

    def identify_pareto(df_pareto, criteria, ordenar, level):
        df_pareto = df_pareto.copy()

        if len(df_pareto) == 0:
            return df_pareto

        if len(df_pareto) == 1:
            df_pareto["level"] = level
            return df_pareto

        if until_level is not None:
            if level > until_level:
                df_pareto["level"] = None
                return df_pareto

        size = len(df_pareto)
        ids = range(size)
        pareto_front = np.zeros(size, dtype=bool)
        dfp = df_pareto.reset_index().sort_values(
            criteria, ascending=[False if o == "max" else True for o in ordenar]
        )
        [c0, *cr] = criteria
        [o0, *ord] = ordenar
        for (c, o) in zip(cr, ord):
            max_y = -np.inf if o == "max" else np.inf
            for (i, r) in dfp.iterrows():
                if o == "max":
                    if r[c] >= max_y:
                        pareto_front[i] = 1
                        max_y = r[c]
                if o == "min":
                    if r[c] <= max_y:
                        pareto_front[i] = 1
                        max_y = r[c]

        df_pareto["pareto"] = pareto_front

        df_nds = df_pareto.query("`pareto` == True").copy()
        df_nds["level"] = level

        remaining = df_pareto.query("`pareto` == False").copy()

        result = yield identify_pareto(remaining, criteria, ordenar, level + 1)

        return pd.concat([df_nds, result])

    def identify_pareto_t(df, criteria, maxmin, level):
        return trampoline(identify_pareto(df, criteria, maxmin, level))

    return (
        identify_pareto_t(df, criteria, maxmin, 1)
        .drop(["pareto"], axis=1)
        .sort_values(
            ["level", *list(criteria)],
            ascending=tuple([True] + [o == "min" for o in maxmin]),
        )
    )

