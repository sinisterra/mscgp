from diversity_op import do_diversity_op
from filtering_op import do_max_filtering, do_filtering
from io_helpers import apply_rounding
from dsl import Context
from nds import nds
import pandas as pd
from validation_op import do_validation, validate_rule

# restriction_expr = "tp > 0 and significant == True and odds >= 1 and absolute_risk > 0"
restriction_expr = "tp > 0"


def do_selector_elitism(ctx: Context, population):
    # split rules based on their selectors
    by_attributes = {}

    # elite_pop = do_elitism(ctx, population)

    for (i, row) in population.iterrows():
        rule = row["rule"]
        (antecedent, consequent) = rule

        for attribute in [s[0] for s in (*antecedent, *consequent)]:
            by_attributes[attribute] = by_attributes.get(attribute, []) + [row]

    acc = []
    for (k, v) in by_attributes.items():
        level_df = pd.DataFrame(v)

        attribute_nds = nds(level_df, list(ctx.measures), list(ctx.optimize))

        acc.append(attribute_nds)

    df_acc = pd.concat(acc)
    df_acc = (
        df_acc.sort_values("level")
        .drop_duplicates(subset=["repr"])
        .reset_index(drop=True)
    )

    # print(f"NDS sorting kept {len(df_acc)} rules out of {len(population)}")

    return do_elitism(ctx, df_acc.loc[0 : ctx.pop_size, :].copy())


def do_single_elitism(ctx: Context, state, population):
    # (measure, *_) = ctx.measures
    # (direction, *_) = ctx.optimize

    elites = population.copy().drop_duplicates(subset="repr")

    elites["restriction"] = elites.eval(restriction_expr)

    filter_result = do_max_filtering(ctx, state, elites)

    valids = elites.query("restriction == True").sort_values(
        *ctx.measures, ascending=[o == "min" for o in list(ctx.optimize)]
    )
    invalids = elites.query("restriction == False").sort_values(
        *ctx.measures, ascending=[o == "min" for o in list(ctx.optimize)]
    )
    filtered = elites[elites["repr"].isin(filter_result["repr"])].sort_values(
        *ctx.measures, ascending=[o == "min" for o in list(ctx.optimize)]
    )

    diversity = do_diversity_op(ctx, state, population)
    # diversity, filtered
    # elites = pd.concat([valids, invalids]).drop_duplicates(subset="repr")
    elites = pd.concat([diversity, filtered, valids, invalids]).drop_duplicates(
        subset="repr"
    )

    elites = elites.head(ctx.pop_size)

    elites = elites.sort_values(
        ["restriction", *ctx.measures],
        ascending=tuple([False, *[o == "min" for o in list(ctx.optimize)]]),
    )

    # elites["selected_by_filter"] = False
    elites["selected_by_filter"] = elites["repr"].isin(list(filtered["repr"].unique()))

    elites.drop(columns=["restriction"])

    # elites["diversity"] = False
    elites["diversity"] = elites["repr"].isin(list(diversity["repr"].unique()))

    return elites


def do_mo_elitism(ctx: Context, state, population):
    elites = population.copy()

    # evaluate rule validity
    elites["restriction"] = elites.eval(restriction_expr)

    # valids = len(elites.query("restriction == True"))

    valids = elites.query("restriction == True")
    invalids = elites.query("restriction == False")

    top = nds(valids, list(ctx.measures), list(ctx.optimize))
    bottom = nds(invalids, list(ctx.measures), list(ctx.optimize))
    filtered = do_max_filtering(ctx, state, population)
    diversity = do_diversity_op(ctx, state, population)
    # diversity, filtered,
    elites = (
        pd.concat([diversity, filtered, top, bottom,])
        .drop_duplicates(subset="repr")
        .head(ctx.pop_size)
        # .head(ctx.pop_size)
    )
    elites.drop(columns=["restriction"])

    elites = nds(elites, list(ctx.measures), list(ctx.optimize))
    elites["restriction"] = elites.eval(restriction_expr)
    elites["selected_by_filter"] = elites["repr"].isin(list(filtered["repr"].unique()))

    # elites["diversity"] = True
    elites["diversity"] = elites["repr"].isin(list(diversity["repr"].unique()))

    return elites


def do_elitism(ctx: Context, population):
    nds_sorted = nds(population, list(ctx.measures), list(ctx.optimize))

    selection = (
        nds_sorted.sort_values(
            ["level", *ctx.measures],
            ascending=tuple([True, *[o == "min" for o in list(ctx.optimize)]]),
        )
        .reset_index(drop=True)
        .iloc[0 : ctx.pop_size]
        .copy()
    )

    # print(selection[["repr", *ctx.measures]])
    for (m, o) in zip(ctx.measures, ctx.optimize):
        print(o, m, apply_rounding(dict(selection[m].describe()), 4))

    return do_validation(ctx, selection)
