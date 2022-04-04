from dataclasses import dataclass
import pandas as pd
import numpy as np
from functools import lru_cache
from dsl import Context

# from xor_template import xor_expression


@lru_cache(maxsize=None)
def get_dataframe(ctx: Context):
    df = pd.read_csv(ctx.dataframe).fillna("?")
    return df


@lru_cache(maxsize=None)
def get_levels(ctx: Context, selector):
    (attribute, _) = selector
    df = get_dataframe(ctx)
    qcut = pd.qcut(df[attribute], ctx.levels, labels=True)
    return qcut


@lru_cache(maxsize=None)
def get_true_selector_value(ctx: Context, selector):
    (_, value) = selector
    levels = get_levels(ctx, selector)
    return levels[value]


@lru_cache(maxsize=None)
def make_query(selectors):
    def make_selector_query(attribute, value):

        if type(value) is float and np.isnan(value):
            return f"{attribute} == {attribute}"

        if isinstance(value, str):
            return f"`{attribute}` == '{value}'"

        return f"`{attribute}` == {value}"

    literals = [
        make_selector_query(attribute, value) for (attribute, value) in list(selectors)
    ]

    expression = " and ".join(
        [
            make_selector_query(attribute, value)
            for (attribute, value) in list(selectors)
        ]
    )
    return expression
    # return xor_expression(literals)


def apply_rounding(d, significant_digits=4):
    acc = {}
    ints = ["tp", "tn", "fp", "fn", "exp_size", "control_size", "full_support"]
    full_ints = [f"full_{f}" for f in ints]
    all_elems = ints + full_ints
    for (k, v) in d.items():
        acc[k] = round(v, significant_digits) if isinstance(v, float) else v
        if k in all_elems:
            acc[k] = int(v)

    return acc


def extend_with_treatment_outcome(df, treatment, outcome):
    if len(df) > 0:
        df["treatment"] = df.eval(make_query(treatment))
        if outcome != ():
            df["outcome"] = df.eval(make_query(outcome))

    return df

