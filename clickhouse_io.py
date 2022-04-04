from numpy import c_
import pandas as pd
from clickhouse_driver import Client
from rule_repr import repr_rule
import functools
from dsl import ConfusionMatrix
import numpy as np
import itertools

np.seterr(divide="ignore")


def client():
    return Client(host="localhost")


def get_attributes(table):
    """
    Returns a list of attributes for the given table
    """
    # res = client().execute(f"DESCRIBE TABLE {table}")
    c_client = client()
    attrs = [c for (c, *_) in c_client.execute(f"DESCRIBE TABLE {table}")]
    c_client.disconnect()
    return sorted(attrs)


def get_crosstab(table, a1, a2, filtered=""):
    query = f"SELECT count(*) as cnt, {a1}, {a2} FROM {table}{filtered if filtered else ''} GROUP BY {a1}, {a2}"
    cl = client()
    result = cl.execute(query)
    df = pd.DataFrame(result, columns=["count", a1, a2])
    ct = df.pivot_table(index=a1, columns=[a2], values="count", aggfunc="sum").fillna(0)
    ct = ct.astype(int)
    cl.disconnect()
    return ct


def get_selectors(table):
    attributes = get_attributes(table)
    selectors = {}
    cl = client()
    for a in attributes:
        a_selectors = cl.execute(f"SELECT DISTINCT {a} FROM {table}")
        a_selectors = [(a, v) for (v, *_) in a_selectors]
        selectors[a] = a_selectors

    cl.disconnect()
    return selectors


def total_records(table):
    cl = client()
    res = cl.execute(f"SELECT count(*) as cnt from {table}")[0][0]
    cl.disconnect()
    return res


@functools.lru_cache(maxsize=None)
def _compute_selector_support(table, selectors):
    selectors = " AND ".join([f"{a} == '{v}'" for (a, v) in selectors])
    cl = client()
    result = cl.execute(f"SELECT count(*) as cnt FROM {table} WHERE {selectors} ")
    support = result[0][0]
    cl.disconnect()
    return support


def compute_selector_support(table, selectors):
    return _compute_selector_support(table, tuple(selectors))


def compute_certainty(support, confidence):
    if support == 1:
        return confidence
    if confidence > support:
        return (confidence - support) / (1 - support)
    if confidence > support:
        return (confidence - support) / (support)

    return 0


@functools.lru_cache(maxsize=None)
def query_rule(table, attributes, filtered=""):
    str_attributes = ", ".join(attributes)

    query = f"SELECT count(*) as count, {str_attributes} FROM {table} {filtered if filtered else ''} GROUP BY {str_attributes} ORDER BY count DESC"
    cl = client()
    res = cl.execute(query)
    qt = pd.DataFrame(res, columns=["count", *attributes])
    cl.disconnect()
    return qt


def build_confusion_matrix(ct, do_normalization=False):

    exp_size = int(
        ((ct.loc[True, True]) if True in ct.index and True in ct.columns else 0)
        + ((ct.loc[True, False]) if True in ct.index and False in ct.columns else 0)
    )

    control_size = int(
        ((ct.loc[False, True]) if False in ct.index and True in ct.columns else 0)
        + ((ct.loc[False, False]) if False in ct.index and False in ct.columns else 0)
    )

    if do_normalization:
        ct = ct.div(ct.sum(axis=1), axis=0)
        ct = ct * min(control_size, exp_size)
        for c in ct.columns:
            ct[c] = np.around(ct[c]).astype(int)
        # ct = ct.astype(int)

    # original = ct.copy()

    tp = (ct.loc[True, True]) if True in ct.index and True in ct.columns else 0
    fp = (ct.loc[True, False]) if True in ct.index and False in ct.columns else 0
    fn = (ct.loc[False, True]) if False in ct.index and True in ct.columns else 0
    tn = (ct.loc[False, False]) if False in ct.index and False in ct.columns else 0

    total = sum([tp, tn, fp, fn])

    cm = ConfusionMatrix(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        total=total,
        control_size=control_size,
        exp_size=exp_size,
        norm_factor="full"
        if do_normalization == False
        else ("exp" if min(exp_size, control_size) == exp_size else "control"),
        used_factor=min(exp_size, control_size),
    )

    return cm


def c_get_confusion_matrix(table, rule, filtered=""):
    (antecedent, consequent) = rule

    selectors = sorted([a for (a, _) in antecedent] + [a for (a, _) in consequent])

    attributes = query_rule(table, tuple(selectors), filtered)
    attributes["treatment"] = attributes.eval(
        " and ".join(
            [
                f"{a} == {v}" if type(v) == int or type(v) == float else f"{a} == '{v}'"
                for (a, v) in antecedent
            ]
        )
    )
    attributes["outcome"] = attributes.eval(
        " and ".join(
            [
                f"{a} == {v}" if type(v) == int or type(v) == float else f"{a} == '{v}'"
                for (a, v) in consequent
            ]
        )
    )

    ct = attributes.pivot_table(
        index="treatment", columns="outcome", values="count", aggfunc="sum"
    ).fillna(0)
    ct = ct.astype(int)
    full_ct = ct.copy()

    return (
        build_confusion_matrix(ct, do_normalization=True),
        build_confusion_matrix(full_ct),
    )


def c_evaluate_assoc_measures(table, rule):

    (antecedent, consequent) = rule
    selectors = [*antecedent, *consequent]
    N = total_records(table)
    a_s = compute_selector_support(table, antecedent)
    c_s = compute_selector_support(table, consequent)
    rs = compute_selector_support(table, selectors)
    support = rs / N
    antecedent_support = a_s / N
    consequent_support = c_s / N
    independent_support = antecedent_support * consequent_support
    lift = 0 if independent_support == 0 else support / independent_support
    confidence = 0 if support == 0 else round(support / antecedent_support, 3)
    certainty = compute_certainty(support, confidence)
    interestingness = (
        -1 if (a_s == 0 or c_s == 0) else (rs / a_s) * (rs / c_s) * (1 - (rs / N))
    )

    return {
        "size": N,
        "support": support,
        "confidence": confidence,
        "lift": lift,
        "antecedent_support": antecedent_support,
        "consequent_support": consequent_support,
        "independent_support": independent_support,
        "antecedent_size": a_s,
        "consequent_size": c_s,
        "certainty": certainty,
        "full_support": int(support * N),
        "interestingness": interestingness,
        "length": len(selectors),
        "antecedent_length": len(antecedent),
        "consequent_length": len(consequent),
    }

