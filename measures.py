from dsl import Context, ConfusionMatrix
from rule_repr import repr_rule
from clickhouse_io import (
    c_evaluate_assoc_measures,
    c_get_confusion_matrix,
    compute_selector_support,
)
from functools import lru_cache
import redis
from io_helpers import apply_rounding
import json
import numpy as np
from scipy.stats import chi2_contingency, gmean
import time
import math
import pandas as pd
import itertools
from tabulate import tabulate
from rule_repr import repr_selectors

from wave import OLA

np.seterr(all="ignore")
r = redis.StrictRedis(host="localhost", port=6379, db=0)


def to_redis(alias, d):
    # try:
    res = r.set(str(alias), json.dumps(d))


# except Exception:
#     print(f"Redis write exception, could not persist {alias}")
# if res == True:
#     print(f"{alias} cached")


def from_redis_exists(alias):
    return r.exists(f"{alias}")


def from_redis(alias):
    # data = r.get(str(alias))
    # print("Hitting cache...")
    try:
        return json.loads(r.get(alias).decode("utf-8"))
    except:
        print("No data")
        return None


def evaluate_assoc_measures(table, rule):
    return c_evaluate_assoc_measures(table, rule)


def odds_significance(cm: ConfusionMatrix):
    tp = cm.tp + 1
    tn = cm.tn + 1
    fp = cm.fp + 1
    fn = cm.fn + 1

    odds = (tp / fp) / (fn / tn)

    odds_root = (sum([1 / c for c in [tp, fp, fn, tn]])) ** (1 / 2)
    odds_log = np.log(odds)

    odds_lower_bound = np.exp(odds_log - (1.96 * odds_root))
    # odds_upper_bound = np.exp(odds_log + (1.96 * odds_root))

    odds_significant = odds_lower_bound > 1

    return (odds, odds_significant)


def evaluate_confusion_matrix(cm: ConfusionMatrix):

    # print(cm)

    # Destructure confusion matrix
    tp = cm.tp + (1 if cm.tp == 0 else 0)
    fp = cm.fp + (1 if cm.fp == 0 else 0)
    tn = cm.tn + (1 if cm.tn == 0 else 0)
    fn = cm.fn + (1 if cm.fn == 0 else 0)
    total = tp + tn + fp + fn

    pp = tp + fp
    pn = tn + fn

    p = tp + fn
    n = fp + tn

    tpr = tp / p
    fnr = fn / p
    tnr = tn / n
    fpr = fp / n

    prevalence = p / (p + n)
    bias = pp / (p + n)

    accuracy = (tp + tn) / (p + n)
    balanced_accuracy = (tpr + tnr) / 2

    ppv = tp / pp
    _for = fn / pn
    fdr = fp / pp
    npv = tn / pn

    f1s = (2 * ppv * tpr) / (ppv + tpr)
    fmi = np.sqrt(ppv * tpr)

    lrp = tpr / fpr
    lrm = fnr / tnr

    informedness = tpr + tnr - 1
    markedness = ppv + npv - 1

    dor = lrp / lrm
    ts = tp / (tp + fn + fp)

    determinant = (tp / total * tn / total) - (fp / total * fn / total)

    mcc = np.sqrt(tpr * tnr * ppv * npv) - np.sqrt(fnr * fpr * _for * fdr)
    # prev_threshold = (np.sqrt(tpr * fpr) - fpr) / (tpr - fpr)

    # Event rates
    # Experimental event rate
    eer = tp / (tp + fp)
    # Control event rate
    cer = fn / (fn + tn)

    # Compute risk
    absolute_risk = eer - cer
    relative_risk = eer / cer

    # Calculate odds and its significance
    (odds, odds_significant) = odds_significance(cm)

    pchange_p = lrp / (1 + lrp)
    pchange_n = lrm / (1 + lrm)

    cm_array = [[tp, fn], [fp, tn]]
    (chi2, p, dof, ex) = chi2_contingency(cm_array, correction=True)
    # cramer = association(cm_array, method="cramer")
    significant = p < 0.001 and odds_significant

    consistent = (tp >= fp) and (tn >= fn)
    strong = significant and consistent

    nnt = math.ceil(1 / markedness) if markedness > 0 else -1000

    # if nnt > :
    #     inp = [round(cm.used_factor * i) for i in [markedness, cer, (1 - eer)]]

    #     inp = [1 if i == 0 else i for i in inp]
    #     ap_result = ap.Apportion(seats=nnt, populations=inp, method="webster",)
    #     print(nnt, list(zip(inp, ap_result.fair_shares)))
    p_pct = (tp + fn) / total
    n_pct = (fp + tn) / total
    tp_pct = tp / total
    tn_pct = tn / total
    fn_pct = fn / total

    pp_pct = (tp + fp) / total
    pn_pct = (tn + fn) / total

    pt = ((1 - tnr) ** 0.5) / ((tpr ** 0.5) + ((1 - tnr) ** 0.5))

    plr = tpr / (1 - tnr)
    # root = (plr) ** 0.5
    # upper = pt * root * (prevalence - 1)
    # lower = pt * ((pt * root) - 1)
    # upper_log = np.log(upper / lower)
    # lower_log = np.log(plr)
    # n_i = upper_log / lower_log
    n_i = np.ceil(1 / np.log(plr))

    values = {
        **cm.__dict__,
        # "tp": tp,
        # "tn": tn,
        # "fp": fp,
        # "fn": fn,
        "markedness": markedness,
        "af_e": 0 if eer <= 0 else (eer - cer) / eer,
        "susceptibility": markedness / (1 - cer),
        "disablement": (pp_pct * markedness) / (prevalence),
        "enablement": (pn_pct * markedness) / (1 - prevalence),
        "n_i": 1000 if n_i < 0 else n_i,
        "prevalence_threshold": pt,
        "prevalence_threshold_diff": abs(pt - prevalence),
        "nnt": nnt,
        "nnt_intervention": round(markedness * nnt),
        "nnt_placebo": round(markedness * cer),
        "nnt_nonresponse": round(markedness * (1 - eer)),
        "tp_pct": tp / total,
        "fn_pct": fn / total,
        "fp_pct": fp / total,
        "tn_pct": tn / total,
        "p_pct": p_pct,
        "n_pct": n_pct,
        "tpr": tpr,
        "tnr": tnr,
        "ppv": ppv,
        "npv": npv,
        "informedness": informedness,
        "mcc": mcc,
        "strong": strong,
        "significant": significant,
        "fnr": fnr,
        "fpr": fpr,
        "determinant": determinant,
        "odds_significant": odds_significant,
        "absolute_risk": absolute_risk,
        "cer_total": cm.used_factor - tn,
        "eer_total": cm.used_factor - fp,
        "absolute_risk_abs": tp - fn,
        "absolute_risk_rev": tpr - fpr,
        "relative_risk": relative_risk,
        "cer": cer,
        "eer": eer,
        "odds": odds,
        "dor": dor,
        "prevalence": prevalence,
        "bias": bias,
        "evenness": (prevalence * (1 - prevalence)) ** 0.5,
        "bias_mean": (bias * (1 - bias)) ** 0.5,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1s,
        "threat_score": ts,
        "lr+": lrp,
        "lr-": lrm,
        "pchange+": pchange_p,
        "pchange-": pchange_n,
        "wr_acc": 4 * (tpr - bias) * prevalence,
    }

    for (k, v) in values.items():
        if type(v) == np.int64:
            values[k] = int(v)

        if type(v) == np.bool_:
            values[k] = bool(v)

    return apply_rounding(values, 4)


@lru_cache(maxsize=None)
def evaluate_causal_measures(table, rule):
    (cm, full_cm) = c_get_confusion_matrix(table, rule)

    cm_eval = evaluate_confusion_matrix(cm)
    full_cm_eval = evaluate_confusion_matrix(full_cm)
    renamed_full_cm_eval = {}
    for (k, v) in full_cm_eval.items():
        renamed_full_cm_eval[f"full_{k}"] = v

    return {**cm_eval, **renamed_full_cm_eval}


def rule_redundancies(table, rule, rule_support):
    (a, b) = rule
    sels = set([*a, *b])

    subsets = [
        (compute_selector_support(table, subset), tuple(sels.difference(subset)),)
        for subset in itertools.combinations([*a, *b], len(sels) - 1)
    ]
    is_redundant = [s[1] for s in subsets if s[0] == rule_support and s[0] > 0]
    return {
        "has_redundant_selectors": len(is_redundant) > 0,
        "redundant_selectors": tuple(is_redundant),
        "repr_redundant_sels": ""
        if len(is_redundant) == 0
        else "".join([repr_selectors(e) for e in is_redundant]),
    }


def apply_evaluation(table, rule):
    (a, b) = rule
    assoc = evaluate_assoc_measures(table, rule)
    causal = evaluate_causal_measures(table, rule)

    paf = (
        0
        if causal["relative_risk"] < 0
        else (assoc["antecedent_support"] * (causal["relative_risk"] - 1))
        / (1 + ((causal["relative_risk"] - 1) * assoc["antecedent_support"]))
    )

    evaluation = apply_rounding(
        {
            "repr": repr_rule(rule),
            "paf": paf,
            "paf_pop": causal["absolute_risk_abs"] / causal["full_total"],
            # **rule_redundancies(table, rule, assoc["full_support"]),
            **causal,
            **assoc,
            "rule": rule,
            "length": len(rule[0]) + len(rule[1]),
        },
        4,
    )

    return evaluation


@lru_cache(maxsize=None)
def do_evaluate_rule(table, rule):
    alias = str(OLA) + repr_rule(rule)
    start = time.time()
    if from_redis_exists(alias):
        return from_redis(alias)

    else:
        end = time.time()
        # print(f"{round(end - start, 4)}s\t{repr_rule(rule)}")
        evaluation = apply_evaluation(table, rule)
        to_redis(alias, evaluation)
        return evaluation


@lru_cache(maxsize=None)
def evaluate_rule(ctx: Context, rule):
    (a, b) = rule
    ev = do_evaluate_rule(ctx.dataframe, rule)
    rev = do_evaluate_rule(ctx.dataframe, (b, a))

    # ar = ev["absolute_risk"]
    # r_ar = rev["absolute_risk"]

    acc = {**ev}

    for (k, v) in rev.items():
        acc[f"r_{k}"] = v

    aptitude = (
        -1
        if any(
            [
                m <= 0.0000001
                for m in [
                    acc["absolute_risk_abs"],
                    # acc["r_absolute_risk_abs"],
                    ev["full_support"],
                    *[acc[v] for v in ctx.aptitude_fn],
                ]
            ]
        )
        else gmean(
            [
                acc[m] if o == "max" else (1 / (1 + acc[m]))
                for (m, o) in zip(ctx.aptitude_fn, ctx.optimize)
            ]
        )
    )

    aptitude = gmean([acc[v] for v in ctx.aptitude_fn])

    # non_redundant = True
    # # verificar que el soporte cambie para todos los conjuntos

    # if len(is_redundant) > 0:
    #     print(acc["repr"], "".join(is_redundant))
    # # print(is_redundant if len(is_redundant) > 0 else )

    return apply_rounding(
        {
            **acc,
            # "redundant_selectors": "".join(is_redundant),
            # "has_redundant_selectors": all(
            #     [rule_support != s[0] and s[0] > 0 for s in subsets]
            # ),
            "aptitude": aptitude
            # "aptitude": ((1 / (1 + acc["prevalence_threshold_diff"])) * acc["paf"])
            # ** (0.5)
            # "aptitude": aptitude,
            # "aptitude": 1
            # / (1 + abs(acc["full_prevalence"] - acc["full_prevalence_threshold"]))
            # "is_n_closed": is_n_closed(ctx, rule),
            # "aptitude": -(
            #     (sum([(1 - acc[m]) ** 2 for m in ctx.measures]))
            #     ** (1 / len(ctx.aptitude_fn))
            # ),
            # "repr": ev["repr"] if ar > r_ar else rev["repr"],
        },
        4,
    )


def power_set(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


@lru_cache(maxsize=None)
def is_n_closed(ctx: Context, rule):
    (antecedent, consequent) = rule

    rule_eval = do_evaluate_rule(ctx.dataframe, rule)

    if not rule_eval["significant"]:
        return False

    if len(antecedent) == 1 and rule_eval["significant"]:
        return True

    rule_causal_effect = rule_eval["absolute_risk"]
    antecedent_subsets = itertools.combinations(antecedent, len(antecedent) - 1)

    acc = []

    for antecedent_set in antecedent_subsets:
        if len(antecedent_set) == 0 or len(antecedent_set) == len(antecedent):
            continue

        subset_rule = (tuple(antecedent_set), consequent)
        ev = do_evaluate_rule(ctx.dataframe, subset_rule)
        s_causal_effect = ev["absolute_risk"]

        if not ev["significant"] or not ev["absolute_risk"] > 0:
            continue

        gain = rule_causal_effect - s_causal_effect
        if gain <= 0.01:
            return False

        acc.append(
            {
                "rule": rule_eval["repr"],
                "repr": repr_rule(subset_rule),
                "reference_effect": rule_causal_effect,
                "rule_effect": s_causal_effect,
                "gain": gain,
            }
        )
    # if len(acc) > 0:
    #     print(tabulate(pd.DataFrame(acc), headers="keys", showindex=False))
    return True
