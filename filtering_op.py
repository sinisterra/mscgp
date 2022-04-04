from dsl import Context
from nds import nds
from functools import lru_cache
import itertools
import pandas as pd
import numpy as np
from matching.games import HospitalResident
import termplotlib as tpl


@lru_cache(None)
def format_selector(s):
    (a, v) = s
    return f"[{a}={v}]"


def do_filtering(ctx: Context, state, population):
    # Step 1 -> Ordering. If mono-objective, then sort by fitness. If multi-objective, then do non-dominated sorting

    df = population.copy().query("tp >= 1 and significant == True")
    selectors = state.selectors

    is_mono = len(ctx.measures) == 1
    if is_mono:
        df = df.sort_values(ctx.measures[0], ascending=(ctx.optimize[0] == "min"))
    else:
        df = nds(df, criteria=ctx.measures, maxmin=ctx.optimize)

    sg0_sels = []
    sg1_sels = []
    sg0 = ctx.groups[0]
    sg1 = ctx.groups[1]
    for sl in sg0:
        sg0_sels += [format_selector(s) for s in selectors[sl]]

    for sl in sg1:
        sg1_sels += [format_selector(s) for s in selectors[sl]]

    m = "level"
    direction = "min"

    pairs = []
    for (i, r) in df.iterrows():
        [antecedent, consequent] = r["rule"]
        for (a, c) in itertools.product(antecedent, consequent):
            pair = {}
            pair["rule"] = r["repr"]
            pair["source"] = format_selector(a)
            pair["target"] = format_selector(c)
            pair[m] = r[m]
            pair["length"] = r["length"]
            pairs.append(pair)

    df_pairs = pd.DataFrame(pairs)
    df_pairs = df_pairs.sort_values(
        by=[m, "length"], ascending=(direction == "min", False)
    )

    # sg0_sels = df_pairs["source"].unique()
    # sg1_sels = df_pairs["target"].unique()

    df_best_pairs = df_pairs.drop_duplicates(subset=["source", "target"])

    sg0_prefs = dict()
    sg1_prefs = dict()
    for sel in sg0_sels:
        ranked = (
            df_best_pairs[df_best_pairs["source"] == sel][m]
            .rank(method="first", ascending=direction == "min")
            .astype(int)
        )
        targets = []
        for (i, rank) in ranked.items():
            elem = df_best_pairs.loc[i, "target"]
            current = sg0_prefs.get(sel, [])
            sg0_prefs[sel] = current + [elem] if elem not in current else current

    for sel in sg1_sels:
        ranked = (
            df_best_pairs[df_best_pairs["target"] == sel][m]
            .rank(method="first", ascending=direction == "min")
            .astype(int)
        )
        targets = []
        for (i, rank) in ranked.items():
            elem = df_best_pairs.loc[i, "source"]
            current = sg1_prefs.get(sel, [])
            sg1_prefs[sel] = current + [elem] if elem not in current else current

    sg0_in_prefs = list(sg0_prefs.keys())
    sg1_in_prefs = list(sg1_prefs.keys())

    # print(sg0_prefs)
    # print(sg1_prefs)

    sg0_preferred_targets = np.ceil(
        df_pairs[df_pairs["source"].isin(sg0_in_prefs)]["target"].value_counts(
            normalize=True
        )
        * len(max(sg0_sels, sg1_sels))
    ).astype(int)

    sg1_preferred_targets = np.ceil(
        df_best_pairs[df_best_pairs["source"].isin(sg1_in_prefs)][
            "target"
        ].value_counts(normalize=True)
        * len(min(sg0_sels, sg1_sels))
    ).astype(int)

    source_bar = dict(
        pd.concat(
            [
                df_best_pairs["source"].value_counts(),
                df_best_pairs["target"].value_counts(),
            ]
        ).sort_index(ascending=True)
    )

    all_sels = [*sg0_sels, *sg1_sels]
    with_freq = sorted(
        zip(all_sels, [source_bar.get(s, 0) for s in all_sels]),
        key=lambda v: v[0],
        reverse=False,
    )
    fig = tpl.figure()
    fig.barh([s[1] for s in with_freq], [s[0] for s in with_freq])
    fig.show()

    # uncomment to make the selection fixed

    # casos: hay más selectores en 0 que en 1 -> cambia la capacidad de 1 para que
    sg1_caps = dict()
    total_sg0_sels = len(sg0_in_prefs)
    total_sg1_sels = len(sg1_in_prefs)

    if True:
        for s in sg1_sels:
            sg1_caps[s] = max(1, int(np.ceil(total_sg0_sels / total_sg1_sels)))
            # sg1_caps[s] = 1

        matched = HospitalResident.create_from_dictionaries(
            sg0_prefs, sg1_prefs, sg1_caps
        ).solve(optimal="hospital")

        matched2 = HospitalResident.create_from_dictionaries(
            sg0_prefs, sg1_prefs, sg1_caps
        ).solve(optimal="resident")

        acc = []
        for (target, ss) in matched.items():
            for (s, t) in itertools.product(ss, [target]):
                r = list(
                    df_best_pairs.query(f"source == '{s}' and target == '{t}'")["rule"]
                )[0]

                selection = df[df["repr"] == str(r)].copy()
                acc.append(selection)

        for (target, ss) in matched2.items():
            for (s, t) in itertools.product(ss, [target]):
                r = list(
                    df_best_pairs.query(f"source == '{s}' and target == '{t}'")["rule"]
                )[0]

                selection = df[df["repr"] == str(r)].copy()
                acc.append(selection)

        selected_rules = (
            pd.concat(acc).drop_duplicates(subset="repr").sort_values("level")
        )

        return selected_rules
    else:
        for s in sg0_sels:
            sg1_caps[s] = max(1, int(np.ceil(total_sg1_sels / total_sg0_sels)))

        matched = HospitalResident.create_from_dictionaries(
            sg1_prefs, sg0_prefs, sg1_caps
        ).solve(optimal="hospital")
        print(matched)
        acc = []
        for (tgts, srs) in matched.items():
            for (s, t) in itertools.product(srs, [tgts]):
                r = list(
                    df_best_pairs.query(f"source == '{s}' and target == '{t}'")["rule"]
                )[0]
                print(s, t, r)
                # r = df_best_pairs[
                #     (df_best_pairs["source"] == s) & (df_best_pairs["target"] == t)
                # ]["rule"]
                selection = df[df["repr"] == str(r)].copy()
                # print(s, t, r, selection.reset_index(drop=True).loc[0, "level"])
                # selection["source"] = s
                # selection["target"] = t
                # print(s, t, r)
                acc.append(selection)

        selected_rules = (
            pd.concat(acc).drop_duplicates(subset="repr").sort_values("level")
        )

        return selected_rules


def do_max_filtering(ctx, state, population):

    df = population.copy().query("tp >= 1 and significant == True")
    selectors = state.selectors

    is_mono = len(ctx.measures) == 1
    if is_mono:
        df = df.sort_values(ctx.measures[0], ascending=(ctx.optimize[0] == "min"))
    else:
        df = nds(df, criteria=ctx.measures, maxmin=ctx.optimize)
    pairs = []

    m = ctx.measures[0] if len(ctx.measures) == 1 else "level"
    direction = "min" if len(ctx.measures) > 1 else ctx.optimize[0]

    for (i, r) in df.iterrows():
        [antecedent, consequent] = r["rule"]
        for (a, c) in itertools.product(antecedent, consequent):
            pair = {}
            pair["rule"] = r["repr"]
            pair["source"] = format_selector(a)
            pair["target"] = format_selector(c)
            pair[m] = r[m]
            pair["length"] = r["aptitude"]
            pairs.append(pair)

    df_pairs = pd.DataFrame(pairs)

    df_pairs = df_pairs.sort_values(
        by=[m, "length"], ascending=(direction == "min", False)
    )

    df_best_pairs = df_pairs.drop_duplicates(subset=["source", "target"])

    source_selections = {}
    target_selections = {}
    is_source = []
    is_target = []
    acc = []
    for t in df_best_pairs["target"].unique():
        # seleccionar el source para el cual se tenga el mayor valor
        s = (
            df_best_pairs[df_best_pairs["target"] == t]
            .sort_values(m, ascending=direction == "min")["source"]
            .reset_index(drop=True)
            .iloc[0]
        )
        r = list(df_best_pairs.query(f"source == '{s}' and target == '{t}'")["rule"])[0]
        selection = df[df["repr"] == str(r)].copy()
        is_target.append(r)
        target_selections[r] = target_selections.get(r, []) + [t]

        acc.append(selection)

    for s in df_best_pairs["source"].unique():
        t = (
            df_best_pairs[df_best_pairs["source"] == s]
            .sort_values(m, ascending=direction == "min")["target"]
            .reset_index(drop=True)
            .iloc[0]
        )
        r = list(df_best_pairs.query(f"source == '{s}' and target == '{t}'")["rule"])[0]
        selection = df[df["repr"] == str(r)].copy()
        # selection["source_selected"] = True
        is_source.append(r)
        source_selections[r] = source_selections.get(r, []) + [s]
        acc.append(selection)

    # for (i, r) in df_best_pairs.iterrows():
    #     s = r["source"]
    #     t = r["target"]

    #     print(s, t, r)

    #     # r = df_best_pairs[
    #     #     (df_best_pairs["source"] == s) & (df_best_pairs["target"] == t)
    #     # ]["rule"]
    #     # print(s, t, r, selection.reset_index(drop=True).loc[0, "level"])
    #     # selection["source"] = s
    #     # selection["target"] = t

    source_bar = dict(
        pd.concat(
            [
                df_best_pairs["source"].value_counts(),
                df_best_pairs["target"].value_counts(),
            ]
        ).sort_index(ascending=True)
    )

    sg0_sels = []
    sg1_sels = []
    sg0 = ctx.groups[0]
    sg1 = ctx.groups[1]
    for sl in sg0:
        sg0_sels += [format_selector(s) for s in selectors[sl]]

    for sl in sg1:
        sg1_sels += [format_selector(s) for s in selectors[sl]]
    # all_sels = [*sg0_sels, *sg1_sels]
    # with_freq = sorted(
    #     zip(all_sels, [source_bar.get(s, 0) for s in all_sels]),
    #     key=lambda v: v[0],
    #     reverse=False,
    # )
    # fig = tpl.figure()
    # fig.barh([s[1] for s in with_freq], [s[0] for s in with_freq])
    # fig.show()

    selected_rules = pd.concat(acc).drop_duplicates(subset="repr")
    selected_rules["source_selected"] = selected_rules["repr"].apply(
        lambda r: r in is_source
    )
    selected_rules["target_selected"] = selected_rules["repr"].apply(
        lambda r: r in is_target
    )
    selected_rules["for_sources"] = selected_rules["repr"].apply(
        lambda r: source_selections.get(r, "")
    )
    selected_rules["for_targets"] = selected_rules["repr"].apply(
        lambda r: target_selections.get(r, "")
    )
    selected_rules["selected_by_filter"] = True

    return selected_rules
