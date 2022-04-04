from dsl import AlgorithmState, Context
from measures import evaluate_rule
from restrictions import enforce_restrictions
from rule_repr import format_for_population, repr_rule, repr_selectors
import pandas as pd
import random


def do_transitive_crossover(ctx: Context, state):
    population = state.population
    # group by selector
    antecedents = {}
    consequents = {}
    for (_, row) in population.iterrows():
        rule = row["rule"]
        (antecedent, consequent) = rule

        for a in antecedent:
            antecedents[a] = antecedents.get(a, set()).union(set([rule]))

        for c in consequent:
            consequents[c] = consequents.get(c, set()).union(set([rule]))

    crossover_rules = []
    for (_, row) in population.iterrows():
        rule = row["rule"]
        (antecedent, consequent) = rule

        # find all rules starting with this rule's consequent
        rules_starting_with_consequent = set()

        # if a -> c; then find rules where c -> x
        for c in consequent:
            rules_starting_with_consequent = rules_starting_with_consequent.union(
                consequents.get(c, set())
            )

        if len(rules_starting_with_consequent) > 0:
            for rule_option in rules_starting_with_consequent:
                (a, c) = rule_option
                next_rule = (antecedent, c)
                next_rule_eval = evaluate_rule(ctx, next_rule)

                if enforce_restrictions(next_rule_eval):

                    crossover_rules.append(
                        format_for_population(next_rule, next_rule_eval)
                    )
        if len(crossover_rules) == ctx.pop_size:
            break

    return pd.DataFrame(crossover_rules)


def crossover_pair(state, pair):
    (l, r) = pair
    rule1 = state.population.iloc[l]["rule"]
    rule2 = state.population.iloc[r]["rule"]

    (ra1, rc1) = rule1
    (ra2, rc2) = rule2

    nr1 = (ra2, rc1)
    nr2 = (ra1, rc2)

    r1_selectors = [r[0] for r in (*nr1[0], *nr1[1])]
    r2_selectors = [r[0] for r in (*nr2[0], *nr2[1])]

    return [
        *([nr1] if len(r1_selectors) == len(set(r1_selectors)) else []),
        *([nr2] if len(r2_selectors) == len(set(r2_selectors)) else []),
    ]


def do_crossover(ctx: Context, state: AlgorithmState):
    elems = []
    for (_, row) in state.population.iterrows():
        (antecedent, consequent) = row["rule"]

        new_rule = (consequent, antecedent)

        evaluation = evaluate_rule(ctx, new_rule)

        if enforce_restrictions(evaluation):
            elems.append(format_for_population(new_rule, evaluation))

    return pd.DataFrame(elems)


def sample_in_bounds(l, bounds):
    return random.sample(l, k=min(len(l), random.randint(*bounds)))


def do_crossover_(ctx: Context, state):
    pop = state.population["rule"]

    # split list in half
    halfway = len(pop) // 2
    pairs = zip(pop[0:halfway], pop[halfway : len(pop)])
    new_rules = []
    mutated = []

    for (r1, r2) in pairs:
        (a1, c1) = r1
        (a2, c2) = r2
        rules_from_pair = []

        a1sels = set([s[0] for s in a1])
        a2sels = set([s[0] for s in a2])
        c1sels = set([s[0] for s in c1])
        c2sels = set([s[0] for s in c2])

        ant_attrs_union = a1sels.union(a2sels)
        cons_attrs_union = c1sels.union(c2sels)

        ant_attrs_union = sample_in_bounds(ant_attrs_union, ctx.antecedent)
        cons_attrs_union = sample_in_bounds(cons_attrs_union, ctx.consequent)

        ant_attrs_sym = a1sels.symmetric_difference(a2sels)
        cons_attrs_sym = c1sels.symmetric_difference(c2sels)

        ant_attrs_sym = sample_in_bounds(ant_attrs_sym, ctx.antecedent)
        cons_attrs_sym = sample_in_bounds(cons_attrs_sym, ctx.consequent)

        # ant_attrs_int = a1sels.intersection(a2sels)
        # cons_attrs_int = c1sels.intersection(c2sels)

        # ant_attrs_int = sample_in_bounds(ant_attrs_int, ctx.antecedent)
        # cons_attrs_int = sample_in_bounds(cons_attrs_int, ctx.consequent)

        selectors_by_attribute = {}
        for s in [*a1, *a2, *c1, *c2]:
            (a, _) = s
            selectors_by_attribute[a] = selectors_by_attribute.get(a, []) + [s]

        if len(ant_attrs_sym) >= 1 and len(cons_attrs_sym) >= 1:
            new_ant = [
                random.choice(selectors_by_attribute.get(a)) for a in ant_attrs_sym
            ]
            new_cons = [
                random.choice(selectors_by_attribute.get(c)) for c in cons_attrs_sym
            ]
            rules_from_pair.append((tuple(new_ant), tuple(new_cons)))

        # if len(ant_attrs_int) >= 1 and len(cons_attrs_int) >= 1:
        #     new_ant = [
        #         random.choice(selectors_by_attribute.get(a)) for a in ant_attrs_int
        #     ]
        #     new_cons = [
        #         random.choice(selectors_by_attribute.get(c)) for c in cons_attrs_int
        #     ]
        #     rules_from_pair.append((tuple(new_ant), tuple(new_cons)))

        if len(ant_attrs_union) >= 1 and len(cons_attrs_union) >= 1:
            new_ant = [
                random.choice(selectors_by_attribute.get(a)) for a in ant_attrs_union
            ]
            new_cons = [
                random.choice(selectors_by_attribute.get(c)) for c in cons_attrs_union
            ]
            rules_from_pair.append((tuple(new_ant), tuple(new_cons)))

        # ants = set(a1).union(a2)
        # cons = set(c1).union(c2)

        # ants_int = set(a1).intersection(a2)
        # cons_int = set(c1).intersection(c2)

        # ants_symdif = set(a1).symmetric_difference(a2)
        # cons_symdif = set(c1).symmetric_difference(c2)

        # if len(ants_int) >= 1 and len(cons_int) >= 1:
        #     rules_from_pair.append((tuple(ants_int), tuple(cons_int)))

        # if len(ants_symdif) >= 1 and len(cons_symdif) >= 1:
        #     rules_from_pair.append((tuple(ants_symdif), tuple(cons_symdif)))
        #     print(
        #         repr_selectors(tuple(ants_symdif)), repr_selectors(tuple(cons_symdif))
        #     )

        rules_from_pair.append((a1, c2))
        rules_from_pair.append((a2, c1))

        # rules_from_pair.append((tuple(ants), tuple(cons)))
        # rules_from_pair.append(
        #     (
        #         tuple(
        #             random.sample(
        #                 tuple(ants),
        #                 k=random.randint(
        #                     ctx.antecedent[0], min(ctx.antecedent[1], len(ants))
        #                 ),
        #             )
        #         ),
        #         tuple(
        #             random.sample(
        #                 tuple(cons),
        #                 k=random.randint(
        #                     ctx.consequent[0], min(ctx.consequent[1], len(cons))
        #                 ),
        #             )
        #         ),
        #     )
        # )

        # for e in rules_from_pair:
        #     print(repr_rule(e))

        new_rules += rules_from_pair

    return new_rules

