import itertools
from filtering_op import do_max_filtering
from validation_op import validate_rule
from dsl import Context
from measures import evaluate_rule
from restrictions import enforce_restrictions
from rule_repr import format_for_population, repr_rule
from validation_op import do_validation
from elitism_op import do_elitism
from nds import nds

import pandas as pd
import random
import numpy as np

random.seed(42)
np.random.seed(42)


def make_individual(ctx, selectors, attributes):

    if ctx.use_groups == False:
        len_antecedent = random.randint(*ctx.antecedent)
        len_consequent = random.randint(*ctx.consequent)

        rule_length = len_antecedent + len_consequent

        itemset = random.sample(attributes, k=rule_length)
        (antecedent_items, consequent_items) = (
            itemset[0:len_antecedent],
            itemset[len_antecedent:],
        )

        antecedent = tuple([random.choice(selectors[a]) for a in antecedent_items])
        consequent = tuple([random.choice(selectors[c]) for c in consequent_items])

        rule = (antecedent, consequent)
        return rule

    if ctx.use_groups:
        (ant_elems, cons_elems) = ctx.groups
        len_antecedent = min(random.randint(*ctx.antecedent), len(ant_elems))
        len_consequent = min(random.randint(*ctx.consequent), len(cons_elems))

        antecedent_items = random.sample(list(ant_elems), k=len_antecedent)
        consequent_items = random.sample(list(cons_elems), k=len_consequent)

        antecedent = tuple([random.choice(selectors[a]) for a in antecedent_items])
        consequent = tuple([random.choice(selectors[c]) for c in consequent_items])

        rule = (antecedent, consequent)
        return rule


def chunk(arr, n):
    elems = [arr[i : i + n] for i in range(0, len(arr), n)]
    return [e for e in elems if len(e) == n]


def make_population_even(ctx: Context, selectors, attributes):
    new_individuals = []
    rules = set()
    selector_list = [item for sublist in selectors.values() for item in sublist]
    evals = 0

    while len(new_individuals) < ctx.pop_size:
        # print(len(new_individuals), evals)
        shuffled_ants = list(selector_list)[:]
        shuffled_cons = list(selector_list)[:]

        random.shuffle(shuffled_ants)
        random.shuffle(shuffled_cons)

        # split selectors based on the specified cardinality restrictions
        n1 = ctx.antecedent[1]
        n2 = ctx.consequent[1]
        split_ants = chunk(shuffled_ants, n1)
        split_cons = chunk(shuffled_cons, n2)

        total_pairs = min(len(split_ants), len(split_cons))

        sa = split_ants[0:total_pairs]
        sc = split_cons[0:total_pairs]

        print("checking", total_pairs)

        for (ant_options, cons_options) in zip(sa, sc):
            evals += 1
            len_ant = random.randint(*ctx.antecedent)
            len_cons = random.randint(*ctx.consequent)
            # print(len(ant_options), len_ant)
            new_ant = random.sample(ant_options, k=len_ant)
            new_cons = random.sample(cons_options, k=len_cons)

            #            print(new_ant, new_cons)

            new_rule = (tuple(new_ant), tuple(new_cons))
            new_rule_eval = evaluate_rule(ctx, new_rule)

            satisfies_restrictions = enforce_restrictions(
                new_rule_eval
            ) and validate_rule(ctx, new_rule)
            print(
                evals,
                ant_options,
                cons_options,
                satisfies_restrictions,
                len(new_individuals),
            )

            if (
                new_rule not in rules
                and satisfies_restrictions
                and len(new_individuals) < ctx.pop_size
            ):

                rules.add(new_rule)
                new_individuals.append(format_for_population(new_rule, new_rule_eval))

            if len(new_individuals) == ctx.pop_size:
                break

    return pd.DataFrame(new_individuals)


def make_two_by_two_pop(ctx: Context, selectors):
    two_by_two = []
    g1 = list(ctx.groups[0])
    g2 = list(ctx.groups[1])
    random.shuffle(g1)
    random.shuffle(g2)

    for (a0, a1) in itertools.product(g1, g2):
        # print(len(two_by_two))
        if len(two_by_two) >= (3 * ctx.pop_size):
            break

        sels_a0 = list(selectors[a0])
        sels_a1 = list(selectors[a1])

        random.shuffle(sels_a0)
        random.shuffle(sels_a1)

        for (s0, s1) in itertools.product(sels_a0, sels_a1):
            new_ind = (
                (s0,),
                (s1,),
            )
            evaluation = evaluate_rule(ctx, new_ind)

            # if enforce_restrictions(evaluation):
            two_by_two.append(format_for_population(new_ind, evaluation))

            if len(two_by_two) >= (3 * ctx.pop_size):
                break

    if len(two_by_two) > 0:
        result_pop = do_validation(ctx, pd.DataFrame(two_by_two)).query(
            "significant == True and tp > 0"
        )
        result_pop.to_csv(f"./two_by_two_{ctx.id}.csv", index=False)

        return result_pop
    else:
        return pd.DataFrame()


def make_population(ctx: Context, selectors, attributes):
    # print("Making initial population")
    # make ctx.n individuals
    new_individuals = []
    rule_history = []
    evals = 0
    max_evals = 10000

    two_by_two_pop = make_two_by_two_pop(ctx, selectors)
    if len(two_by_two_pop) >= ctx.pop_size:
        if len(ctx.measures) > 1:
            two_by_two_pop = nds(
                two_by_two_pop, ctx.measures, ctx.optimize
            ).sort_values("level", ascending=False)
        else:
            two_by_two_pop = two_by_two_pop.sort_values(
                list(ctx.measures), ascending=[o == "min" for o in ctx.optimize]
            ).reset_index(drop=True)

    # two_by_two_pop = two_by_two_pop.iloc[0 : min(len(two_by_two_pop), ctx.pop_size)]
    # two_by_two_pop = two_by_two_pop.sample(min(len(two_by_two_pop,), ctx.pop_size))

    # new_individuals = random.sample(two_by_two, k=int(ctx.pop_size * 1))
    # print(pd.DataFrame(new_individuals)[["repr", *ctx.measures]])

    while (len(two_by_two_pop) + len(new_individuals)) < ctx.pop_size:
        if evals >= max_evals:
            return None
        evals += 1
        # make a new individual and evaluate it
        new_ind = make_individual(ctx, selectors, attributes)
        if validate_rule(ctx, new_ind):
            evaluation = evaluate_rule(ctx, new_ind)
            # print(evals)
            # lift > 1 and must have support
            if enforce_restrictions(evaluation):
                new_individuals.append(format_for_population(new_ind, evaluation))

    return do_validation(
        ctx, pd.concat([two_by_two_pop, pd.DataFrame(new_individuals)])
    )
