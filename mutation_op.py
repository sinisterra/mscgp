import multiprocessing
from turtle import clear

from matplotlib.cbook import ls_mapper
from validation_op import attributes_must_be_unique
from dsl import Context, AlgorithmState
from measures import evaluate_rule
from rule_repr import format_for_population, repr_rule, repr_selector
from restrictions import enforce_restrictions
import pandas as pd
import random
from itertools import chain
from functools import partial, reduce


def do_local_search_mutation(ctx: Context, state: AlgorithmState, population):

    top_elements = population
    for_mutation = top_elements

    rules_in_pop = set(population)

    mutated = []
    all_mutations = []
    for rule in population:
        (antecedent, consequent) = rule
        rule_mutations = []
        for (i, a) in enumerate(antecedent):
            a_selector = a[0]
            sels = state.selectors[a_selector]
            sels = [sa for sa in sels if sa != a]
            bound = len(sels)
            a_selectors = random.sample(sels, k=min(bound, len(sels)))
            for new_selector in a_selectors:
                new_antecedent = list(antecedent)
                new_antecedent[i] = new_selector
                rule_mutations.append((tuple(new_antecedent), consequent,))

        for (i, a) in enumerate(consequent):
            a_selector = a[0]
            sels = state.selectors[a_selector]
            sels = [sa for sa in sels if sa != a]
            bound = len(sels)
            a_selectors = random.sample(sels, k=min(bound, len(sels)))
            for new_selector in a_selectors:
                new_antecedent = list(consequent)
                new_antecedent[i] = new_selector
                rule_mutations.append((antecedent, tuple(new_antecedent),))

        if len(antecedent) > 1:
            for (i, a) in enumerate(antecedent):
                # create a new rule by dropping each selector
                new_antecedent = [*antecedent]
                del new_antecedent[i]
                rule_mutations.append((tuple(new_antecedent), consequent))

        if len(consequent) > 1:
            for (i, a) in enumerate(consequent):
                # create a new rule by dropping each selector
                new_antecedent = [*consequent]
                del new_antecedent[i]
                rule_mutations.append((antecedent, tuple(new_antecedent)))

        all_mutations += rule_mutations

    rule_mutations = [
        s
        for s in all_mutations
        if repr_rule(s) not in rules_in_pop
        and repr_rule(s) not in all_mutations
        and repr_rule(s)
        not in ([] if len(state.history) == 0 else state.history["repr"].unique())
    ]

    return random.sample(
        set(all_mutations), k=min(len(set(all_mutations)), ctx.pop_size)
    )


def do_extension_mutation(ctx: Context, state: AlgorithmState, population):
    antecedent_attrs = ctx.groups[0]
    consequent_attrs = ctx.groups[1]

    attribute_options = set([*ctx.groups[0], *ctx.groups[1]])
    mutated = []
    ready = []
    for rule in population:
        # rule = row["rule"]
        (antecedent, consequent) = rule
        rule_attributes = [s[0] for s in [*rule[0], *rule[1]]]
        options = list(attribute_options.difference(set(rule_attributes)))

        if len(options) > 0:
            for o in random.sample(options, k=min(1, 3, len(options))):
                if o in antecedent_attrs:
                    mutated.append(
                        (
                            (*antecedent, *random.sample(state.selectors[o], k=1)),
                            consequent,
                        )
                    )
                if o in consequent_attrs:
                    mutated.append(
                        (
                            antecedent,
                            (*consequent, *random.sample(state.selectors[o], k=1)),
                        )
                    )

    return random.sample(mutated, k=min(len(mutated), ctx.pop_size))


def do_length_mutation(ctx: Context, state: AlgorithmState, population):
    p_mutation = 0.5
    for_mutation = population.sample(frac=p_mutation)
    new_rules_discovered = 0

    mutated = []
    for (_, row) in for_mutation.iterrows():
        try:
            rule = row["rule"]
            (antecedent, consequent) = rule

            new_antecedent = [*antecedent]
            new_consequent = [*consequent]

            rule_attributes = set([s[0] for s in [*antecedent, *consequent]])
            attribute_options = set(state.attributes).difference(set(rule_attributes))

            # select a new length for the antecedent
            antecedent_length_options = [
                l for l in range(ctx.antecedent[0], ctx.antecedent[1] + 1)
            ]
            consequent_length_options = [
                l for l in range(ctx.consequent[0], ctx.consequent[1] + 1)
            ]

            if (
                len(antecedent_length_options) == 0
                or len(consequent_length_options) == 0
            ):
                continue

            new_antecedent_length = random.choice(antecedent_length_options)
            new_consequent_length = random.choice(consequent_length_options)

            antecedent_length_diff = abs(new_antecedent_length - len(antecedent))
            consequent_length_diff = abs(new_consequent_length - len(consequent))

            if new_antecedent_length > len(antecedent):
                new_attributes = random.choices(
                    list(attribute_options),
                    k=min(len(attribute_options), antecedent_length_diff),
                )
                attribute_options = attribute_options.difference(set(new_attributes))
                new_selectors = [
                    random.choice(state.selectors.get(a)) for a in new_attributes
                ]
                new_antecedent += new_selectors

            if new_antecedent_length < len(antecedent):
                new_antecedent = random.choices(new_antecedent, k=new_antecedent_length)

            if new_consequent_length > len(consequent):
                new_attributes = random.choices(
                    list(attribute_options), k=consequent_length_diff
                )
                attribute_options = attribute_options.difference(set(new_attributes))
                new_selectors = [
                    random.choice(state.selectors.get(a)) for a in new_attributes
                ]
                new_consequent += new_selectors

            if new_consequent_length < len(consequent):
                new_consequent = random.choices(new_consequent, k=new_consequent_length)

            new_rule = (tuple(new_antecedent), tuple(new_consequent))
            evaluation = evaluate_rule(ctx, new_rule)

            if rule != new_rule:
                if enforce_restrictions(evaluation):
                    new_rules_discovered += 1
                    mutated.append(format_for_population(new_rule, evaluation))
        except Exception:
            print("Length mutation error")

    # print(f"Length mutation, rules discovered: {new_rules_discovered} ")
    return pd.DataFrame(mutated)


def do_attribute_mutation(ctx: Context, state: AlgorithmState, population):
    for_mutation = population.sample(frac=1)
    p_mutation = 0.5
    new_rules_discovered = 0

    mutated = []
    for (_, row) in for_mutation.iterrows():
        rule = row["rule"]
        (antecedent, consequent) = rule

        rule_attributes = [s[0] for s in [*antecedent, *consequent]]
        rule_selectors = [s for s in [*antecedent, *consequent]]

        attribute_options = set(state.attributes).difference(set(rule_attributes))

        new_selectors = []
        for s in rule_selectors:
            if random.random() <= p_mutation and len(attribute_options) > 0:
                attribute_option = random.choice(list(attribute_options))
                attribute_options.discard(attribute_option)

                new_selector = random.choice(state.selectors.get(attribute_option))
                new_selectors.append(new_selector)
            else:
                new_selectors.append(s)

        new_antecedent = new_selectors[: len(antecedent)]
        new_consequent = new_selectors[len(antecedent) :]

        new_rule = (tuple(new_antecedent), tuple(new_consequent))
        evaluation = evaluate_rule(ctx, new_rule)

        if rule != new_rule:
            if enforce_restrictions(evaluation):
                new_rules_discovered += 1
                mutated.append(format_for_population(new_rule, evaluation))

    print(f"Attribute mutation, rules discovered: {new_rules_discovered} ")
    return pd.DataFrame(mutated)


def do_value_mutation_(ctx: Context, state: AlgorithmState, population):
    for_mutation = population.sample(frac=1)
    p_selector_mutation = 0.5
    new_rules_discovered = 0
    mutated = []
    for (_, row) in for_mutation.iterrows():
        rule = row["rule"]
        (antecedent, consequent) = rule

        rule_selectors = [*antecedent, *consequent]
        new_selectors = []
        for s in rule_selectors:
            if random.random() <= p_selector_mutation:
                (attribute, _) = s

                attribute_selectors = set(state.selectors.get(attribute))
                attribute_selectors.discard(s)
                if len(list(attribute_selectors)) > 0:
                    new_selectors.append(random.choice(list(attribute_selectors)))
            else:
                new_selectors.append(s)

        new_antecedent = new_selectors[: len(antecedent)]
        new_consequent = new_selectors[len(antecedent) :]

        new_rule = (tuple(new_antecedent), tuple(new_consequent))

        if rule != new_rule:
            evaluation = evaluate_rule(ctx, new_rule)

            if enforce_restrictions(evaluation):
                new_rules_discovered += 1

                mutated.append(format_for_population(new_rule, evaluation))

    print(f"Value mutation, rules discovered:{new_rules_discovered} ")
    return pd.DataFrame(mutated)


def do_mutation(ctx: Context, state: AlgorithmState, population):
    for_mutation = population.sample(frac=1)

    mutated = []
    for (_, row) in for_mutation.iterrows():

        rule = row["rule"]
        (antecedent, consequent) = rule

        len_antecedent = len(antecedent)
        len_consequent = len(consequent)

        len_antecedent_cons = max(0, len_antecedent - 1)
        len_consequent_cons = max(0, len_consequent - 1)

        antecedent_conserved = random.sample(antecedent, k=len_antecedent_cons,)
        consequent_conserved = random.sample(consequent, k=len_consequent_cons)

        total_remaining = len_antecedent_cons + len_consequent_cons
        attributes_in_rule = [r[0] for r in (*antecedent, *consequent)]

        # pick new attributes as long as these aren't present in the current rules
        remaining_attributes = random.sample(
            [a for a in state.attributes if a not in attributes_in_rule],
            k=(len_antecedent + len_consequent) - total_remaining,
        )
        remaining_ant = len_antecedent - len_antecedent_cons
        # print(state.attributes, attributes_in_rule, remaining_attributes, remaining_ant)

        [new_sels_antecedent, new_sels_consequent] = [
            tuple(
                [
                    random.choice(state.selectors[a])
                    for a in remaining_attributes[0:remaining_ant]
                ]
            ),
            tuple(
                [
                    random.choice(state.selectors[a])
                    for a in remaining_attributes[remaining_ant:]
                ]
            ),
        ]

        new_rule = (
            (*antecedent_conserved, *new_sels_antecedent),
            (*consequent_conserved, *new_sels_consequent),
        )

        # print(rule, new_rule)

        evaluation = evaluate_rule(ctx, new_rule)
        if enforce_restrictions(evaluation):
            mutated.append(format_for_population(new_rule, evaluation))

    return pd.DataFrame(mutated)


def do_single_extension_mutation(ctx: Context, state: AlgorithmState, population):

    mutated = []
    for r in population:
        (antecedent, consequent) = r
        selectors_in_rule = [*[s[0] for s in antecedent], *[s[0] for s in consequent]]
        selector_options = set([*ctx.groups[0], *ctx.groups[1]]).difference(
            selectors_in_rule
        )
        if len(list(selector_options)) == 0:
            continue

        option = random.choice(list(selector_options))

        new_rule = None
        if option in ctx.groups[0]:
            new_rule = (
                (*antecedent, random.choice(list(state.selectors[option]))),
                consequent,
            )

        if option in ctx.groups[1]:
            new_rule = (
                antecedent,
                (*consequent, random.choice(list(state.selectors[option]))),
            )

        if new_rule is None:
            continue

        new_rule = (
            tuple(sorted(new_rule[0], key=lambda s: s[0])),
            tuple(sorted(new_rule[1], key=lambda s: s[0])),
        )

        mutated.append(new_rule)
        # evaluation = evaluate_rule(ctx, new_rule)
        # if enforce_restrictions(evaluation):

    return mutated


def do_value_mutation(ctx: Context, state: AlgorithmState, population):
    mutated = []
    for r in population:
        (antecedent, consequent) = r

        acc = []
        # se crea una tupla para saber a qué parte de la regla corresponde cada selector
        for a in antecedent:
            acc.append(("a", a))
        for c in consequent:
            acc.append(("c", c))

        # elegir un selector aleatorio
        random_selector = random.choice(acc)

        # descomponer en atributo y valor
        (_, (attribute, value)) = random_selector

        # crear dos listas en las que se acumularán los selectores
        na = []
        nc = []

        # crear una lista exclyendo al selector siendo mutado
        new_selector_options = [
            (attr, val) for (attr, val) in state.selectors[attribute] if val != value
        ]

        # si es el único valor, excluir
        if len(new_selector_options) == 0:
            continue

        new_selector = random.choice(new_selector_options)

        # iterar por la lista de selectores, agregar a la lista acumuladora si
        # aún no se ha reemplazado
        for (p, t) in acc:
            if p == "a":
                na.append(new_selector if t == random_selector[1] else t)
            if p == "c":
                nc.append(new_selector if t == random_selector[1] else t)

        new_rule = (tuple(na), tuple(nc))

        mutated.append(new_rule)

    return mutated


def do_contraction_mutation(ctx: Context, state: AlgorithmState, population):

    mutated = []
    for r in population:
        (antecedent, consequent) = r
        if len(antecedent) == 1 and len(consequent) == 1:
            continue

        removable_selectors = []
        if len(antecedent) > 1:
            for s in antecedent:
                removable_selectors.append(s[0])

        if len(consequent) > 1:
            for s in consequent:
                removable_selectors.append(s[0])

        to_remove = random.choice(removable_selectors)

        new_rule = (
            tuple([s for s in antecedent if s[0] != to_remove]),
            tuple([s for s in consequent if s[0] != to_remove]),
        )

        new_rule = (
            tuple(sorted(new_rule[0], key=lambda s: s[0])),
            tuple(sorted(new_rule[1], key=lambda s: s[0])),
        )

        mutated.append(new_rule)
        # evaluation = evaluate_rule(ctx, new_rule)
        # if enforce_restrictions(evaluation):
    return mutated
