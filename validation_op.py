from pymonad.tools import curry
from pymonad.either import Right, Left
from functools import lru_cache
from dsl import Context
from restrictions import enforce_restrictions
from rule_repr import repr_rule


@curry(2)
def m_filter(fn, x):
    return Right(x) if fn(x) else Left(fn.__name__)


# a rule must not have duplicate selectors
def selectors_must_be_unique(rule):
    (antecedent, consequent) = rule
    selectors = (*antecedent, *consequent)
    if len(selectors) == len(set(list(selectors))):
        return True

    return False


# a rule must not have selectors with the same attribute
def attributes_must_be_unique(rule):
    (antecedent, consequent) = rule
    attributes = [s[0] for s in (*antecedent, *consequent)]
    if len(attributes) == len(set(attributes)):
        return True

    return False


@curry(2)
def check_cardinality(ctx, rule):
    (antecedent, consequent) = rule

    if ctx.antecedent[0] <= len(antecedent) <= ctx.antecedent[1]:
        if ctx.consequent[0] <= len(consequent) <= ctx.consequent[1]:
            return True

    return False


@curry(2)
def match_sg(ctx: Context, rule):
    if ctx.use_groups:
        (antecedent, consequent) = rule
        (g_antecedent, g_consequent) = ctx.groups
        ant_attributes = set([s[0] for s in antecedent])
        cons_attributes = set([s[0] for s in consequent])

        a_belong = (
            len(ant_attributes.difference(set(g_antecedent)))
            == 0
            # or len(ant_attributes.difference(set(g_consequent))) == 0
        )
        c_belong = (
            len(cons_attributes.difference(set(g_consequent)))
            == 0
            # or len(ant_attributes.difference(set(g_antecedent))) == 0
        )
        # print((antecedent, a_belong), (consequent, c_belong))
        # if a_belong and c_belong:
        #     print(repr_rule(rule))
        return a_belong and c_belong

    return True


@lru_cache(maxsize=None)
def validate_rule(ctx, rule):
    return (
        Right(rule)
        .bind(m_filter(selectors_must_be_unique))
        .bind(m_filter(attributes_must_be_unique))
        .bind(m_filter(check_cardinality(ctx)))
        .bind(m_filter(match_sg(ctx)))
        .either(lambda _: False, lambda _: True)
    )


def do_validation(ctx: Context, population):
    return population[
        population.apply(lambda row: validate_rule(ctx, row["rule"]), axis=1)
    ]

