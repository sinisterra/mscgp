from dsl import AlgorithmState
import pandas as pd


def show_report(state: AlgorithmState):
    pass


def show_population_report(ctx, population):
    # print(count_rules_by_attribute(population))
    print(show_population(ctx, population))
    pass


def show_population(ctx, population):
    return (
        population[["antecedent", "consequent", *ctx.measures,]]
        # .query("level == 1 and strong == True")
        .reset_index(drop=True)
    )


def count_rules_by_attribute(population):
    by_attribute = {}
    for (_, row) in population.iterrows():
        rule = row["rule"]
        (antecedent, consequent) = rule
        rule_attributes = [s[0] for s in [*antecedent, *consequent]]

        for attribute in rule_attributes:
            by_attribute[attribute] = by_attribute.get(attribute, 0) + 1

    table = []
    for (k, v) in by_attribute.items():
        table.append({"Attribute": k, "Rules with attribute": v})

    return (
        pd.DataFrame(table)
        .sort_values("Rules with attribute", ascending=False)
        .reset_index(drop="True")
    )
