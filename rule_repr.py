def repr_selector(s):
    return f"[{s[0]} = {s[1]}]"


def repr_selectors(selectors):
    return "".join(
        [repr_selector(s) for s in sorted(selectors, key=lambda t: f"{t[0]}_{t[1]}")]
    )


def repr_rule(rule):
    return f"{repr_selectors(rule[0])} -> {repr_selectors(rule[1])}"


def format_for_population(rule, evaluation):
    return {
        "repr": repr_rule(rule),
        "antecedent": repr_selectors(rule[0]),
        "consequent": repr_selectors(rule[1]),
        **evaluation,
        "rule": rule,
    }
