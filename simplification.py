# %%
from more_itertools import first
import pandas as pd
from rule_repr import repr_selector, repr_selectors
from sympy import Symbol, Or, And, satisfiable, latex
from sympy.logic import simplify_logic
from sympy import *
from tabulate import tabulate

init_printing(use_latex=False)


def do_simplification(df, force=False):
    FORCE = force

    d_symbols = {}
    by_antecedent = {}
    by_consequent = {}
    antecedents = {}
    consequents = {}
    for (i, r) in df.iterrows():
        (a, c) = r["rule"]
        r_antecedent = repr_selectors(a)
        r_consequent = repr_selectors(c)
        for e in [*a, *c]:
            r_e = repr_selector(e)
            s_e = Symbol(r_e)
            d_symbols[r_e] = s_e
        antecedent_symbols = [d_symbols[repr_selector(s)] for s in a]
        consequent_symbols = [d_symbols[repr_selector(s)] for s in c]
        antecedents[r_antecedent] = antecedent_symbols
        consequents[r_consequent] = consequent_symbols
        by_antecedent[r_antecedent] = by_antecedent.get(r_antecedent, []) + [
            consequent_symbols
        ]
        by_consequent[r_consequent] = by_consequent.get(r_consequent, []) + [
            antecedent_symbols
        ]

    implications_a = []
    first_simpl = {}
    for (a, consqs) in by_antecedent.items():
        # join consequents as a SOP
        antecedent_elems = antecedents[a]
        antecedent_expr = And(*antecedent_elems)
        simplified = simplify_logic(Or(*[And(*exprs) for exprs in consqs]), force=FORCE)
        expr = antecedent_expr >> simplified
        first_simpl[str(simplified)] = first_simpl.get(str(simplified), []) + [
            (antecedent_expr, simplified)
        ]
        implications_a.append(
            {"antecedent": antecedent_expr, "consequent": simplified, "mode": "C",}
        )

    # %%
    implications_c = []
    for (c, antecs) in by_consequent.items():
        # join antecs as a SOP
        consequent_elems = consequents[c]
        consequent_expr = And(*consequent_elems)
        simplified = simplify_logic(Or(*[And(*exprs) for exprs in antecs]), force=FORCE)
        expr = simplified >> consequent_expr
        implications_c.append(
            {"antecedent": simplified, "consequent": consequent_expr, "mode": "A",}
        )
        # print(expr)

    # %%
    after_simps = []
    for (_, rs) in first_simpl.items():
        # simplificar los antecedentes
        ants = []
        for (a, c) in rs:
            ants.append(a)
        ant = simplify_logic(Or(*ants), force=FORCE)
        cons = rs[0][1]
        expr = ant >> cons
        # pretty_print(expr)
        # sat = satisfiable(expr)
        # print(tabulate(pd.DataFrame([sat]).transpose(), headers="keys"))
        # print("---")
        # print(expr, satisfiable(expr))
        after_simps.append({"antecedent": ant, "consequent": cons, "mode": "AC"})

    return pd.DataFrame([*after_simps, *implications_a, *implications_c])


# %%
# do_simplification(df)
