from multiprocessing import Pool
from crossover_op import do_crossover, do_transitive_crossover, do_crossover_
from df_selectors import get_selectors
from dsl import AlgorithmState, Context
import termplotlib as tpl
from elitism_op import do_elitism, do_mo_elitism, do_selector_elitism, do_single_elitism
from filtering_op import do_filtering, do_max_filtering
from mutation_op import (
    do_contraction_mutation,
    do_extension_mutation,
    do_mutation,
    do_value_mutation,
    do_attribute_mutation,
    do_length_mutation,
    do_local_search_mutation,
    do_single_extension_mutation,
)
from tabulate import tabulate


import random
from population_init import make_population, make_two_by_two_pop
from trampoline import trampoline
from simplification import do_simplification
from validation_op import do_validation
from reports import show_population_report
import pandas as pd
from nds import nds
import time
from measures import evaluate_rule
from rule_repr import format_for_population, repr_rule
from functools import reduce, partial
from itertools import chain
from restrictions import enforce_restrictions

random.seed(0)


def map_rule_evaluation(rm, ctx):
    # print(repr_rule(rm))
    evaluation = evaluate_rule(ctx, rm)
    return (
        [format_for_population(rm, evaluation)]
        if enforce_restrictions(evaluation)
        else []
    )


mutation_fn = do_mutation
crossover_fn = do_crossover_
elitism_fn = do_elitism
init_fn = make_population


def make_initial_state(ctx: Context):
    selectors = get_selectors(ctx)
    all_attributes = list(selectors.keys())

    if ctx.use_groups:
        group_attributes = [*ctx.groups[0], *ctx.groups[1]]
        attributes = [g for g in all_attributes if g in group_attributes]
    else:
        attributes = all_attributes

    initial_pop = init_fn(ctx, selectors, attributes)

    if initial_pop is None:
        return None

    pop2x2 = make_two_by_two_pop(ctx, selectors)
    pop2x2["generation"] = -1

    new_state = AlgorithmState(
        generation=0,
        evaluations=len(initial_pop),
        population=initial_pop,
        history=pop2x2,
        selectors=selectors,
        attributes=attributes,
        convergence=pd.DataFrame(),
        pop_history=pop2x2,
        elites=pd.DataFrame(),
    )

    do_max_filtering(ctx, new_state, pop2x2).to_csv(
        f"./two_by_two_filtered_{ctx.id}.csv", index=False
    )

    return new_state


def stop_condition_met(ctx, state):
    (stop_criteria, stop_value) = ctx.stop_condition
    stop_criteria_options = ["n_eval", "n_gen"]
    if stop_criteria not in stop_criteria_options:
        raise Exception(
            f"StopCriteria error, '{stop_criteria}' not in {stop_criteria_options}"
        )

    if "n_eval" == stop_criteria:
        return state.evaluations > stop_value

    if "n_gen" == stop_criteria:
        return state.generation > stop_value

    return False


def evolve_t(ctx, state, start_time):
    start = time.time()
    if stop_condition_met(ctx, state):
        state.convergence.to_csv(f"{ctx.exec_run_path}/convergence.csv", index=False)
        return state

    # crossover and combine with the original population
    # print("Do crossover")
    # crossover_this_gen = random.choice([do_crossover])
    # after_crossover = crossover_this_gen(ctx, state)
    crossover_start = time.time()
    after_crossover = do_crossover_(ctx, state)
    crossover_end = time.time()
    state_population = list(state.population["rule"])
    # after_local_search = do_local_search_mutation(ctx, state, after_crossover)
    # after_extension = do_extension_mutation(ctx, state, after_crossover)
    extension_start = time.time()
    after_single_extension = do_single_extension_mutation(ctx, state, state_population)
    extension_end = time.time()

    contraction_start = time.time()
    after_contraction = do_contraction_mutation(ctx, state, state_population)
    contraction_end = time.time()

    from_operators = [*after_crossover, *after_single_extension, *after_contraction]
    rules_to_evaluate = set(
        [
            *state_population,
            *from_operators
            # *random.sample(after_crossover, k=ctx.pop_size),
            # *after_extension,
            # *after_local_search,
            # *after_extension
            # *random.sample(
            #     set([*after_local_search, *after_extension]).difference(
            #         set(state.history.drop_duplicates(subset="repr"))
            #     ),
            #     k=ctx.pop_size,
            # ),
        ]
    )
    total_new_offspring = len(
        set(from_operators).difference(state.pop_history["rule"].unique())
    )
    # print(f"{total_new_offspring} rules to evaluate")
    # pool = Pool()
    evaluation_start = time.time()
    evald_rules = map(partial(map_rule_evaluation, ctx=ctx), rules_to_evaluate)
    evald = reduce(chain, evald_rules)
    evaluation_end = time.time()

    merge_population = pd.DataFrame(evald).drop_duplicates(subset=["repr"])

    merge_population["generation"] = state.generation
    after_restriction_check = merge_population

    # select the best
    # print("Do selection")
    elitism_start = time.time()
    after_elitism = None
    if len(ctx.measures) == 1:
        after_elitism = do_single_elitism(ctx, state, merge_population)
    else:
        after_elitism = do_mo_elitism(ctx, state, merge_population)
    elitism_end = time.time()
    # .drop_duplicates(
    #     subset="itemset"
    # )
    # print(
    #     f"ready for selection:\n{after_elitism[(['repr', *ctx.measures, 'level', ])].sort_values(['level', *list(ctx.measures)], ascending=tuple( [True]+[o=='min' for o in ctx.optimize])).reset_index(drop=True)}"
    # )

    final_population = do_validation(ctx, after_elitism)

    # final_population = after_elitism
    # fp_size = len(final_population)

    # if fp_size < ctx.pop_size:
    #     rules_in_fp = set(final_population["repr"].unique())
    #     # all_rules = set(state.pop_history["repr"].unique())
    #     # not_in_fp = all_rules.difference(rules_in_fp)
    #     fillup = state.pop_history[
    #         ~state.pop_history["repr"].isin(list(rules_in_fp))
    #     ].sample(n=ctx.pop_size - fp_size, weights="aptitude")
    #     print(fp_size, ctx.pop_size - fp_size)
    #     final_population = pd.concat([final_population, fillup])

    # show_population_report(ctx, final_population)
    pop_history = merge_population.copy()
    elites = final_population.copy()
    pop_history["generation"] = state.generation
    elites["generation"] = state.generation

    end = time.time()
    risk_bar = (
        pd.cut(elites["markedness"], bins=[s / 5 for s in range(5 + 1)],)
        .value_counts()
        .sort_index(ascending=True)
    )

    time_df = pd.DataFrame(
        [
            {
                "event": "CROSSOVER",
                "duration": crossover_end - crossover_start,
                "rules": len(after_crossover),
            },
            {
                "event": "MUTATE_EXTENSION",
                "duration": extension_end - extension_start,
                "rules": len(after_single_extension),
            },
            {
                "event": "MUTATE_CONTRACTION",
                "duration": contraction_end - contraction_start,
                "rules": len(after_contraction),
            },
            {
                "event": "EVALUATION",
                "duration": evaluation_end - evaluation_start,
                "rules": len(merge_population),
            },
            {
                "event": "ELITISM",
                "duration": elitism_end - elitism_start,
                "rules": len(after_elitism),
            },
        ]
    )
    time_df["context"] = ctx.id
    time_df = time_df[["event", "duration", "rules"]]
    time_df["duration"] = time_df["duration"].round(4)
    # print(tabulate(time_df, headers="keys", tablefmt="psql"))

    is_mono = len(ctx.measures) == 1
    # print(final_population[["repr", "absolute_risk_rev"]])
    df_print = (
        final_population
        # .query(
        #     f"selected_by_filter == True or diversity == True {'or level == 1' if not is_mono else ''}"
        # )
        .copy()
        .sort_values(
            ["level", *ctx.measures] if not is_mono else list(ctx.measures),
            ascending=(True, *[o == "min" for o in ctx.optimize])
            if not is_mono
            else tuple([o == "min" for o in ctx.optimize]),
        )
        .reset_index(drop=True)
    )

    df_print["DIV"] = df_print["diversity"].apply(lambda v: "*" if v else "")
    if not is_mono:
        df_print["PAR"] = df_print["level"].apply(lambda v: "*" if v == 1 else "")
    df_print["FS"] = df_print["selected_by_filter"].apply(lambda v: "*" if v else "")
    df_print["TOP10"] = df_print.index.to_series().map(lambda v: "*" if v <= 10 else "")
    # df_print["rev_abs"] = df_print["full_tpr"] - df_print["full_fpr"]

    extra_measures = [
        # "cer",
        # "eer",
        # "fn",
        # "tp",
        # "cer_total",
        # "eer_total",
        # "absolute_risk_abs",
        # "full_absolute_risk_rev",
        # "used_factor",
        # "total",
        # "relative_risk",
        # "aptitude",
        # "nnt",
        # "nnt_placebo",
        # "nnt_intervention",
        # "nnt_nonresponse"
        # "susceptibility",
        *ctx.aptitude_fn,
        "aptitude",
        # "absolute_risk_abs",
        "full_support",
        "prevalence"
        # "cer",
        # "absolute_risk",
        # "af_e",
        # "paf"
        # "absolute_risk_abs",
        # "prevalence_threshold_diff"
        # "nnt",
        # "r_nnt"
        # "fn",
    ]

    # for e in extra_measures:
    #     df_print[f"{e}_total"] = (df_print[e] * df_print["used_factor"]).astype(int)

    should_include_pareto = "PAR == '*' or " if not is_mono else ""
    table_result = tabulate(
        df_print[
            [
                "repr",
                *(["level", *ctx.measures] if not is_mono else ctx.measures),
                *[e for e in extra_measures if e not in ctx.measures],
                # "absolute_risk",
                # "r_absolute_risk",
                *(["PAR"] if not is_mono else []),
                "DIV",
                "FS",
                "TOP10",
            ]
        ].reset_index(drop=True),
        # .query("DIV == '*'"),
        # .query(f"{should_include_pareto}FS == '*' or DIV == '*'"),
        headers="keys",
        tablefmt="pgsql",
        showindex=False,
    )

    force_cond = ctx.stop_condition[1] == state.generation
    # simplification_result = tabulate(
    #     do_simplification(final_population, force=False).query("mode == 'AC'"),
    #     headers="keys",
    #     showindex=False,
    # )
    print(
        f"[{str(ctx.id).zfill(3)}] - Generation {state.generation}\t ({len(state.history)} rules) [+{total_new_offspring}] {(str(round(end-start,2))).zfill(2)}s {str(round(end - start_time, 2)).zfill(2)}s\n"
        + f"{table_result}\n\n"
        # + f"{simplification_result}\n\n"
    )

    # fig = tpl.figure()
    # fig.barh([*list(dict(risk_bar).values()),], [*list(dict(risk_bar).keys()),])
    # fig.show()

    # %%

    avg_measures = {
        "Generation": state.generation,
        "Length": len(state.history.drop_duplicates(subset="repr")),
    }
    for m in set([*ctx.measures, "support", "confidence"]):
        avg_measures[f"{m}_mean"] = state.population[m].mean()
        avg_measures[f"{m}_min"] = state.population[m].min()
        avg_measures[f"{m}_max"] = state.population[m].max()
        avg_measures[f"{m}_std"] = state.population[m].std()
        avg_measures[f"time"] = end - start
        avg_measures[f"total_time"] = end - start_time

    convergence = pd.DataFrame([avg_measures])

    return (
        yield evolve_t(
            ctx,
            AlgorithmState(
                generation=state.generation + 1,
                evaluations=state.evaluations + len(final_population),
                selectors=state.selectors,
                attributes=state.attributes,
                population=final_population,
                # individuals before elitism are kept in history
                history=pd.concat([state.history, merge_population]).drop_duplicates(
                    subset="repr"
                ),
                convergence=pd.concat([state.convergence, convergence])
                .round(4)
                .reset_index(drop=True),
                pop_history=pd.concat([state.pop_history, pop_history]),
                elites=pd.concat([state.elites, elites]),
            ),
            start_time,
        )
    )


def evolve(ctx, state):
    start_time = time.time()
    return trampoline(evolve_t(ctx, state, start_time))


def run_algorithm(ctx):
    print(f"[{str(ctx.id).zfill(3)}] Begin execution")
    start = time.time()
    # generate an initial population
    state = make_initial_state(ctx)
    if state is None:
        return None
    end = time.time()
    print(
        f"[{str(ctx.id).zfill(3)}]  Population creation finished ({round(end - start, 2)}s)"
    )
    return evolve(ctx, state)
