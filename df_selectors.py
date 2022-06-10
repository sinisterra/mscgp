# %%
from functools import lru_cache
from io_helpers import get_dataframe
from clickhouse_io import get_selectors as clickhouse_get_selectors


@lru_cache(maxsize=None)
def get_selectors(ctx):
    selectors = clickhouse_get_selectors(ctx.dataframe)
    filtered_selectors = {**selectors}

    for (restriction, (restriction_type, valid_selectors)) in ctx.selector_restrictions:
        for elem in restriction:
            if elem in filtered_selectors.keys():
                filtered_res = []
                for sel in filtered_selectors[elem]:
                    (_, v) = sel
                    if "keep" == restriction_type:
                        if str(v) in [str(sl) for sl in valid_selectors]:
                            filtered_res.append(sel)

                    if "remove" == restriction_type:
                        if str(v) not in [str(sl) for sl in valid_selectors]:
                            filtered_res.append(sel)

                filtered_selectors[elem] = filtered_res

    # count how many selectors remain after filtering
    selectors_in_groups = [*ctx.groups[0], *ctx.groups[1]]
    total_selectors = sum([len(filtered_selectors[e]) for e in selectors_in_groups])
    print(total_selectors)

    return filtered_selectors
