# %%

from io_helpers import get_dataframe
from clickhouse_io import get_selectors as clickhouse_get_selectors


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
        # str_valid_selectors = [str(s) for s in valid_selectors]
        # for e in restriction:
        #     if e in filtered_selectors:
        #         filtered_selectors[e] = [
        #             s for s in filtered_selectors[e] if str(s[1]) in str_valid_selectors
        #         ]

    return filtered_selectors
