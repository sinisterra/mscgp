# def build_xor(literals):
#     def _build_xor(acc, literals):
#         a xor b = (a and not b) or (not a and b)
# %%


def xor_expression(literals):
    def associate(x, y):
        return f"((({x}) and (not ({y}))) or ((not ({x})) and ({y})))"

    if len(literals) == 0:
        return ""

    if len(literals) == 1:
        return literals[0]

    if len(literals) == 2:
        return associate(literals[0], literals[1])

    if len(literals) > 2:
        [x, y, *rest] = literals
        base = associate(x, y)

        for l in rest:
            base = associate(base, l)

        return base


# %%
