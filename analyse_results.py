# %%
import pandas as pd
import seaborn as sns

mkgain = "results/1636695003"
support = "results/1637202579"
confidence = "results/1637202588"

p = mkgain
run_path = f"{p}/finals.csv"
df = pd.read_csv(run_path)


def calc_interestingness(r):
    # a_supp = r["antecedent_support"]
    # c_supp = r["consequent_support"]
    return r["lift"] * r["support"] * (1 - r["support"])


# %%
df_repeats = pd.DataFrame(
    df[
        [
            "repr",
            "sg_pair",
            "markedness_gain",
            "markedness",
            "cer",
            "eer",
            "absolute_risk",
            "relative_risk",
            "support",
            # "confidence",
            # "full_support",
            # "antecedent_support",
            # "consequent_support",
            "lift",
        ]
    ].value_counts()
).reset_index()
df_repeats["consequent"] = df_repeats["repr"].apply(lambda r: r.split("-> ")[1])
# df_repeats["antecedent_support"] = df_repeats["antecedent_support"] * (11200000)
# df_repeats["consequent_support"] = df_repeats["consequent_support"] * (11200000)
df_repeats["interestingness"] = df_repeats.apply(
    lambda r: calc_interestingness(r), axis=1
)
df_repeats = df_repeats.rename(columns={0: "repeats"})
df_repeats

df_repeats.to_csv("./repeats.csv", index=False)
# %%
sns.heatmap(
    pd.crosstab(df_repeats["sg_pair"], df_repeats["repeats"], normalize="index"),
    cmap="BuGn",
)

# %%
sns.countplot(data=df_repeats, x="repeats")

# %%
sns.catplot(data=df_repeats, col="sg_pair", x="repeats", col_wrap=4, kind="count")

# %%
