# %%
import json
import pandas as pd
import seaborn as sns

f = open("./perCountryData.json")


data = json.load(f)
data


dists = data["regions"][0]["distributions"]
mexico = [d for d in dists if d["country"] == "Mexico"][0]["distribution"]
mexico

acc = []
for e in mexico:
    acc.append(
        {
            **e["cluster_counts"],
            "week": e["week"],
            "total_sequences": e["total_sequences"],
        }
    )

not_others = [
    "20I (Alpha, V1)",
    "20H (Beta, V2)",
    "20J (Gamma, V3)",
    "21A (Delta)",
    "21I (Delta)",
    "21J (Delta)",
    "21K (Omicron)",
    "21L (Omicron)",
    "21B (Kappa)",
    "21D (Eta)",
    "21F (Iota)",
    "21G (Lambda)",
    "21H (Mu)",
    "20B/S:732A",
    "20A/S:126A",
    "20E (EU1)",
    "21C (Epsilon)",
    "20A/S:439K",
    "S:677H.Robin1",
    "S:677P.Pelican",
    "20A.EU2",
    "20A/S:98F",
    "20C/S:80Y",
    "20B/S:626S",
    "20B/S:1122L",
]
df = pd.DataFrame(acc)
variants = [c for c in df.columns if c not in ["0", "week", "total_sequences"]]
dominant_variants = []
for (i, r) in df.iterrows():
    week = r["week"]
    ts = r["total_sequences"]

    r_variants = [(v, r[v]) for v in variants]
    sum_clusters = sum([e[1] for e in r_variants])
    others = ts - sum_clusters

    max_value = max(r_variants + [("others", others)], key=lambda item: item[1])
    dominant_variants.append(
        {"variant": max_value[0], "total_variants": max_value[1], "week": week}
    )

    # if max_value[1] == 0:
    #     print(week, "others")
    # else:
    #     print(week, max_value)

dfdv = pd.DataFrame(dominant_variants)
week_start = dfdv["week"]
week_end = week_start[1:]
dominant_variants_per_day = []
zipped = (
    [("2020-01-01", "2020-04-27")]
    + list(zip(week_start, week_end))
    + [("2022-03-07", "2022-03-21"), ("2022-03-22", "2022-04-02"),]
)
for (ws, we) in zipped:
    if ws == "2020-01-01":
        dv = "others"

    if ws == "2022-03-22":
        dv = "21K (Omicron)"

    if ws not in ["2020-01-01", "2022-03-22"]:
        dv = dfdv.query(f"week == '{ws}'")["variant"].values[0] or "others"

    dr = [e.strftime("%Y-%m-%d") for e in list(pd.date_range(ws, we, freq="d"))[:-1]]
    for d in dr:
        dominant_variants_per_day.append({"dominant_variant": dv, "day": d})

dvpd = pd.DataFrame(dominant_variants_per_day)
sns.countplot(dvpd["dominant_variant"])
dvpd.to_csv("./dominant_variants.csv")
