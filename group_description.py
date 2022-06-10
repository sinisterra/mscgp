# %%
import pandas as pd
from clickhouse_io import client, get_attributes
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
from tabulate import tabulate

# sns.set(rc={"figure.figsize": (11, 8.5)})
table = "arm.covid"
# por cada selector, determinar cuántos

group = get_attributes(table)
group = [
    g
    for g in group
    if g
    not in [
        "PAIS_ORIGEN",
        "PAIS_NACIONALIDAD",
        "INTERVAL_FECHA_INGRESO_FECHA_DEF",
        "INTERVAL_FECHA_SINTOMAS_FECHA_DEF",
        "INTERVAL_FECHA_SINTOMAS_FECHA_INGRESO",
        "ENTIDAD_NAC",
        "VARIANTE",
    ]
]

N = client().execute(f"select count(*) as cnt from {table}")[0][0]

df_groups = pd.read_csv("./sg_groups.csv")
df_groups["Attributes"] = df_groups["Attributes"].apply(lambda v: eval(v))


group_list = list(df_groups["Attributes"])
# group_list = [
#     ["EDAD", "SEXO"],
#     [
#         "DIABETES",
#         "HIPERTENSION",
#         "CARDIOVASCULAR",
#         "EPOC",
#         "ASMA",
#         "TABAQUISMO",
#         "INMUSUPR",
#         "OTRO_CASO",
#         "RENAL_CRONICA",
#         "OTRA_COM",
#         "OBESIDAD",
#         "CLASIFICACION_FINAL",
#         "NEUMONIA",
#     ],
#     ["DEFUNCION", "TIPO_PACIENTE", "INTUBADO", "UCI"],
#     ["ENTIDAD_UM", "ATN_MISMA_ENTIDAD", "ENTIDAD_RES"],
#     [
#         "TOMA_MUESTRA_ANTIGENO",
#         "TOMA_MUESTRA_LAB",
#         "RESULTADO_ANTIGENO",
#         "RESULTADO_LAB",
#     ],
# ]
# group_list = [
#     ["EDAD", "SEXO"],
#     [
#         "DIABETES",
#         "HIPERTENSION",
#         "CARDIOVASCULAR",
#         "EPOC",
#         "ASMA",
#         "TABAQUISMO",
#         "INMUSUPR",
#         "OTRO_CASO",
#         "RENAL_CRONICA",
#         "OTRA_COM",
#         "OBESIDAD",
#         "CLASIFICACION_FINAL",
#         "NEUMONIA",
#     ],
# ]
#     [
#         "DEFUNCION",
#         "EDAD",
#         "EPOCH_FECHA_DEF",
#         "INTUBADO",
#         "RESULTADO_ANTIGENO",
#         "TIPO_PACIENTE",
#         "TOMA_MUESTRA_ANTIGENO",
#         "TOMA_MUESTRA_LAB",
#         "UCI",
#     ],
#     ["NEUMONIA", "INDIGENA", "HABLA_LENGUA_INDIG",],
#     ["SEXO", "EMBARAZO",],
#     ["ATN_MISMA_ENTIDAD", "ENTIDAD_RES", "ENTIDAD_UM", "SECTOR",],
#     ["EPOCH_FECHA_INGRESO", "EPOCH_FECHA_SINTOMAS", "RESULTADO_LAB",],
#     ["ENTIDAD_NAC", "MIGRANTE", "NACIONALIDAD"],
#     ["ORIGEN"],
# ]
group_hues = {}
for (i, g) in enumerate(group_list):
    for e in g:
        group_hues[e] = i
# group_hues = {}

acc = []
for attribute in group:
    # print(attribute)
    r = client().execute(
        f"SELECT {attribute}, count(*) as cnt from {table} GROUP BY {attribute}"
    )
    res = [
        {"attribute": attribute, "value": v, "count": c, "count_norm": (c / N) * 100}
        for (v, c) in r
    ]
    acc += res

df = pd.DataFrame(acc)
# df = df[~df["value"].isin([97, 98, 99])]
df = df.sort_values("count", ascending=False).query("attribute != 'index'").round(4)
df.to_csv("./supports.csv", index=False)
df

# %%
df["count_pct"] = (df["count"].rank(ascending=True) / len(df)).round(2)


df["hue"] = df["attribute"].apply(lambda v: group_hues.get(v, "0"))
sort_order = (
    df.groupby(["attribute"]).median().sort_values(by="count_pct", ascending=True)
)

# %%
df["zscore_count"] = zscore(df["count"])
# df["hue"] = "red"


# %%
plt.figure(figsize=(8.5, 11))
order = [c for c in list(sort_order.index) if c not in ["index"]][::-1]
ax = sns.boxplot(
    data=df, y="attribute", x="count_pct", orient="h", order=order, color="#7887c2",
)
ax.set(xlabel="Percentil de soporte", ylabel="Atributo")
plt.savefig("./charts/boxplot_soporte_pct.pdf", dpi=300, bbox_inches="tight")
# %%
plt.figure(figsize=(8.5, 11))
ax = sns.boxplot(
    data=df, y="attribute", x="count", orient="h", order=order, color="#999"
)

# %%
acc_q = []
for a in df["attribute"].unique():
    dfq = df.query(f"attribute == '{a}'")
    l = dfq["count"].describe()
    l["attribute"] = a
    l["group"] = group_hues.get(a, -1)
    acc_q.append(l)
df_describe = (
    pd.DataFrame(acc_q)
    .round(3)
    .sort_values("50%", ascending=False)
    .reset_index(drop=True)
)
df_describe["range"] = df_describe["max"] - df_describe["min"]
df_describe


# %%
sns.boxplot(x="hue", data=df, y="count_pct", color="#88c9e2")
# %%
sns.displot(data=df_describe, x="count")
# %%
# graficar cuántos atributos tiene cada grupo
sns.countplot(df.drop_duplicates("attribute")["hue"])

# %%
# graficar cuántos selectores tiene cada grupo
sns.countplot(df["hue"])

# %%
n_selectors = df["hue"].value_counts()
n_attributes = df.drop_duplicates("attribute")["hue"].value_counts()
# %%
sa = pd.DataFrame({"n_attributes": n_attributes, "n_selectors": n_selectors,})

# %%
sns.jointplot(data=sa, x="n_selectors", y="n_attributes")

# %%
print(tabulate(sa, headers="keys", tablefmt="github",))

# %%
df_describe["count"] = df_describe["count"] + 1
df_describe

# %%
l = df_describe.groupby("group").sum().astype(int)
l["count"] = l["count"] - 1
l
# %%
