# %%
import pandas as pd
from clickhouse_io import c_get_confusion_matrix
from measures import evaluate_confusion_matrix

table = "arm.covid"
rule = (
    (
        ("RESULTADO_ANTIGENO", 1),
        ("INTUBADO", 2),
        ("HIPERTENSION", 1),
        ("TABAQUISMO", 1),
        ("OBESIDAD", 1),
        ("DIABETES", 1),
        ("SEXO", 1)
        # ("INMUSUPR", 1),
        # ("NEUMONIA", 1),
        # ("HIPERTENSION", 1),
        # ("INTUBADO", 2),
        # ("RESULTADO_ANTIGENO", 1),
        # ("SEXO", 2),
        # ("TABAQUISMO", 1),
        # ("DIABETES", 2),
        # ("OBESIDAD", 1)
        # ("HABLA_LENGUA_INDIG", 1),
        # ("OBESIDAD", 1),
        # ("SEXO", 2),
        # ("TIPO_PACIENTE", 2),
        # ("RESULTADO_ANTIGENO", 1),
        # ("INMUSUPR", 1),
        # ("TABAQUISMO", 2),
        # ("CARDIOVASCULAR", 2),
        # ("INDIGENA", 2),
        # ("EPOC", 2),
        # ("INDIGENA", 1),
        # ("OBESIDAD", 1),
        # ("UCI", 2),
        # ("ASMA", 1)
        # ("TOMA_MUESTRA_ANTIGENO", 2),
        # ("TABAQUISMO", 1),
        # ("INMUSUPR", 2),
        # ("ASMA", 2)
        # ("CARDIOVASCULAR", 1),
    ),
    (("DEFUNCION", 1), ("NEUMONIA", 2)),
)
sels = [*[a[0] for a in rule[0]], *[a[0] for a in rule[1]]]
# %%
def eval_modification(var):
    ct = c_get_confusion_matrix(table, rule)[0]
    cmt = evaluate_confusion_matrix(ct)
    cmtv = cmt["markedness"]
    cmtv
    c1 = c_get_confusion_matrix(table, rule, filtered=f" WHERE {var} = 1",)[0]
    cm1 = evaluate_confusion_matrix(c1)
    cm1v = cm1["markedness"]
    # (cm1v, cm1v - cmtv)

    c2 = c_get_confusion_matrix(table, rule, filtered=f" WHERE {var} = 2",)[0]
    cm2 = evaluate_confusion_matrix(c2)
    cm2v = cm2["markedness"]
    # (cm2v, cm2v - cmtv)

    zm = pd.Series([cm1v, cm2v]).mean()
    zstd = pd.Series([cm1v, cm2v]).std()

    pos1 = cm1["tp"] > 0 and cm1["fp"] > 0 and cm1["tn"] > 0 and cm1["fn"] > 0

    pos2 = cm2["tp"] > 0 and cm2["tp"] > 0 and cm2["tp"] > 0 and cm2["tp"] > 0
    return [
        {
            "var": var,
            "val": 1,
            "ate": cmtv,
            "prev": cm1["prevalence"],
            "v": cm1v,
            "sig": cm1["significant"] and pos1,
            # "c2": cm2v,
            # "prevc1": cm1["prevalence"],
            # "prevc2": cm2["prevalence"],
            # "mean": zm,
            # "std": zstd,
        },
        {
            "var": var,
            "val": 2,
            "ate": cmtv,
            "prev": cm2["prevalence"],
            "v": cm2v,
            "sig": cm2["significant"] and pos2,
            # "c2": cm2v,
            # "prevc1": cm1["prevalence"],
            # "prevc2": cm2["prevalence"],
            # "mean": zm,
            # "std": zstd,
        },
    ]


# %%
import itertools

acc = []
l = [
    "INMUSUPR",
    "SEXO",
    "HIPERTENSION",
    "DIABETES",
    "OBESIDAD",
    "EPOC",
    "TIPO_PACIENTE",
    "RENAL_CRONICA",
    "CARDIOVASCULAR",
    "ASMA",
    "TABAQUISMO",
    "NEUMONIA",
    "INTUBADO",
    "UCI",
    "EMBARAZO",
    "TOMA_MUESTRA_ANTIGENO",
    "INDIGENA",
    "RESULTADO_ANTIGENO",
    "HABLA_LENGUA_INDIG",
]
for v in l:
    if v not in sels:
        acc += eval_modification(v)

pd.DataFrame(acc).sort_values(["v", "prev"], ascending=(False, True)).query(
    "v > ate and sig == True"
)

# %%
evald = pd.DataFrame(
    [evaluate_confusion_matrix(c_get_confusion_matrix(table, rule)[0])]
)
evald[["markedness", "significant", "tp", "fp", "tn", "fn"]]

# %%
