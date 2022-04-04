# %%
import pandas as pd
import plotly.express as px
from nds import nds
import os
import multiprocessing


# df = pd.read_csv("./results/1634229828/1634230398/elites.csv").query("tp > 0")
mono = 1648609699
multi = 1648660703
sers = 1648662471
states = 1648671929
execution = f"results/{states}/0/0"

os.makedirs(f"./animate/{execution}/results", exist_ok=True)


def animate_exploration(run_path, x_d, y_d):
    (x, x_opt) = x_d
    (y, y_opt) = y_d
    # point coloring based on the latest generation
    df = pd.read_csv(f"./{run_path}/elites.csv").query("tp > 0")
    os.makedirs(f"./animate/{run_path}/exploration_{x}_{y}", exist_ok=True)
    df["generation"] = df["generation"].fillna(-1).astype(int)

    for g in df.generation.unique():
        q = nds(
            df.query(f"generation <= {g} and significant == True"),
            [x, y],
            [x_opt, y_opt],
            until_level=1,
        )

        q["pareto"] = q["level"] == 1
        q["size_m"] = pd.cut(q["markedness"], bins=10, labels=False)
        # q = q.query("level == 1")
        print(len(q))
        fig = px.scatter(
            q.sort_values("generation"),
            x=x,
            y=y,
            color="generation",
            color_discrete_map={
                False: "#9FA8DA",
                True: "#E65100",
                "infrequent": "#8eb0d5",
                "top": "orange"
                # "No, but strong": "#5C6BC0",
                # "Yes, not significant": "#FFEE58",
                # "Yes & strong": "#E65100",
            },
            labels={
                "pareto": "Pareto Set",
                "ppv": "Control Event Rate (CER)",
                "tpr": "Experimental Event Rate (EER)",
                "markedness": "Causal Effect",
                "absolute_risk": "Causal Effect",
                "r_absolute_risk": "Causal Effect (Reciprocal)",
                "ppv": "Positive Predictive Value (PPV)",
                "npv": "Negative Predictive Value (PPV)",
                "tpr": "True Positive Rate (TPR)",
                # "ppv": "False Positive Rate (FPR)",
                "support": "Support",
                "top": "Best Families",
                "infrequent": "Bottom 80%",
                "itemset_color": "Family frequency",
            },
            color_continuous_scale=px.colors.sequential.Viridis,
            color_discrete_sequence=px.colors.qualitative.D3,
            range_x=[-0.1, 1.1],
            range_y=[-0.1, 1.1],
            # range_color=[-1, 1],
            # log_x=True,
            # log_y=True,
            # range_x=[0.00005, 1.5],
            # range_y=[0.00005, 1.5],
        )
        # fig.update_xaxes(range=[0.0001, 1.1])
        # fig.update_yaxes(range=[0.0001, 1.1])
        # fig.update_layout(showlegend=False)

        fig.update_traces(marker={"size": 12})
        fig.write_image(
            f"./animate/{run_path}/exploration_{x}_{y}/{str(g).zfill(3)}.png"
        )
    os.system(
        f"convert -delay 10 -loop 0 ./animate/{run_path}/exploration_{x}_{y}/*.png ./animate/{run_path}/results/exploration_{x}_{y}.gif"
    )
    os.system(f"rm -rf ./animate/{run_path}/exploration_{x}_{y}")


def animate_pareto(run_path, x_d, y_d):
    # print(run_path, x_d, y_d)
    (x, x_opt) = x_d
    (y, y_opt) = y_d
    # point coloring based on the latest generation
    df = pd.read_csv(f"./{run_path}/elites.csv").query("tp > 0")
    os.makedirs(f"./animate/{run_path}/pareto_{x}_{y}", exist_ok=True)

    for g in df["generation"].dropna().astype(int).unique():
        q = nds(
            df.query(f"generation <= {g} and significant == True"),
            [x, y],
            [x_opt, y_opt],
            until_level=1,
        )

        q["pareto"] = q["level"] == 1
        q["size_m"] = pd.cut(q["markedness"], bins=10, labels=False)
        # q = q.query("level == 1")
        print(len(q))
        fig = px.scatter(
            q.sort_values("pareto"),
            x=x,
            y=y,
            color="pareto",
            color_discrete_map={
                False: "#9FA8DA",
                True: "#E65100",
                "infrequent": "#8eb0d5",
                "top": "orange"
                # "No, but strong": "#5C6BC0",
                # "Yes, not significant": "#FFEE58",
                # "Yes & strong": "#E65100",
            },
            labels={
                "pareto": "Pareto Set",
                "ppv": "Control Event Rate (CER)",
                "tpr": "Experimental Event Rate (EER)",
                "markedness": "Causal Effect",
                "absolute_risk": "Causal Effect",
                "r_absolute_risk": "Causal Effect (Reciprocal)",
                "ppv": "Positive Predictive Value (PPV)",
                "npv": "Negative Predictive Value (PPV)",
                "tpr": "True Positive Rate (TPR)",
                "fpr": "False Positive Rate (FPR)",
                # "ppv": "False Positive Rate (FPR)",
                "support": "Support",
                "top": "Best Families",
                "infrequent": "Bottom 80%",
                "itemset_color": "Family frequency",
            },
            color_continuous_scale=px.colors.sequential.Viridis,
            color_discrete_sequence=px.colors.qualitative.D3,
            range_x=[-0.05, 1.1],
            range_y=[-0.05, 0.3],
            # range_color=[-1, 1],
            # log_x=True,
            # log_y=True,
            # range_x=[0.00005, 1.5],
            # range_y=[0.00005, 1.5],
        )
        # fig.update_xaxes(range=[0.0001, 1.1])
        # fig.update_yaxes(range=[0.0001, 1.1])
        # fig.update_layout(showlegend=False)

        fig.update_traces(marker={"size": 12})
        fig.write_image(f"./animate/{run_path}/pareto_{x}_{y}/{str(g).zfill(3)}.png")
    os.system(
        f"convert -delay 10 -loop 0 ./animate/{run_path}/pareto_{x}_{y}/*.png ./animate/{run_path}/results/pareto_{x}_{y}.gif"
    )
    os.system(f"rm -rf ./animate/{run_path}/pareto_{x}_{y}")


def animate_value(run_path, value, x_d, y_d):
    # point coloring based on the latest generation
    (x, x_opt) = x_d
    (y, y_opt) = y_d
    # point coloring based on the latest generation
    df = pd.read_csv(f"./{run_path}/elites.csv").query("tp > 0")
    os.makedirs(f"./animate/{run_path}/value_{value}_{x}_{y}", exist_ok=True)
    df["generation"] = df["generation"].fillna(-1).astype(int)

    for g in df.generation.unique():
        q = nds(
            df.query(f"generation == {g} and significant == True"),
            [x, y],
            [x_opt, y_opt],
            until_level=1,
        )

        q["pareto"] = q["level"] == 1
        q["size_m"] = pd.cut(q["markedness"], bins=10, labels=False)
        # q = q.query("level == 1")
        print(len(q))
        fig = px.scatter(
            q.sort_values(value),
            x=x,
            y=y,
            color=value,
            color_discrete_map={
                False: "#9FA8DA",
                True: "#E65100",
                "infrequent": "#8eb0d5",
                "top": "orange"
                # "No, but strong": "#5C6BC0",
                # "Yes, not significant": "#FFEE58",
                # "Yes & strong": "#E65100",
            },
            labels={
                "pareto": "Pareto Set",
                "ppv": "Control Event Rate (CER)",
                "tpr": "Experimental Event Rate (EER)",
                "markedness": "Causal Effect",
                "absolute_risk": "Causal Effect",
                "r_absolute_risk": "Causal Effect (Reciprocal)",
                "ppv": "Positive Predictive Value (PPV)",
                "npv": "Negative Predictive Value (PPV)",
                "tpr": "True Positive Rate (TPR)",
                "support": "Support",
                "top": "Best Families",
                "infrequent": "Bottom 80%",
                "itemset_color": "Family frequency",
            },
            color_continuous_scale=px.colors.sequential.Viridis,
            color_discrete_sequence=px.colors.qualitative.D3,
            range_x=[-0.1, 1.1],
            range_y=[-0.1, 1.1],
            range_color=[-1, 1],
            # log_x=True,
            # log_y=True,
            # range_x=[0.00005, 1.5],
            # range_y=[0.00005, 1.5],
        )
        # fig.update_xaxes(range=[0.0001, 1.1])
        # fig.update_yaxes(range=[0.0001, 1.1])
        # fig.update_layout(showlegend=False)

        fig.update_traces(marker={"size": 12})
        fig.write_image(
            f"./animate/{run_path}/value_{value}_{x}_{y}/{str(g).zfill(3)}.png"
        )
    os.system(
        f"convert -delay 10 -loop 0 ./animate/{run_path}/value_{value}_{x}_{y}/*.png ./animate/{run_path}/results/value_{value}_{x}_{y}.gif"
    )
    os.system(f"rm -rf ./animate/{run_path}/value_{value}_{x}_{y}")


# %%
def do_charts(m):
    animate_exploration(execution, *m)
    animate_pareto(execution, *m)
    animate_value(execution, "aptitude", *m)
    # animate_value(execution, "r_absolute_risk", *m)
    # animate_value(execution, "confidence", *m)
    # animate_value(execution, "lift", *m)
    # animate_value(execution, "tp", *m)
    # animate_value(execution, "certainty", *m)
    # animate_value(execution, "markedness", *m)
    # animate_value(execution, "prevalence", *m)
    # animate_value(execution, "informedness", *m)


with multiprocessing.Pool(5) as pool:

    pool.map(
        do_charts,
        [
            # (("cer", "min"), ("eer", "max")),
            # (("markedness", "max"), ("prevalence", "max")),
            # (("eer", "max"), ("af_e", "max"))
            # (("af_e", "max"), ("susceptibility", "max")),
            # (("r_af_e", "max"), ("r_susceptibility", "max")),
            # (("paf", "max"), ("susceptibility", "max"))
            # (("ppv", "max"), ("tpr", "max")),
            # (("fpr", "min"), ("tpr", "max")),
            # (("ppv", "max"), ("npv", "max")),
            # (("absolute_risk", "max"), ("r_absolute_risk", "max")),
            # (("informedness", "max"), ("markedness", "max")),
            (("confidence", "max"), ("support", "max")),
        ],
    )


# animate_pareto(execution)
# animate_value(execution, "markedness")
# animate_value(execution, "informedness")
