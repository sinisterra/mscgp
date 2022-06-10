# %%
import pandas as pd
import plotly.express as px


df = pd.read_csv("./pareto3.csv")
df["key"] = "pareto"
x = "susceptibility"
y = "paf"
df[x] = df[x] * 100
df[y] = df[y] * 100

titles = {
    "absolute_risk": "Efecto causal (%)",
    "r_absolute_risk": "Efecto causal recíproco (%)",
    "susceptibility": "Susceptiblidad (%)",
    "paf": "Impacto en la población (%)",
}
fig = px.line(df, x=x, y=y, markers=True)
fig.update_traces(marker={"size": 12, "line": {"width": 2}}, line={"width": 4})
fig.update_layout(xaxis_title=titles[x], yaxis_title=titles[y])
fig.show()
fig.write_image("pareto_b3.pdf")
# %%
