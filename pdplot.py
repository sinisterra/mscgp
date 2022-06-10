# %%
import pandas as pd
import plotly.express as px

df = pd.read_csv("./pareto3d.csv")

fig = px.parallel_coordinates(
    df,
    # color="species_id",
    # labels={
    #     "species_id": "Species",
    #     "sepal_width": "Sepal Width",
    #     "sepal_length": "Sepal Length",
    #     "petal_width": "Petal Width",
    #     "petal_length": "Petal Length",
    # },
    # color_continuous_scale=px.colors.diverging.Tealrose,
    # color_continuous_midpoint=2,
)
fig.show()

# %%
