# %%
import pandas as pd
from scipy.spatial import ConvexHull
import alphashape
import matplotlib.pyplot as plt
from descartes import PolygonPatch


df = pd.read_csv("./results/1648311288/0/0/uniques.csv")

hull_points = []
for (i, r) in df.iterrows():
    hull_points.append([r["absolute_risk"], r["full_informedness"]])

# hull = ConvexHull(hull_points)

# df["convex_hull"] = df.index.isin(hull.vertices)
# df[["repr", "convex_hull", "paf", "absolute_risk"]].query("convex_hull == True")

# %%
alpha_shape = alphashape.alphashape(hull_points)
fig, ax = plt.subplots()
ax.scatter(*zip(*hull_points))
ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
plt.show()

# %%
