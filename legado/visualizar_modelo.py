import joblib
import matplotlib.pyplot as plt, pandas as pd
from sklearn.tree import plot_tree

model = joblib.load("modelo_formato_rosto-t1.joblib")

print(type(model))
print(model )
print(model["modelo"].get_params())
print(model["modelo"].classes_)
print("Classes:", model["modelo"].classes_)
print("Quantidade de features:", model["modelo"].n_features_in_)
print("Importância das features:", model["modelo"].feature_importances_)

# visualizar árvores
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    filled=True,
    feature_names=None,
    rounded=True
)

plt.show()

# visualizar árvore
tree = model.estimators_[0]

plt.figure(figsize=(25, 12))

plot_tree(
    tree,
    filled=True,
    rounded=True
)

plt.show()

# visualizar importancia das caracteristicas
importances = pd.Series(
    model.feature_importances_
).sort_values(ascending=True)

importances.plot(kind="barh", figsize=(10, 8))

plt.show()