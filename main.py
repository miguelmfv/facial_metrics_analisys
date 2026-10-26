import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features_faciais import build

CSV = sys.argv[1] if len(sys.argv) > 1 else "faces - faces (1).csv"

df, X = build(CSV)          # X = features calculadas; df = CSV original
y = df["source_type"]       # classe de cada rosto
print(f"{len(X)} rostos, {X.shape[1]} features\n{y.value_counts()}\n")

modelos = {
    "RandomForest": RandomForestClassifier(1500, max_depth=10, min_samples_leaf=3, random_state=42),
    "RegLogistica": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=4000)),
}

cv = StratifiedKFold(10, shuffle=True, random_state=42)
for nome, modelo in modelos.items():
    acc = cross_val_score(modelo, X, y, cv=cv)
    print(f"{nome}: {acc.mean():.2f} ± {acc.std():.2f}")

# matriz de confusão e métricas por classe (RandomForest)
rf = modelos["RandomForest"]
pred = cross_val_predict(rf, X, y, cv=cv)
labels = sorted(y.unique())
print("\nMatriz de confusão (linhas = real, colunas = previsto):", labels)
print(confusion_matrix(y, pred, labels=labels))
print(classification_report(y, pred))

# treina com tudo, mostra importância das features e salva o modelo
rf.fit(X, y)
print(pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False).head(10).round(3))
joblib.dump({"modelo": rf, "features": list(X.columns)}, "modelo_formato_rosto.joblib")
print("\nModelo salvo em modelo_formato_rosto.joblib")