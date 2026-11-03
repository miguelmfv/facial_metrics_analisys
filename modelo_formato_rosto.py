"""Treina o modelo de FORMATO DE ROSTO.

Uso: python main.py [dados.csv] [--alvo source_type] [--saida modelo_formato_rosto.joblib] [--sobrescrever]

--alvo          coluna do CSV com a classe de cada rosto (dataset de formato de rosto: source_type)
--saida         arquivo .joblib de saida
--sobrescrever  permite substituir um .joblib que ja existe (por padrao, aborta)
"""
import argparse
import os
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

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default="faces-frame-type-frame-id.csv")
ap.add_argument("--alvo", default="source_type")
ap.add_argument("--saida", default="/resultados/modelo_formato_rosto.joblib")
ap.add_argument("--sobrescrever", action="store_true")
args = ap.parse_args()

if os.path.exists(args.saida) and not args.sobrescrever:
    sys.exit(f"ERRO: '{args.saida}' ja existe e seria sobrescrito. Use outro --saida ou --sobrescrever.")

df, X = build(args.csv)          # X = features calculadas; df = CSV original
if args.alvo not in df.columns:
    sys.exit(f"ERRO: coluna '{args.alvo}' nao existe. Colunas: {list(df.columns)[:15]}...")
y = df[args.alvo]                # classe de cada rosto
print(f"{len(X)} rostos, {X.shape[1]} features\n{y.value_counts()}\n")
if y.nunique() > 12:
    sys.exit(f"ERRO: '{args.alvo}' tem {y.nunique()} valores distintos; parece um identificador, nao uma classe.")

modelos = {
    "RandomForest": RandomForestClassifier(500, max_depth=10, min_samples_leaf=20, random_state=42, max_features=0.3, class_weight='balanced_subsample'),
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
joblib.dump({"modelo": rf, "features": list(X.columns), "alvo": args.alvo}, args.saida)
print(f"\nModelo salvo em {args.saida} (classes: {list(rf.classes_)})")