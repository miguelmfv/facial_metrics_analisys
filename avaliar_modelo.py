"""Uso: python avaliar_modelo.py [modelo.joblib] [dados.csv]"""
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features_faciais import build

MODELO = sys.argv[1] if len(sys.argv) > 1 else "modelo_formato_rosto.joblib"
CSV = sys.argv[2] if len(sys.argv) > 2 else "faces-teste-supervisionado-formato-rosto.csv"

# 1) carrega o modelo salvo (dicionário com o modelo e a ordem das features)
pacote = joblib.load(MODELO)
modelo, colunas = pacote["modelo"], pacote["features"]

# 2) calcula as features do CSV novo, com as mesmas colunas e na mesma ordem do treino
df, X = build(CSV)

# confere se as features calculadas agora são as mesmas com que o modelo foi treinado
faltando = [c for c in colunas if c not in X.columns]
sobrando = [c for c in X.columns if c not in colunas]
if faltando:
    print("ERRO: o modelo espera features que o features_faciais.py atual não gera.")
    print(f"  Faltando ({len(faltando)}): {faltando}")
    print(f"  Geradas agora mas desconhecidas pelo modelo ({len(sobrando)}): {sobrando}")
    print("  Use o MESMO features_faciais.py com que o .joblib foi treinado, ou retreine o modelo.")
    sys.exit(1)
X = X[colunas]

# linhas com valores inválidos (NaN) não podem ser previstas
validas = X.notna().all(axis=1)
if (~validas).any():
    print(f"Aviso: {(~validas).sum()} linhas ignoradas por features inválidas")
df, X = df[validas].reset_index(drop=True), X[validas].reset_index(drop=True)

# 3) previsões e probabilidades por classe
classes = list(modelo.classes_)
proba = modelo.predict_proba(X)
pred = np.array(classes)[proba.argmax(axis=1)]

saida = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
saida.insert(0, "previsto", pred)
saida.insert(0, "arquivo", df["file_name"] if "file_name" in df.columns else df.index)

# 4) se o CSV tem a classe real, calcula as métricas
if "source_type" in df.columns:
    # padroniza o texto do rótulo (espaços extras e maiúsculas) e trata vazios como NaN
    y_bruto = df["source_type"].astype("string").str.strip().str.lower().replace("", pd.NA)
    saida.insert(1, "real", y_bruto)

    # só avalia as linhas rotuladas com uma das classes que o modelo conhece
    rotulada = y_bruto.notna() & y_bruto.isin(classes)
    sem_rotulo = int(y_bruto.isna().sum())
    desconhecido = y_bruto[y_bruto.notna() & ~y_bruto.isin(classes)]
    if sem_rotulo:
        print(f"Aviso: {sem_rotulo} linhas sem rótulo em source_type foram ignoradas na avaliação")
    if len(desconhecido):
        print(f"Aviso: {len(desconhecido)} linhas com rótulo fora das classes {classes} foram ignoradas:")
        print(desconhecido.value_counts().to_string())

    y = y_bruto[rotulada].astype(str).to_numpy()
    p_av, proba_av = pred[rotulada.to_numpy()], proba[rotulada.to_numpy()]
    saida["acertou"] = np.where(rotulada, saida["real"] == saida["previsto"], pd.NA)

    if len(y) == 0:
        print("Nenhuma linha rotulada com uma classe válida; não há como calcular a acurácia.")
    else:
        top2 = np.array(classes)[np.argsort(proba_av, axis=1)[:, -2:]]
        top2_acc = np.mean([r in t for r, t in zip(y, top2)])

        print(f"Rostos avaliados: {len(y)}")
        print(f"Acurácia: {accuracy_score(y, p_av):.3f}")
        print(f"Top-2 accuracy: {top2_acc:.3f}")
        print("\nMatriz de confusão (linhas = real, colunas = previsto):", classes)
        print(confusion_matrix(y, p_av, labels=classes))
        print(classification_report(y, p_av, labels=classes, zero_division=0))

saida.round(3).to_csv("resultados_previsao.csv", index=False)
print("Previsões salvas em resultados_previsao.csv")