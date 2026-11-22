"""Treina e versiona o classificador de FORMATO DE ARMACAO (D-79, D-80, D-81).

Este script e a segunda etapa do pipeline original (`legado/pipeline_rosto_armacao.py`),
isolada e transformada em artefato reusavel: la ela era medida em validacao
cruzada e descartada; aqui ela e medida **e** salva, para que a inferencia do
pacote `vizo_rosto` possa rodar os dois classificadores em sequencia.

A cadeia que o artefato deste script completa::

    medicao -> 29 features -> classificador de ROSTO -> p_rosto_*
            -> classificador de ARMACAO -> ranking dos 7 formatos

O que o script faz, em ordem:

1. le o CSV de extracao e calcula as 29 features com o **mesmo**
   ``vizo_rosto.features_faciais.extract`` que a inferencia usa (se fossem
   codigos diferentes, o numero da monografia e o do app divergiriam);
2. carrega o modelo de formato de rosto congelado (D-48) e calcula, para cada
   imagem, a probabilidade dos 4 formatos de rosto;
3. monta o conjunto de entrada escolhido (``features``, ``probs_rosto`` ou
   ``features+probs_rosto``);
4. **mede** tres estimadores por validacao cruzada agrupada por modelo de
   armacao (``StratifiedGroupKFold``), junto com as linhas de base do acaso;
5. escolhe o estimador pela metrica de selecao (default: acuracia top-3, que e
   a metrica que corresponde ao produto -- um ranking de 3 formatos);
6. reajusta o escolhido em todas as linhas e grava o ``.joblib`` com as
   metricas medidas e a procedencia dentro.

**Agrupar por ``frame_id`` nao e detalhe.** Sem isso, fotos do mesmo modelo de
armacao caem no treino e no teste, e a acuracia sobe por memorizacao do produto,
nao por sinal facial. Foi a validacao agrupada que revelou o resultado negativo
deste trabalho, e nao ha razao para abandona-la ao versionar o modelo.

**O que este script nao esconde:** o classificador de armacao fica no nivel do
acaso. As metricas medidas vao impressas e gravadas no artefato de proposito,
inclusive a comparacao com o ranking fixo das 3 classes mais frequentes, que na
rodada anterior **superou** o modelo em top-3 (0,69 contra 0,627). Quem carregar
o artefato tem como ler isso sem ir atras de documento nenhum.

Uso tipico (na raiz do repositorio)::

    python treinar_modelo_armacao.py \
        --csv arquivos_csv/faces-frame-type-frame-id.csv \
        --modelo-rosto modelos/modelo_formato_rosto_sem_linhas_branco.joblib \
        --saida modelos/modelo_formato_armacao.joblib

Nada aqui imprime porcentagem para o backend: a fronteira de D-66 e no pacote
`vizo_rosto`, nao neste script, que e de bancada.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# O pacote vive em src/; permite rodar o script sem instalar o projeto.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vizo_rosto import carregar  # noqa: E402
from vizo_rosto.features_faciais import extract  # noqa: E402
from vizo_rosto.modelo_armacao import (  # noqa: E402
    CLASSES_ESPERADAS_ARMACAO,
    CONJUNTOS_ENTRADA,
    PREFIXO_PROB_ROSTO,
)

#: Os tres estimadores candidatos, com a mesma configuracao do pipeline
#: original (``legado/comparar_modelos.py``), para que os numeros medidos aqui
#: sejam comparaveis aos da monografia. O booleano diz se o ajuste usa
#: ``sample_weight`` balanceado.
CANDIDATOS: dict[str, tuple[Any, bool]] = {}


def montar_candidatos(seed: int) -> dict[str, tuple[Any, bool]]:
    """Os tres estimadores comparados no pipeline original, com a mesma config."""
    imp = ("imp", SimpleImputer(strategy="median"))
    return {
        "Regressao logistica": (
            Pipeline(
                [
                    imp,
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
                ]
            ),
            False,
        ),
        "Random Forest (mais regularizada)": (
            Pipeline(
                [
                    imp,
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=6,
                            min_samples_leaf=20,
                            max_features=0.3,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            False,
        ),
        "HistGradientBoosting": (
            Pipeline(
                [
                    imp,
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_depth=4,
                            learning_rate=0.05,
                            max_iter=200,
                            min_samples_leaf=40,
                            l2_regularization=1.0,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            True,
        ),
    }


def montar_baselines(seed: int) -> dict[str, tuple[Any, bool]]:
    """As duas linhas de base do acaso, medidas nos mesmos folds."""
    return {
        "Baseline: classe majoritaria": (DummyClassifier(strategy="most_frequent"), False),
        "Baseline: sorteio por proporcao": (
            DummyClassifier(strategy="stratified", random_state=seed),
            False,
        ),
    }


def ajustar(estimador: Any, usa_peso: bool, X: pd.DataFrame, y: np.ndarray) -> Any:
    """Ajusta o estimador, aplicando peso balanceado quando ele pede."""
    if usa_peso:
        peso = compute_sample_weight("balanced", y)
        estimador.fit(X, y, clf__sample_weight=peso)
    else:
        estimador.fit(X, y)
    return estimador


def acuracia_top_k(proba: np.ndarray, classes: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Fracao de linhas em que a classe verdadeira esta entre as ``k`` primeiras.

    E a metrica que corresponde ao produto: o contrato com o backend leva um
    ranking de 3 formatos, e o que importa e a classe certa estar nele, nao ser
    a primeira.
    """
    topk = np.argsort(proba, axis=1)[:, -k:]
    rotulos = np.asarray(classes)[topk]
    return float(np.mean([verdade in linha for verdade, linha in zip(y, rotulos)]))


def extrair_features(df: pd.DataFrame, nomes_esperados: tuple[str, ...]) -> pd.DataFrame:
    """Calcula as 29 features de cada linha do CSV, com o codigo da inferencia.

    Linhas em que ``extract`` falha viram linha de ``NaN``, que o
    ``SimpleImputer`` dos pipelines resolve por mediana de coluna. Isso e
    legitimo **no treino**, onde existe coluna inteira para tirar mediana, e e
    exatamente o que o pipeline original fazia. Na inferencia de uma medicao
    unica nao existe mediana, e por isso o pacote **recusa** a medicao com
    feature nao finita (D-56) em vez de preencher.
    """
    linhas: list[dict[str, float]] = []
    falhas = 0
    for _, linha in df.iterrows():
        try:
            linhas.append({str(k): float(v) for k, v in extract(linha).items()})
        except Exception:
            linhas.append({})
            falhas += 1
    X = pd.DataFrame(linhas, index=df.index).replace([np.inf, -np.inf], np.nan)
    if falhas:
        print(f"Aviso: extract() falhou em {falhas} de {len(df)} linhas (viraram NaN).")

    faltando = [nome for nome in nomes_esperados if nome not in X.columns]
    if faltando:
        raise SystemExit(
            f"O extract() atual nao gerou as features {faltando}, que o modelo de "
            "rosto espera. O features_faciais.py e o .joblib precisam ser o mesmo par "
            "(D-50)."
        )
    # Ordem do treino do modelo de rosto, e so as colunas dele.
    return X[list(nomes_esperados)]


def probabilidades_de_rosto(modelo_rosto: Any, X: pd.DataFrame) -> pd.DataFrame:
    """As 4 probabilidades de formato de rosto, uma coluna por classe.

    Preenche ``NaN`` por mediana de coluna antes de prever, como o pipeline
    original fazia: o RandomForest aceitaria o ``NaN`` em silencio, e aqui
    existe a coluna inteira para tirar a mediana.
    """
    Xin = X.fillna(X.median(numeric_only=True))
    proba = modelo_rosto.estimador.predict_proba(Xin)
    classes = modelo_rosto.classes
    return pd.DataFrame(
        proba,
        columns=[f"{PREFIXO_PROB_ROSTO}{classe}" for classe in classes],
        index=X.index,
    )


def montar_entrada(conjunto: str, X_features: pd.DataFrame, P_rosto: pd.DataFrame) -> pd.DataFrame:
    """Monta a matriz de entrada do classificador de armacao (D-80)."""
    if conjunto == "features":
        return X_features
    if conjunto == "probs_rosto":
        return P_rosto
    if conjunto == "features+probs_rosto":
        return pd.concat([X_features, P_rosto], axis=1)
    raise SystemExit(f"Conjunto de entrada desconhecido: {conjunto!r}. Use um de {CONJUNTOS_ENTRADA}.")


def medir(
    X: pd.DataFrame,
    y: pd.Series,
    grupos: pd.Series,
    folds: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Mede candidatos e baselines por validacao cruzada agrupada.

    Returns:
        A tabela impressa e um dicionario ``nome -> metricas`` com tipos
        nativos, pronto para ser gravado no artefato.
    """
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    particoes = list(cv.split(X, y, grupos))
    y_arr = y.to_numpy()

    resultados: dict[str, dict[str, float]] = {}
    linhas: list[dict[str, Any]] = []

    for nome, (estimador, usa_peso) in {**montar_baselines(seed), **montar_candidatos(seed)}.items():
        acc, bal, f1m, top3 = [], [], [], []
        for treino, teste in particoes:
            ajustado = ajustar(estimador, usa_peso, X.iloc[treino], y_arr[treino])
            previsto = ajustado.predict(X.iloc[teste])
            acc.append(float(np.mean(previsto == y_arr[teste])))
            bal.append(float(balanced_accuracy_score(y_arr[teste], previsto)))
            f1m.append(float(f1_score(y_arr[teste], previsto, average="macro")))
            if hasattr(ajustado, "predict_proba") and not nome.startswith("Baseline"):
                top3.append(
                    acuracia_top_k(
                        ajustado.predict_proba(X.iloc[teste]), ajustado.classes_, y_arr[teste]
                    )
                )
        metricas = {
            "acuracia": float(np.mean(acc)),
            "acuracia_balanceada": float(np.mean(bal)),
            "acuracia_balanceada_desvio": float(np.std(bal)),
            "f1_macro": float(np.mean(f1m)),
            "top3": float(np.mean(top3)) if top3 else float("nan"),
        }
        resultados[nome] = metricas
        linhas.append({"modelo": nome, **metricas})

    tabela = pd.DataFrame(linhas).set_index("modelo")
    return tabela, resultados


def referencias_do_acaso(y: pd.Series) -> dict[str, float]:
    """Os pontos de comparacao que dizem se o modelo aprendeu algo.

    O ultimo e o que mais importa para este produto: um ranking **fixo** das 3
    classes mais frequentes, igual para todo rosto, acerta esta fracao das
    linhas no top-3. Se o modelo nao superar isso, o ranking que ele produz nao
    vale mais do que uma lista chumbada por frequencia.
    """
    proporcoes = (y.value_counts(normalize=True)).to_numpy()
    k = len(proporcoes)
    return {
        "acuracia_uniforme": float(1 / k),
        "acuracia_por_proporcao": float(np.sum(proporcoes**2)),
        "acuracia_classe_majoritaria": float(proporcoes[0]),
        "top3_uniforme": float(min(3, k) / k),
        "top3_ranking_fixo_das_3_mais_frequentes": float(np.sort(proporcoes)[::-1][:3].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="arquivos_csv/faces-frame-type-frame-id.csv")
    ap.add_argument("--sep", default=";")
    ap.add_argument("--target", default="source_type", help="coluna do formato de armacao")
    ap.add_argument("--group", default="frame_id", help="coluna do modelo de armacao (grupo da CV)")
    ap.add_argument("--modelo-rosto", default="modelos/modelo_formato_rosto_sem_linhas_branco.joblib")
    ap.add_argument("--entrada", default="features+probs_rosto", choices=list(CONJUNTOS_ENTRADA))
    ap.add_argument("--excluir", nargs="*", default=[], help="classes de armacao a descartar")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--selecionar-por",
        default="top3",
        choices=["top3", "acuracia", "acuracia_balanceada", "f1_macro"],
        help="metrica que decide qual estimador e versionado (default: top3, "
        "que e a metrica do produto -- um ranking de 3 formatos)",
    )
    ap.add_argument("--saida", default="modelos/modelo_formato_armacao.joblib")
    args = ap.parse_args()

    caminho_csv = Path(args.csv)
    if not caminho_csv.is_file():
        raise SystemExit(f"CSV nao encontrado: {caminho_csv}")

    df = pd.read_csv(caminho_csv, sep=args.sep, index_col=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for coluna in (args.target, args.group):
        if coluna not in df.columns:
            raise SystemExit(f"Coluna '{coluna}' nao existe no CSV. Colunas: {list(df.columns)[:12]}...")

    if args.excluir:
        antes = len(df)
        df = df[~df[args.target].astype(str).isin(args.excluir)].reset_index(drop=True)
        print(f"Classes excluidas {args.excluir}: {antes - len(df)} linhas removidas.")

    y = df[args.target].astype(str)
    grupos = df[args.group].astype(str)

    print(f"CSV: {caminho_csv}  |  linhas: {len(df)}  |  modelos de armacao: {grupos.nunique()}")
    print("\nLinhas e modelos distintos por classe:")
    print(
        pd.DataFrame({"classe": y, "grupo": grupos})
        .groupby("classe")
        .agg(linhas=("grupo", "size"), modelos=("grupo", "nunique"))
        .sort_values("linhas", ascending=False)
        .to_string()
    )

    modelo_rosto = carregar(args.modelo_rosto)
    print(f"\nModelo de rosto: {Path(args.modelo_rosto).name}  |  classes: {list(modelo_rosto.classes)}")

    X_features = extrair_features(df, modelo_rosto.features)
    P_rosto = probabilidades_de_rosto(modelo_rosto, X_features)
    print("Distribuicao do formato de rosto previsto (top1) nestas imagens:")
    top1 = P_rosto.idxmax(axis=1).str.replace(PREFIXO_PROB_ROSTO, "", regex=False)
    print((top1.value_counts(normalize=True) * 100).round(1).to_string())

    X = montar_entrada(args.entrada, X_features, P_rosto)
    print(f"\nConjunto de entrada: {args.entrada}  |  colunas: {X.shape[1]}")

    minimo = grupos.groupby(y).nunique().min()
    if minimo < args.folds:
        print(
            f"Aviso: a classe mais rara tem {minimo} modelos de armacao, menos que os "
            f"{args.folds} folds. Considere --excluir nela ou menos folds."
        )

    tabela, metricas = medir(X, y, grupos, args.folds, args.seed)
    print(f"\n{'=' * 78}\nMEDIDO ({args.folds} folds agrupados por {args.group})\n{'=' * 78}")
    print(tabela.round(3).to_string())

    acaso = referencias_do_acaso(y)
    print("\nReferencias do acaso:")
    for nome, valor in acaso.items():
        print(f"  {nome}: {valor:.3f}")

    candidatos = list(montar_candidatos(args.seed))
    escolhido = max(candidatos, key=lambda nome: metricas[nome][args.selecionar_por])
    print(f"\nEscolhido por '{args.selecionar_por}': {escolhido}")

    referencia = acaso["top3_ranking_fixo_das_3_mais_frequentes"]
    if metricas[escolhido]["top3"] <= referencia:
        print(
            f"ATENCAO: o top-3 do modelo escolhido ({metricas[escolhido]['top3']:.3f}) NAO "
            f"supera o ranking fixo das 3 classes mais frequentes ({referencia:.3f}).\n"
            "         O ranking que este artefato produz nao vale mais, em acerto, que uma "
            "lista chumbada por frequencia. Isso nao impede o uso -- e a decisao D-79 --, "
            "mas tem de estar no aviso que viaja com a recomendacao."
        )

    estimador, usa_peso = montar_candidatos(args.seed)[escolhido]
    ajustar(estimador, usa_peso, X, y.to_numpy())

    classes_obtidas = tuple(str(c) for c in estimador.classes_)
    if args.target == "source_type" and not args.excluir:
        esperadas = tuple(sorted(CLASSES_ESPERADAS_ARMACAO))
        if classes_obtidas != esperadas:
            print(
                f"Aviso: as classes treinadas {classes_obtidas} nao sao as "
                f"{esperadas} que o pacote espera por default. A carga vai exigir "
                "classes_esperadas=None ou a constante atualizada."
            )

    pacote = {
        "modelo": estimador,
        "features": list(X.columns),
        "alvo": args.target,
        "conjunto_entrada": args.entrada,
        "classes": list(classes_obtidas),
        "metricas": metricas,
        "metrica_de_selecao": args.selecionar_por,
        "modelo_escolhido": escolhido,
        "referencias_do_acaso": acaso,
        "procedencia": {
            "csv": str(caminho_csv),
            "linhas": int(len(df)),
            "grupos": int(grupos.nunique()),
            "modelo_rosto": str(Path(args.modelo_rosto).name),
            "folds": int(args.folds),
            "seed": int(args.seed),
            "excluidas": list(args.excluir),
            "treinado_em": date.today().isoformat(),
            "sklearn": sklearn.__version__,
            "python": platform.python_version(),
            "decisoes": ["D-79", "D-80", "D-81"],
        },
    }

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, saida)
    print(f"\nArtefato salvo em: {saida}")
    print(json.dumps(pacote["procedencia"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
