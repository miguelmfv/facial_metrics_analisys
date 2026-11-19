"""
Compara modelos para prever frame_shape a partir das features faciais.

Uso:
    python comparar_modelos_armacao.py --csv seu_arquivo.csv --sep ";" \
        --target frame_shape --group model_id --drop image_path
    # (coloque este arquivo na mesma pasta do features_faciais.py)

Opcoes de features:
    --features derivadas   usa features_faciais.extract() (padrao)
    --features brutas      usa as colunas numericas do CSV (coordenadas)
    --features ambas       junta os dois
    --sem-olhos            remove features que dependem de olhos/sobrancelhas

Pontos importantes:
- Validacao cruzada AGRUPADA por model_id (StratifiedGroupKFold): imagens do
  mesmo modelo de armacao nunca ficam em treino e teste ao mesmo tempo.
- Compara com baselines (classe majoritaria e sorteio pela proporcao).
- Mostra acuracia de treino x teste (diagnostico de overfitting).
- Mostra top-3, que e o que importa para recomendacao.
"""
import argparse
import importlib
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")


# Features de features_faciais.extract() que dependem de olhos/sobrancelhas
# (podem ser distorcidas pela armacao nas fotos de review)
DEPENDEM_OLHOS_SOBRANCELHAS = [
    "pos_sobrancelha", "pos_olho", "terco_testa", "terco_meio",
    "larg_testa_sobr", "larg_sobrancelha", "larg_olho",
    "testa_maca", "mandib_maca", "mandib_testa", "maca_vs_max", "olhos_larg",
]


def extrair_derivadas(df, modulo):
    """Chama features_faciais.extract() em cada linha do CSV."""
    linhas, falhas = [], 0
    for _, row in df.iterrows():
        try:
            linhas.append(modulo.extract(row))
        except Exception:
            linhas.append({})
            falhas += 1
    if falhas:
        print(f"Aviso: extract() falhou em {falhas} linhas (viram NaN).")
    return pd.DataFrame(linhas, index=df.index).replace([np.inf, -np.inf], np.nan)


def carregar(args):
    sys.path[:0] = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    df = pd.read_csv(args.csv, sep=args.sep, index_col=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for col in (args.target, args.group):
        if col not in df.columns:
            raise SystemExit(f"Coluna '{col}' nao encontrada. Colunas: {list(df.columns)}")

    # filtros opcionais (usam as colunas relative_path e pose_yaw do CSV de extracao)
    filtro = getattr(args, "filtro_caminho", None)
    if filtro:
        if "relative_path" not in df.columns:
            raise SystemExit("--filtro-caminho exige a coluna 'relative_path' no CSV.")
        caminhos = df["relative_path"].astype(str).str.replace("\\", "/", regex=False)
        antes = len(df)
        df = df[caminhos.str.contains(filtro, regex=False)].reset_index(drop=True)
        print(f"Filtro de caminho '{filtro}': {len(df)} de {antes} linhas mantidas.")
    max_yaw = getattr(args, "max_yaw", None)
    if max_yaw is not None:
        if "pose_yaw" not in df.columns:
            raise SystemExit("--max-yaw exige a coluna 'pose_yaw' no CSV.")
        antes = len(df)
        df = df[df["pose_yaw"].abs() <= max_yaw].reset_index(drop=True)
        print(f"Filtro de pose (|yaw| <= {max_yaw}): {len(df)} de {antes} linhas mantidas.")

    if args.excluir:
        antes = len(df)
        df = df[~df[args.target].astype(str).isin(args.excluir)].reset_index(drop=True)
        print(f"Classes excluidas {args.excluir}: {antes - len(df)} linhas removidas.")

    y_raw = df[args.target].astype(str)
    groups = df[args.group].astype(str)

    excluir = {args.target, args.group, *args.drop}
    brutas = df.drop(columns=[c for c in excluir if c in df.columns])
    brutas = brutas.select_dtypes(include=[np.number])

    if args.features == "brutas":
        X = brutas
    else:
        ff = importlib.import_module(args.modulo)
        derivadas = extrair_derivadas(df, ff)
        if args.sem_olhos:
            derivadas = derivadas.drop(columns=DEPENDEM_OLHOS_SOBRANCELHAS, errors="ignore")
        X = derivadas if args.features == "derivadas" else pd.concat([derivadas, brutas], axis=1)

    if X.shape[1] == 0:
        raise SystemExit("Nenhuma feature numerica restou. Confira --drop.")

    print(f"Linhas: {len(df)} | Features: {X.shape[1]} (modo: {args.features}"
          f"{', sem olhos/sobrancelhas' if args.sem_olhos and args.features != 'brutas' else ''})")
    print("\nLinhas por classe e modelos (grupos) distintos por classe:")
    resumo = (
        pd.DataFrame({"classe": y_raw, "grupo": groups})
        .groupby("classe")
        .agg(linhas=("grupo", "size"), modelos=("grupo", "nunique"))
    )
    print(resumo.to_string())
    nomes = list(X.columns)
    mostra = nomes if len(nomes) <= 40 else nomes[:40] + ["..."]
    print(f"\nFeatures usadas ({len(nomes)}): {mostra}\n")
    args._df = df  # CSV ja filtrado, para quem chama poder recuperar relative_path etc.
    return X, y_raw, groups


def montar_modelos(seed):
    imp = ("imp", SimpleImputer(strategy="median"))
    # (pipeline, usa_sample_weight_balanceado)
    return {
        "Baseline: classe majoritaria": (DummyClassifier(strategy="most_frequent"), False),
        "Baseline: sorteio por proporcao": (
            DummyClassifier(strategy="stratified", random_state=seed),
            False,
        ),
        "Regressao logistica": (
            Pipeline([
                imp,
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]),
            False,
        ),
        "Random Forest (sua config)": (
            Pipeline([
                imp,
                ("clf", RandomForestClassifier(
                    n_estimators=200, max_depth=10, min_samples_leaf=3,
                    max_features="sqrt", class_weight="balanced",
                    random_state=seed, n_jobs=-1)),
            ]),
            False,
        ),
        "Random Forest (mais regularizada)": (
            Pipeline([
                imp,
                ("clf", RandomForestClassifier(
                    n_estimators=300, max_depth=6, min_samples_leaf=20,
                    max_features=0.3, class_weight="balanced_subsample",
                    random_state=seed, n_jobs=-1)),
            ]),
            False,
        ),
        "HistGradientBoosting": (
            Pipeline([
                imp,
                ("clf", HistGradientBoostingClassifier(
                    max_depth=4, learning_rate=0.05, max_iter=200,
                    min_samples_leaf=40, l2_regularization=1.0,
                    random_state=seed)),
            ]),
            True,  # aplica pesos balanceados via sample_weight
        ),
    }


def top_k_acc(proba, classes, y_true, k=3):
    topk = np.argsort(proba, axis=1)[:, -k:]
    topk_labels = classes[topk]
    return np.mean([yt in row for yt, row in zip(y_true, topk_labels)])


def ajustar(modelo, usa_sw, Xtr, ytr):
    if usa_sw:
        sw = compute_sample_weight("balanced", ytr)
        modelo.fit(Xtr, ytr, clf__sample_weight=sw)
    else:
        modelo.fit(Xtr, ytr)
    return modelo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sep", default=";")
    ap.add_argument("--target", default="source_type")
    ap.add_argument("--group", default="frame_id")
    ap.add_argument("--drop", nargs="*", default=["image_path"],
                    help="colunas a ignorar (caminhos, ids, etc.)")
    ap.add_argument("--features", choices=["derivadas", "brutas", "ambas"],
                    default="derivadas",
                    help="derivadas = features_faciais.extract(); brutas = colunas numericas do CSV")
    ap.add_argument("--modulo", default="features_faciais",
                    help="nome do modulo com a funcao extract(row)")
    ap.add_argument("--sem-olhos", action="store_true",
                    help="remove features que dependem de olhos/sobrancelhas")
    ap.add_argument("--excluir", nargs="*", default=[],
                    help="classes a remover antes de treinar (ex.: --excluir coracao)")
    ap.add_argument("--filtro-caminho", default=None,
                    help="mantem so linhas cujo relative_path contem este texto (ex.: /review/)")
    ap.add_argument("--max-yaw", type=float, default=None,
                    help="descarta rostos com |pose_yaw| maior que este valor (ex.: 20)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y_raw, groups = carregar(args)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(cv.split(X, y, groups))

    modelos = montar_modelos(args.seed)
    linhas = []
    preds_oof = {}

    for nome, (modelo, usa_sw) in modelos.items():
        accs_tr, accs_te, bal, f1m, top3 = [], [], [], [], []
        oof = np.full(len(y), -1)
        for tr, te in folds:
            m = ajustar(modelo, usa_sw, X.iloc[tr], y[tr])
            p_tr, p_te = m.predict(X.iloc[tr]), m.predict(X.iloc[te])
            accs_tr.append(accuracy_score(y[tr], p_tr))
            accs_te.append(accuracy_score(y[te], p_te))
            bal.append(balanced_accuracy_score(y[te], p_te))
            f1m.append(f1_score(y[te], p_te, average="macro"))
            if nome.startswith("Baseline"):
                top3.append(np.nan)  # proba dos Dummy e "one-hot": top3 nao faz sentido
            else:
                top3.append(top_k_acc(m.predict_proba(X.iloc[te]), m.classes_, y[te]))
            oof[te] = p_te
        preds_oof[nome] = oof
        linhas.append({
            "modelo": nome,
            "acc_treino": np.mean(accs_tr),
            "acc_teste": np.mean(accs_te),
            "±": np.std(accs_te),
            "acc_balanceada": np.mean(bal),
            "f1_macro": np.mean(f1m),
            "top3": np.nan if nome.startswith("Baseline") else np.mean(top3),
        })

    tab = pd.DataFrame(linhas).set_index("modelo")
    print("=" * 78)
    print(f"RESULTADOS (media de {args.folds} folds agrupados por '{args.group}')")
    print("=" * 78)
    print(tab.round(3).to_string())

    prop = np.bincount(y) / len(y)
    print("\nReferencias do acaso:")
    print(f"  sorteio uniforme ({n_classes} classes): acc = {1/n_classes:.3f} | top3 = {3/n_classes:.3f}")
    print(f"  sorteio pela proporcao das classes:    acc = {np.sum(prop**2):.3f}")
    print(f"  sempre a classe majoritaria:           acc = {prop.max():.3f}")
    print(f"  top3 com ranking fixo das 3 classes mais frequentes: {np.sort(prop)[::-1][:3].sum():.3f}")

    # Detalhe do melhor modelo (por F1 macro, ignorando baselines)
    candidatos = tab[~tab.index.str.startswith("Baseline")]
    melhor = candidatos["f1_macro"].idxmax()
    oof = preds_oof[melhor]
    print(f"\nMelhor por F1 macro: {melhor}")
    print("\nMatriz de confusao (linhas = real, colunas = previsto):")
    print(list(le.classes_))
    print(confusion_matrix(y, oof))
    print("\nRelatorio por classe (predicoes fora-da-amostra):")
    print(classification_report(y, oof, target_names=le.classes_, digits=3))

    print("Como ler:")
    print("- acc_treino muito acima de acc_teste -> overfitting; aumente a regularizacao.")
    print("- acc_treino tambem baixa e todos proximos do acaso -> sem sinal nas features.")
    print("- compare top3 com 3/n_classes: so ha sinal se estiver claramente acima.")


if __name__ == "__main__":
    main()