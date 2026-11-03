"""
Pipeline rosto -> armacao.

Etapas:
 1. Le o CSV das imagens de review e extrai as features (features_faciais.extract).
 2. Carrega o modelo de FORMATO DE ROSTO (joblib) e calcula, para cada imagem,
    a probabilidade de cada formato de rosto (+ top1 e top2). Salva um CSV novo.
 3. ANALISE DA RELACAO: tabela "formato de armacao x probabilidade media de cada
    formato de rosto" + teste de permutacao (embaralhando o rotulo ENTRE os
    modelos de armacao, nao entre as imagens).
 4. COMPARACAO: o modelo de armacao melhora ao receber as probabilidades de rosto?
    (so features | features + prob. rosto | so prob. rosto), com a mesma
    validacao agrupada por model_id.

Uso (na mesma pasta de features_faciais.py e comparar_modelos.py):
    python pipeline_rosto_armacao.py --csv extracao.csv --modelo-rosto modelo_formato_rosto.joblib \
        --filtro-caminho /review/ --max-yaw 20 --excluir coracao
    (target padrao = source_type = pasta raiz = formato da armacao;
     group padrao = frame_id = modelo da armacao; drop padrao = image_path)
"""
import argparse
import importlib
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path[:0] = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]

from comparar_modelos import (  # noqa: E402
    DEPENDEM_OLHOS_SOBRANCELHAS,
    ajustar,
    carregar,
    montar_modelos,
    top_k_acc,
)

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# 2) Probabilidades do modelo de formato de rosto
# --------------------------------------------------------------------------
def carregar_modelo_rosto(caminho):
    """Aceita o modelo direto ou um dict (qualquer nome de chave).

    No dict, procura sozinho: o objeto com predict_proba (modelo), um LabelEncoder
    ou lista de nomes de classes, e a lista de nomes das features.
    """
    obj = joblib.load(caminho)
    colunas, nomes_classes = None, None

    if isinstance(obj, dict):
        print(f"O joblib e um dict com as chaves: {list(obj.keys())}")
        modelo = next((v for v in obj.values() if hasattr(v, "predict_proba")), None)

        if modelo is None:
            raise SystemExit("Nenhum valor do dict tem predict_proba. Chaves/tipos: "
                             + str({k: type(v).__name__ for k, v in obj.items()}))

        for k, v in obj.items():
            if v is modelo:
                continue

            kl = str(k).lower()

            if hasattr(v, "inverse_transform") and hasattr(v, "classes_"):  # LabelEncoder
                nomes_classes = [str(c) for c in v.classes_]

            elif isinstance(v, (list, tuple, np.ndarray, pd.Index)) and len(v) > 0 \
                    and all(isinstance(i, str) for i in v):

                if any(w in kl for w in ("class", "label", "target")):
                    nomes_classes = [str(i) for i in v]

                elif any(w in kl for w in ("feat", "col", "nome", "name")):
                    colunas = [str(i) for i in v]

    else:
        modelo = obj

    if not hasattr(modelo, "predict_proba"):
        raise SystemExit("O modelo de rosto precisa ter predict_proba.")

    return modelo, colunas, nomes_classes


def probs_rosto(modelo, colunas_salvas, nomes_classes, X_full):
    nomes = colunas_salvas if colunas_salvas is not None else getattr(modelo, "feature_names_in_", None)

    if nomes is not None:
        faltam = [c for c in nomes if c not in X_full.columns]

        if faltam:
            raise SystemExit(f"O modelo de rosto espera features que o extract() nao gerou: {faltam}")

        Xin = X_full[list(nomes)]

    else:
        n = getattr(modelo, "n_features_in_", X_full.shape[1])

        if n != X_full.shape[1]:
            raise SystemExit(f"Modelo espera {n} features, extract() gerou {X_full.shape[1]}.")

        Xin = X_full  # assume a mesma ordem usada no treino

    Xin = Xin.fillna(Xin.median())
    proba = modelo.predict_proba(Xin)
    cls = list(modelo.classes_)

    if nomes_classes is not None and all(isinstance(c, (int, np.integer)) for c in cls):
        classes = [nomes_classes[int(c)] for c in cls]  # classes codificadas -> nomes

    else:
        classes = [str(c) for c in cls]

        if all(c.lstrip("-").isdigit() for c in classes):
            print("AVISO: as classes do modelo sao numeros e nao ha LabelEncoder/lista de nomes no joblib.")

    P = pd.DataFrame(proba, columns=[f"p_rosto_{c}" for c in classes], index=X_full.index)
    ordem = np.argsort(-proba, axis=1)
    top = pd.DataFrame({
        "rosto_top1": [classes[i] for i in ordem[:, 0]],
        "rosto_top2": [classes[i] for i in ordem[:, 1]],
    }, index=X_full.index)

    return P, top, classes


def checar_modelo_rosto(classes_rosto, classes_armacao, caminho):
    """Aborta se o joblib parece ser um modelo de ARMACAO (e nao de rosto)."""
    so_armacao = {"aviador", "gatinho", "retangular", "coracao", "coração"}
    achadas = so_armacao & {c.lower() for c in classes_rosto}
    comuns = {c.lower() for c in classes_rosto} & {c.lower() for c in classes_armacao}

    if achadas or len(comuns) >= 5:
        raise SystemExit(
            f"\nERRO: o modelo '{caminho}' tem classes {classes_rosto}, que parecem formatos de "
            f"ARMACAO, nao de rosto.\nProvavelmente ele foi sobrescrito pelo treino do modelo de "
            f"armacao ou o caminho esta errado.\nUse o joblib treinado no dataset de formato de "
            f"rosto (ex.: oval, redondo, quadrado, diamante). Para ignorar esta checagem: "
            f"--ignorar-checagem."
        )


# --------------------------------------------------------------------------
# 3) Relacao armacao x rosto
# --------------------------------------------------------------------------
def estatistica(M, codigos, n_cls):
    """soma_c n_c * (media_c - media_geral)^2, por coluna de probabilidade."""
    geral = M.mean(0)
    s = np.zeros(M.shape[1])

    for c in range(n_cls):
        sel = codigos == c
        if sel.any():
            s += sel.sum() * (M[sel].mean(0) - geral) ** 2

    return s


def analise_relacao(P, y, grupos, top, n_perm, seed):
    cols = list(P.columns)
    d = P.copy()
    d["classe"] = y.values
    d["grupo"] = grupos.values
    multi = d.groupby("grupo")["classe"].nunique()

    if (multi > 1).any():
        print(f"AVISO: {int((multi > 1).sum())} modelos aparecem em mais de uma classe de armacao.")

    # uma linha por modelo de armacao: evita que modelos com muitas fotos dominem
    pm = d.groupby("grupo").agg({**{c: "mean" for c in cols}, "classe": "first"})
    classes = sorted(pm["classe"].unique())
    cod = pm["classe"].map({c: i for i, c in enumerate(classes)}).values
    M = pm[cols].values

    tab = pm.groupby("classe")[cols].mean()
    tab.insert(0, "n_modelos", pm.groupby("classe").size())
    print("\n" + "=" * 78)
    print("RELACAO ARMACAO x ROSTO")
    print("=" * 78)
    print("\nProbabilidade media de cada formato de rosto, por formato de armacao")
    print("(uma media por modelo de armacao, depois media entre os modelos da classe):")
    print(tab.round(3).to_string())
    print("\nSe todas as linhas forem parecidas, nao ha relacao.")
    print("Amplitude (max - min entre classes) por formato de rosto:")
    print((tab[cols].max() - tab[cols].min()).round(3).to_string())

    # formato de rosto previsto (top1) por classe de armacao, em % das imagens
    cruz = pd.crosstab(y.values, top["rosto_top1"].values, normalize="index") * 100
    print("\n% das imagens por formato de rosto previsto (top1), por formato de armacao:")
    print(cruz.round(1).to_string())

    rng = np.random.default_rng(seed)
    obs = estatistica(M, cod, len(classes))
    cont = np.zeros_like(obs)
    cont_g = 0

    for _ in range(n_perm):
        perm = rng.permutation(cod)
        s = estatistica(M, perm, len(classes))
        cont += s >= obs
        cont_g += s.sum() >= obs.sum()

    p_col = (1 + cont) / (1 + n_perm)
    p_glob = (1 + cont_g) / (1 + n_perm)
    print(f"\nTeste de permutacao ({n_perm} embaralhamentos do rotulo entre {len(pm)} modelos de armacao):")
    print(f"  p global (algum formato de rosto difere entre armacoes): {p_glob:.4f}")
    for c, p in zip(cols, p_col):
        print(f"  p {c}: {p:.4f}")

    print(f"  (com {len(cols)} testes por coluna, use p < {0.05/len(cols):.4f} como corte aproximado)")
    return tab


# --------------------------------------------------------------------------
# 4) Comparacao com e sem probabilidades de rosto
# --------------------------------------------------------------------------
def comparar(conjuntos, y, grupos, seed, folds_n):
    le = LabelEncoder()
    yi = le.fit_transform(y)
    cv = StratifiedGroupKFold(n_splits=folds_n, shuffle=True, random_state=seed)
    ref = next(iter(conjuntos.values()))
    folds = list(cv.split(ref, yi, grupos))
    todos = montar_modelos(seed)
    escolhidos = ["Regressao logistica", "Random Forest (mais regularizada)", "HistGradientBoosting"]

    linhas = []

    def rodar(nome_conj, X, nome_modelo, modelo, usa_sw):
        acc, bal, f1m, top3 = [], [], [], []

        for tr, te in folds:
            m = ajustar(modelo, usa_sw, X.iloc[tr], yi[tr])
            p = m.predict(X.iloc[te])
            acc.append((p == yi[te]).mean())
            bal.append(balanced_accuracy_score(yi[te], p))
            f1m.append(f1_score(yi[te], p, average="macro"))

            if hasattr(m, "predict_proba") and not nome_modelo.startswith("Baseline"):
                top3.append(top_k_acc(m.predict_proba(X.iloc[te]), m.classes_, yi[te]))

        linhas.append({
            "conjunto": nome_conj, "modelo": nome_modelo,
            "acc_teste": np.mean(acc), "acc_balanc.": np.mean(bal),
            "bal±": np.std(bal), "f1_macro": np.mean(f1m),
            "top3": np.mean(top3) if top3 else np.nan,
        })

    for nome in ("Baseline: classe majoritaria", "Baseline: sorteio por proporcao"):
        m, sw = todos[nome]
        rodar("-", ref, nome, m, sw)

    for nome_conj, X in conjuntos.items():
        for nome in escolhidos:
            m, sw = todos[nome]
            rodar(nome_conj, X, nome, m, sw)

    tab = pd.DataFrame(linhas).set_index(["conjunto", "modelo"])
    print("\n" + "=" * 78)
    print(f"COMPARACAO (media de {folds_n} folds agrupados por modelo de armacao)")
    print("=" * 78)
    print(tab.round(3).to_string())

    prop = np.sort(np.bincount(yi) / len(yi))[::-1]
    k = len(prop)
    print("\nReferencias do acaso:")
    print(f"  acuracia: uniforme = {1/k:.3f} | por proporcao = {np.sum(prop**2):.3f} | majoritaria = {prop[0]:.3f}")
    print(f"  acuracia balanceada do acaso = {1/k:.3f}")
    print(f"  top3: uniforme = {3/k:.3f} | ranking fixo das 3 classes mais frequentes = {prop[:3].sum():.3f}")
    print("\nComo ler: compare 'features + prob. rosto' com 'so features' usando acc_balanc.")
    print("e f1_macro. Diferencas menores que o desvio (bal±) nao sao evidencia de ganho.")
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sep", default=";")
    ap.add_argument("--target", default="source_type")
    ap.add_argument("--group", default="frame_id")
    ap.add_argument("--drop", nargs="*", default=["image_path"])
    ap.add_argument("--modelo-rosto", required=True, help="arquivo .joblib do modelo de formato de rosto")
    ap.add_argument("--modulo", default="features_faciais")
    ap.add_argument("--sem-olhos", action="store_true",
                    help="o modelo de ARMACAO nao usa features de olhos/sobrancelhas "
                         "(o modelo de rosto continua recebendo as features completas)")
    ap.add_argument("--ignorar-checagem-classes", action="store_true")
    ap.add_argument("--excluir", nargs="*", default=[])
    ap.add_argument("--filtro-caminho", default=None,
                    help="mantem so linhas cujo relative_path contem este texto (ex.: /review/)")
    ap.add_argument("--max-yaw", type=float, default=None,
                    help="descarta rostos com |pose_yaw| maior que este valor (ex.: 20)")
    ap.add_argument("--permutacoes", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida-csv", default="review_com_prob_rosto.csv")
    ap.add_argument("--ignorar-checagem", action="store_true",
                    help="nao abortar se o modelo de rosto parecer um modelo de armacao")
    args = ap.parse_args()

    # features completas (o modelo de rosto foi treinado com todas)
    sem_olhos, args.sem_olhos = args.sem_olhos, False
    args.features = "derivadas"
    X_full, y, grupos = carregar(args)
    args.sem_olhos = sem_olhos

    n_cls = y.nunique()

    if n_cls < 2:
        raise SystemExit(
            f"\nERRO: '{args.target}' tem apenas {n_cls} classe(s): {sorted(y.unique())}.\n"
            f"Esta coluna e a pasta raiz do arquivo. Organize os modelos de armacao em pastas por "
            f"formato (aviador, gatinho...) e gere o CSV de novo antes de rodar o pipeline."
        )

    modelos_por_classe = grupos.groupby(y).nunique()

    if (modelos_por_classe < args.folds).any():
        print(f"AVISO: classes com menos modelos de armacao que folds ({args.folds}):")
        print(modelos_por_classe[modelos_por_classe < args.folds].to_string())

    modelo, colunas, nomes_classes = carregar_modelo_rosto(args.modelo_rosto)
    P, top, classes_rosto = probs_rosto(modelo, colunas, nomes_classes, X_full)
    print(f"Modelo de rosto: classes = {classes_rosto}")

    if not args.ignorar_checagem:
        checar_modelo_rosto(classes_rosto, sorted(y.unique()), args.modelo_rosto)

    # trava de seguranca: as classes do modelo "de rosto" nao podem ser as da armacao
    if set(classes_rosto) <= set(y.unique()) and not args.ignorar_checagem_classes:
        raise SystemExit(
            "\nPARADO: todas as classes do modelo carregado "
            f"({classes_rosto}) tambem sao formatos de ARMACAO do CSV.\n"
            "Esse joblib parece ser um modelo de armacao, nao de formato de rosto.\n"
            "Confira o arquivo passado em --modelo-rosto. Se os nomes coincidem por serem\n"
            "mesmo formatos de rosto (ex.: quadrado, redondo), rode com --ignorar-checagem-classes.")

    print("Distribuicao do rosto previsto (top1) nas imagens de review:")
    print((top["rosto_top1"].value_counts(normalize=True) * 100).round(1).to_string())
    print("(se quase tudo cair em uma classe, o modelo de rosto esta perdido nessas fotos)")

    df_ref = getattr(args, "_df", None)
    ident = {}

    if df_ref is not None:
        for c in ("relative_path", "file_name", "pose_yaw"):
            if c in df_ref.columns:
                ident[c] = df_ref[c].reset_index(drop=True)

    saida = pd.concat([pd.DataFrame(ident), y.rename("frame_shape"), grupos.rename("model_id"),
                       X_full, P, top], axis=1)

    saida.to_csv(args.saida_csv, sep=";", index=False)
    print(f"\nCSV com probabilidades salvo em: {args.saida_csv}")

    analise_relacao(P, y, grupos, top, args.permutacoes, args.seed)

    X_arm = X_full.drop(columns=DEPENDEM_OLHOS_SOBRANCELHAS, errors="ignore") if sem_olhos else X_full
    conjuntos = {
        "so features": X_arm,
        "features + prob. rosto": pd.concat([X_arm, P], axis=1),
        "so prob. rosto": P,
    }
    comparar(conjuntos, y, grupos, args.seed, args.folds)


if __name__ == "__main__":
    main()