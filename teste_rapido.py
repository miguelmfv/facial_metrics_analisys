"""Teste rapido do caminho entrada -> saida do pacote vizo_rosto.

Rodar da raiz do repositorio facial_metrics_analisys:

    # PowerShell
    $env:PYTHONPATH="src"; .venv\\Scripts\\python teste_rapido.py

    # bash / Git Bash
    PYTHONPATH=src .venv/Scripts/python teste_rapido.py

Nao escreve nada, nao treina nada, nao toca no git.
"""

from __future__ import annotations

import json

import pandas as pd

from vizo_rosto import (
    CONTORNOS_EXIGIDOS,
    EntradaInvalidaError,
    FeatureNaoFinitaError,
    ModeloIncompativelError,
    carregar,
    recomendar,
)

MODELO = "modelos/modelo_formato_rosto_sem_linhas_branco.joblib"
CSV = "arquivos_csv/faces-teste-supervisionado-formato-rosto.csv"


def json_do_backend(linha: pd.Series) -> dict:
    """Remonta a linha achatada do CSV no formato que o backend guarda."""
    return {
        "contours": {
            nome: [
                [float(linha[f"ct_{nome}_x{i}"]), float(linha[f"ct_{nome}_y{i}"])]
                for i in range(n)
            ]
            for nome, n in CONTORNOS_EXIGIDOS.items()
        }
    }


def main() -> None:
    modelo = carregar(MODELO)
    print(f"1) MODELO carregado — classes: {modelo.classes}")
    print(f"   features no contrato: {len(modelo.features)}")

    df = pd.read_csv(CSV, sep=";")
    linha = df.iloc[0]
    print(f"\n2) CSV: {len(df)} linhas · usando a linha 0 ({linha['file_name']})")

    # --- caminho A: linha achatada de CSV ---
    rec_csv = recomendar(linha, modelo)

    # --- caminho B: JSON de medicao, como o backend guarda ---
    medicao = json_do_backend(linha)
    rec_json = recomendar(medicao, modelo)

    print("\n3) SAIDA pelo caminho CSV")
    print(f"   formato de rosto : {rec_csv.formato_rosto}")
    print(f"   probabilidades   : {rec_csv.previsao.probabilidades}")
    print(f"   confianca/margem : {rec_csv.previsao.confianca:.3f} / {rec_csv.previsao.margem:.3f}")
    print(f"   armacoes         : {[a.formato for a in rec_csv.armacoes]}")
    print(f"   sustentacao 1a   : {rec_csv.armacoes[0].sustentacao}")
    print(f"   fonte 1a         : {[f.id for f in rec_csv.armacoes[0].fontes]}")

    iguais = (
        rec_csv.formato_rosto == rec_json.formato_rosto
        and rec_csv.previsao.probabilidades == rec_json.previsao.probabilidades
    )
    print(f"\n4) CSV vs JSON do backend — mesma saida: {iguais}")
    print(f"   rotulo real da linha (NAO usado pelo codigo): {linha['source_type']}")

    # --- as recusas: e aqui que a estrutura nova se prova ---
    print("\n5) RECUSAS")

    faltando = json.loads(json.dumps(medicao))
    del faltando["contours"]["noseBottom"]
    try:
        recomendar(faltando, modelo)
        print("   contorno faltando  : NAO RECUSOU  <-- defeito")
    except EntradaInvalidaError as erro:
        print(f"   contorno faltando  : recusou — {str(erro)[:90]}...")

    curto = json.loads(json.dumps(medicao))
    curto["contours"]["face"] = curto["contours"]["face"][:30]
    try:
        recomendar(curto, modelo)
        print("   36 -> 30 pontos    : NAO RECUSOU  <-- defeito")
    except EntradaInvalidaError as erro:
        print(f"   36 -> 30 pontos    : recusou — {str(erro)[:90]}...")

    nao_finito = json.loads(json.dumps(medicao))
    nao_finito["contours"]["face"][0] = [float("inf"), 10.0]
    try:
        recomendar(nao_finito, modelo)
        print("   coordenada inf     : NAO RECUSOU  <-- defeito")
    except (EntradaInvalidaError, FeatureNaoFinitaError) as erro:
        print(f"   coordenada inf     : recusou — {str(erro)[:90]}...")

    try:
        carregar(MODELO, classes_esperadas=["aviador", "coracao", "gatinho"])
        print("   modelo trocado     : NAO RECUSOU  <-- defeito")
    except ModeloIncompativelError as erro:
        print(f"   modelo trocado     : recusou — {str(erro)[:90]}...")

    print("\n6) para_dicionario() — primeiras chaves do JSON de saida")
    d = rec_csv.para_dicionario()
    print(f"   {list(d)}")
    print(f"   formato_rosto: {list(d['formato_rosto'])}")
    print(f"   ranking[0]   : {d['formatos_armacao']['ranking'][0]['formato']}")


if __name__ == "__main__":
    main()
