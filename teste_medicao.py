"""Roda uma medicao avulsa pelo pacote vizo_rosto e imprime a saida inteira.

Uso, da raiz do repositorio:

    $env:PYTHONPATH="src"
    .venv\\Scripts\\python teste_medicao.py medicao-teste.json
    .venv\\Scripts\\python teste_medicao.py medicao-teste.csv

Aceita .json (medicao como o backend guarda) ou .csv (linha achatada, sep=';').
Nao escreve nada.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vizo_rosto import carregar, recomendar

MODELO = "modelos/modelo_formato_rosto_sem_linhas_branco.joblib"


def carregar_medicao(caminho: Path):
    if caminho.suffix.lower() == ".json":
        with caminho.open(encoding="utf-8") as f:
            return json.load(f)
    if caminho.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(caminho, sep=";")
        if df.shape[1] == 1:  # separador errado: tudo grudado em uma coluna
            df = pd.read_csv(caminho, sep=",")
        if len(df) != 1:
            print(f"aviso: o CSV tem {len(df)} linhas; usando a linha 0.")
        return df.iloc[0]
    raise SystemExit(f"Extensao nao reconhecida: '{caminho.suffix}'. Use .json ou .csv.")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("uso: python teste_medicao.py <medicao.json|medicao.csv>")

    caminho = Path(sys.argv[1])
    if not caminho.is_file():
        raise SystemExit(f"Nao encontrei '{caminho}'. Rode da raiz do repositorio.")

    modelo = carregar(MODELO)
    medicao = carregar_medicao(caminho)
    rec = recomendar(medicao, modelo)

    print(f"entrada: {caminho}  ({'JSON de medicao' if caminho.suffix == '.json' else 'linha de CSV'})")
    print()
    print("DESCRITIVO (modelo aprendido)")
    print(f"  formato de rosto : {rec.formato_rosto}")
    for nome, p in rec.previsao.probabilidades.items():
        barra = "#" * round(p * 40)
        print(f"    {nome:9s} {p:.4f}  {barra}")
    print(f"  confianca        : {rec.previsao.confianca:.3f}")
    print(f"  margem 1a-2a     : {rec.previsao.margem:.3f}")
    print(f"  top-2            : {rec.previsao.top(2)}")
    print()
    print("PRESCRITIVO (regra de visagismo declarada)")
    print(f"  principio: {rec.principio}")
    for a in rec.armacoes:
        print(f"  {a.posicao}. {a.formato:11s} [{a.sustentacao}]  fontes: {', '.join(f.id for f in a.fontes)}")
        print(f"     {a.justificativa}")
        if a.observacao:
            print(f"     obs: {a.observacao}")
    print()
    print("AVISO")
    for linha in rec.aviso.strip().splitlines():
        print(f"  {linha}")


if __name__ == "__main__":
    main()
