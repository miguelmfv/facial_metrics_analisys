"""Fixtures compartilhadas pelos testes.

Decisao deliberada: as fixtures de dado usam uma linha **real** de
``arquivos_csv/faces-teste-supervisionado-formato-rosto.csv``, nunca um rosto
sintetico. Dado sintetico esconde exatamente os problemas que importam, e isso
ja aconteceu neste projeto: um payload de exemplo do app tinha, como contorno de
rosto, uma **elipse gerada**, nao um rosto, e passou por validacao de forma
porque a forma estava certa.

Nenhum caminho absoluto: tudo e derivado de ``__file__``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from vizo_rosto import carregar
from vizo_rosto.entrada import CONTORNOS_EXIGIDOS
from vizo_rosto.modelo import ModeloFormatoRosto

RAIZ = Path(__file__).resolve().parent.parent
CSV_TESTE = RAIZ / "arquivos_csv" / "faces-teste-supervisionado-formato-rosto.csv"
JOBLIB = RAIZ / "modelos" / "modelo_formato_rosto_sem_linhas_branco.joblib"


@pytest.fixture(scope="session")
def raiz() -> Path:
    """Raiz do repositorio."""
    return RAIZ


@pytest.fixture(scope="session")
def df_teste() -> pd.DataFrame:
    """O CSV de teste supervisionado, 78 rostos reais."""
    if not CSV_TESTE.is_file():
        pytest.skip(f"CSV de teste nao encontrado em {CSV_TESTE}")
    return pd.read_csv(CSV_TESTE, sep=";", index_col=False)


@pytest.fixture(scope="session")
def modelo() -> ModeloFormatoRosto:
    """O modelo versionado, carregado uma vez por sessao.

    Escopo de sessao porque sao 500 arvores: recarregar por teste seria
    desperdicio sem ganho de isolamento, dado que o modelo e imutavel.
    """
    if not JOBLIB.is_file():
        pytest.skip(f"Modelo nao encontrado em {JOBLIB}")
    return carregar(JOBLIB)


@pytest.fixture
def linha_csv(df_teste: pd.DataFrame) -> pd.Series:
    """Uma linha real do CSV, como pandas.Series."""
    return df_teste.iloc[0]


@pytest.fixture
def medicao_json(linha_csv: pd.Series) -> dict[str, Any]:
    """A MESMA linha real, remontada no formato JSON do backend.

    Reconstroi ``{"contours": {"face": [[x, y], ...]}}`` a partir das colunas
    ``ct_*``, o que permite provar que os dois caminhos de entrada convergem
    para o mesmo resultado usando o mesmo rosto de verdade.
    """
    contornos: dict[str, list[list[float]]] = {}
    for nome, n_pontos in CONTORNOS_EXIGIDOS.items():
        contornos[nome] = [
            [float(linha_csv[f"ct_{nome}_x{i}"]), float(linha_csv[f"ct_{nome}_y{i}"])]
            for i in range(n_pontos)
        ]
    # Campos que o backend manda junto e que o pacote deve ignorar, inclusive
    # um que parece rotulo: se algum caminho do codigo passar a ler rotulo, este
    # valor errado de proposito faz o teste falhar.
    return {
        "contours": contornos,
        "height_to_width_ratio": 1.23,
        "yaw": 0.5,
        "smiling_probability": 0.1,
        "source_type": "ESTE_ROTULO_NAO_DEVE_SER_LIDO",
    }
