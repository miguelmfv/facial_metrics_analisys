"""Teste ponta a ponta da orquestracao, com entrada real e saida conferida."""

from __future__ import annotations

import copy
import json
from typing import Any

import pandas as pd
import pytest

from vizo_rosto import recomendar
from vizo_rosto.entrada import EntradaInvalidaError
from vizo_rosto.modelo import FeatureNaoFinitaError, ModeloFormatoRosto
from vizo_rosto.recomendacao import calcular_features
from vizo_rosto.visagismo import FORMATOS_ARMACAO, FORMATOS_ROSTO

# Valores de referencia, medidos na PRIMEIRA linha de
# arquivos_csv/faces-teste-supervisionado-formato-rosto.csv
# (arquivo 06c919bde2344cd1eadcf61658455438.jpg) com o artefato versionado.
# Conferem, digito a digito, com a linha-base colhida ANTES do refactor, em
# contexto/desenvolvimento/dados/linha-base-T2-resultados_previsao-rodada-3-tarefa-1.csv
#
# Se este teste quebrar, uma de duas coisas aconteceu: o features_faciais.py
# mudou de logica (proibido por D-50) ou o modelo foi substituido (proibido por
# D-48). Nos dois casos e defeito, nao melhoria.
PRIMEIRA_LINHA_ESPERADA = {
    "formato": "diamante",
    "probabilidades": {
        "diamante": 0.508,
        "oval": 0.313,
        "quadrado": 0.108,
        "redondo": 0.071,
    },
}


def test_ponta_a_ponta_com_linha_real(modelo: ModeloFormatoRosto, linha_csv: pd.Series) -> None:
    """O caso que fecha a tarefa: entrada conhecida, saida conferida."""
    rec = recomendar(linha_csv, modelo)

    assert rec.formato_rosto == PRIMEIRA_LINHA_ESPERADA["formato"]
    for classe, esperado in PRIMEIRA_LINHA_ESPERADA["probabilidades"].items():
        assert rec.previsao.probabilidades[classe] == pytest.approx(esperado, abs=5e-4), (
            f"a probabilidade de '{classe}' mudou em relacao a linha-base"
        )

    assert rec.armacoes, "a recomendacao tem de trazer pelo menos um formato de armacao"
    assert [a.posicao for a in rec.armacoes] == list(range(1, len(rec.armacoes) + 1))
    assert all(a.formato in FORMATOS_ARMACAO for a in rec.armacoes)
    assert all(a.fontes for a in rec.armacoes), "toda posicao do ranking tem de ter fonte"
    assert rec.aviso and rec.principio
    assert len(rec.features) == 29


def test_os_dois_caminhos_de_entrada_dao_a_mesma_recomendacao(
    modelo: ModeloFormatoRosto, linha_csv: pd.Series, medicao_json: dict[str, Any]
) -> None:
    """JSON do backend e linha de CSV tem de convergir, no mesmo rosto real.

    Esta e a prova de que o adaptador de D-46 nao distorce nada: se o backend e
    o CSV divergissem, o numero da monografia e o numero do app seriam
    diferentes.
    """
    do_csv = recomendar(linha_csv, modelo)
    do_json = recomendar(medicao_json, modelo)

    assert do_json.previsao == do_csv.previsao
    assert do_json.features == do_csv.features
    assert [a.formato for a in do_json.armacoes] == [a.formato for a in do_csv.armacoes]


def test_recomendacao_e_deterministica(
    modelo: ModeloFormatoRosto, linha_csv: pd.Series
) -> None:
    primeira = recomendar(linha_csv, modelo)
    assert recomendar(linha_csv, modelo).previsao == primeira.previsao


def test_nao_exige_rotulo_na_entrada(
    modelo: ModeloFormatoRosto, medicao_json: dict[str, Any]
) -> None:
    """A entrada vem de um usuario real: nao existe classe verdadeira para ela."""
    sem_rotulo = {k: v for k, v in medicao_json.items() if k != "source_type"}
    assert recomendar(sem_rotulo, modelo).previsao == recomendar(medicao_json, modelo).previsao


def test_rotulo_errado_na_entrada_nao_muda_a_saida(
    modelo: ModeloFormatoRosto, medicao_json: dict[str, Any]
) -> None:
    """Se algum caminho passasse a ler o rotulo, este teste pegaria."""
    mentiroso = copy.deepcopy(medicao_json)
    mentiroso["source_type"] = "quadrado"
    assert recomendar(mentiroso, modelo).previsao == recomendar(medicao_json, modelo).previsao


def test_funciona_para_as_78_linhas_do_csv_de_teste(
    modelo: ModeloFormatoRosto, df_teste: pd.DataFrame
) -> None:
    """Nenhuma linha real do CSV de teste pode falhar nem sair fora do contrato."""
    for i in range(len(df_teste)):
        rec = recomendar(df_teste.iloc[i], modelo)
        assert rec.formato_rosto in FORMATOS_ROSTO
        assert sum(rec.previsao.probabilidades.values()) == pytest.approx(1.0, abs=1e-9)
        assert rec.armacoes


def test_medicao_com_contorno_quebrado_e_recusada(
    modelo: ModeloFormatoRosto, medicao_json: dict[str, Any]
) -> None:
    quebrada = copy.deepcopy(medicao_json)
    quebrada["contours"]["face"] = quebrada["contours"]["face"][:20]
    with pytest.raises(EntradaInvalidaError):
        recomendar(quebrada, modelo)


def test_medicao_que_gera_feature_nao_finita_e_recusada(
    modelo: ModeloFormatoRosto, medicao_json: dict[str, Any]
) -> None:
    """D-56: colapsar o contorno do rosto em uma linha torna a largura nao medivel.

    Confirma que a recusa acontece de verdade no caminho completo, e nao apenas
    quando se injeta NaN a mao na feature.
    """
    quebrada = copy.deepcopy(medicao_json)
    y = quebrada["contours"]["face"][0][1]
    quebrada["contours"]["face"] = [[p[0], y] for p in quebrada["contours"]["face"]]
    with pytest.raises((FeatureNaoFinitaError, EntradaInvalidaError)):
        recomendar(quebrada, modelo)


def test_formato_de_entrada_irreconhecivel_explica_as_opcoes(
    modelo: ModeloFormatoRosto
) -> None:
    with pytest.raises(EntradaInvalidaError) as erro:
        recomendar({"altura": 1.7, "peso": 70}, modelo)
    assert "entrada.py" in str(erro.value)


def test_para_dicionario_e_serializavel_e_separa_as_naturezas(
    modelo: ModeloFormatoRosto, linha_csv: pd.Series
) -> None:
    """A saida para o backend tem de dizer o que e descritivo e o que e prescritivo."""
    d = recomendar(linha_csv, modelo).para_dicionario()
    texto = json.dumps(d, ensure_ascii=False)  # nao levanta => serializavel
    assert "descritivo" in d["formato_rosto"]["natureza"]
    assert "prescritivo" in d["formatos_armacao"]["natureza"]
    assert len(d["formato_rosto"]["probabilidades"]) == 4
    assert d["formatos_armacao"]["ranking"][0]["posicao"] == 1
    assert d["formatos_armacao"]["ranking"][0]["fontes"], "a fonte tem de chegar ao consumidor"
    assert d["aviso"]
    assert "features" not in d, "as features nao vao por default"
    assert len(texto) > 0


def test_para_dicionario_inclui_features_so_quando_pedido(
    modelo: ModeloFormatoRosto, linha_csv: pd.Series
) -> None:
    d = recomendar(linha_csv, modelo).para_dicionario(incluir_features=True)
    assert len(d["features"]) == 29


def test_calcular_features_devolve_as_29_na_ordem_do_modelo(
    modelo: ModeloFormatoRosto, linha_csv: pd.Series
) -> None:
    features = calcular_features(linha_csv)
    assert list(features) == list(modelo.features)
    assert all(isinstance(v, float) for v in features.values())
