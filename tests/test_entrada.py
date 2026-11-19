"""Testes do adaptador de entrada (D-46)."""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from vizo_rosto.entrada import (
    CONTORNOS_EXIGIDOS,
    EntradaInvalidaError,
    achatar_medicao,
    linha_de_csv,
    nomes_de_coluna,
)


def test_nomes_de_coluna_tem_duas_por_ponto() -> None:
    nomes = nomes_de_coluna()
    esperado = 2 * sum(CONTORNOS_EXIGIDOS.values())
    assert sum(CONTORNOS_EXIGIDOS.values()) == 111
    assert len(nomes) == esperado == 222
    assert len(set(nomes)) == len(nomes), "ha nome de coluna repetido"
    assert nomes[0] == "ct_face_x0"
    assert nomes[1] == "ct_face_y0"


def test_json_valido_vira_linha_achatada(medicao_json: dict[str, Any]) -> None:
    linha = achatar_medicao(medicao_json)
    assert set(linha) == set(nomes_de_coluna())
    assert all(isinstance(v, float) for v in linha.values())
    assert all(math.isfinite(v) for v in linha.values())


def test_json_e_linha_de_csv_dao_o_mesmo_resultado(
    medicao_json: dict[str, Any], linha_csv: Any
) -> None:
    """Os dois caminhos de entrada tem de convergir, usando o mesmo rosto real."""
    do_json = achatar_medicao(medicao_json)
    do_csv = linha_de_csv(linha_csv)
    assert do_json == do_csv


def test_aceita_dicionario_de_contornos_direto(medicao_json: dict[str, Any]) -> None:
    """Sem a chave 'contours', mas com os contornos no nivel de cima."""
    assert achatar_medicao(medicao_json["contours"]) == achatar_medicao(medicao_json)


def test_rotulo_na_entrada_e_ignorado(medicao_json: dict[str, Any]) -> None:
    """A entrada vem sem rotulo; se vier um, nao pode influenciar nada."""
    sem_rotulo = {k: v for k, v in medicao_json.items() if k != "source_type"}
    assert achatar_medicao(sem_rotulo) == achatar_medicao(medicao_json)


def test_contorno_faltando_diz_qual(medicao_json: dict[str, Any]) -> None:
    quebrada = copy.deepcopy(medicao_json)
    del quebrada["contours"]["noseBottom"]
    with pytest.raises(EntradaInvalidaError) as erro:
        achatar_medicao(quebrada)
    msg = str(erro.value)
    assert "noseBottom" in msg, "a mensagem tem de nomear o contorno que falta"
    assert "nova captura" in msg, "a mensagem tem de dizer o que fazer"


def test_contagem_de_pontos_errada_diz_quantos_vieram(medicao_json: dict[str, Any]) -> None:
    quebrada = copy.deepcopy(medicao_json)
    quebrada["contours"]["face"] = quebrada["contours"]["face"][:30]
    with pytest.raises(EntradaInvalidaError) as erro:
        achatar_medicao(quebrada)
    msg = str(erro.value)
    assert "face" in msg
    assert "30" in msg, "a mensagem tem de dizer quantos pontos vieram"
    assert "36" in msg, "a mensagem tem de dizer quantos eram esperados"


def test_contorno_com_pontos_a_mais_tambem_e_recusado(medicao_json: dict[str, Any]) -> None:
    """Pontos sobrando sao tao perigosos quanto faltando: o extract indexa por posicao."""
    quebrada = copy.deepcopy(medicao_json)
    quebrada["contours"]["face"] = quebrada["contours"]["face"] + [[0.0, 0.0]]
    with pytest.raises(EntradaInvalidaError) as erro:
        achatar_medicao(quebrada)
    assert "37" in str(erro.value)


def test_coordenada_nao_numerica_e_recusada(medicao_json: dict[str, Any]) -> None:
    quebrada = copy.deepcopy(medicao_json)
    quebrada["contours"]["leftEye"][3] = ["abc", 10]
    with pytest.raises(EntradaInvalidaError) as erro:
        achatar_medicao(quebrada)
    msg = str(erro.value)
    assert "leftEye" in msg and "3" in msg


def test_coordenada_nan_e_recusada(medicao_json: dict[str, Any]) -> None:
    quebrada = copy.deepcopy(medicao_json)
    quebrada["contours"]["face"][10] = [float("nan"), 5.0]
    with pytest.raises(EntradaInvalidaError) as erro:
        achatar_medicao(quebrada)
    assert "finito" in str(erro.value)


def test_ponto_com_tres_coordenadas_e_recusado(medicao_json: dict[str, Any]) -> None:
    quebrada = copy.deepcopy(medicao_json)
    quebrada["contours"]["noseBottom"][0] = [1.0, 2.0, 3.0]
    with pytest.raises(EntradaInvalidaError) as erro:
        achatar_medicao(quebrada)
    assert "2" in str(erro.value)


def test_aceita_ponto_como_objeto_x_y(medicao_json: dict[str, Any]) -> None:
    """Alguns serializadores mandam {'x': .., 'y': ..} em vez de [x, y]."""
    variante = copy.deepcopy(medicao_json)
    variante["contours"]["noseBottom"] = [
        {"x": p[0], "y": p[1]} for p in medicao_json["contours"]["noseBottom"]
    ]
    assert achatar_medicao(variante) == achatar_medicao(medicao_json)


def test_medicao_sem_contornos_explica_o_que_mandar() -> None:
    with pytest.raises(EntradaInvalidaError) as erro:
        achatar_medicao({"yaw": 1.0, "pitch": 2.0})
    msg = str(erro.value)
    assert "contours" in msg


def test_medicao_que_nao_e_dicionario_orienta_o_json_loads() -> None:
    with pytest.raises(EntradaInvalidaError) as erro:
        achatar_medicao("{\"contours\": {}}")  # type: ignore[arg-type]
    assert "json.loads" in str(erro.value)


def test_linha_de_csv_sem_colunas_de_contorno_conta_as_ct() -> None:
    with pytest.raises(EntradaInvalidaError) as erro:
        linha_de_csv({"file_name": "a.jpg", "ct_face_x0": 1.0})
    msg = str(erro.value)
    assert "ct_" in msg
    assert "sep=';'" in msg, "a mensagem tem de sugerir a causa mais provavel"


def test_linha_de_csv_ignora_colunas_extras(linha_csv: Any) -> None:
    """A linha real tem 307 colunas; so as 222 de contorno exigidas saem."""
    assert len(linha_csv) == 307
    assert len(linha_de_csv(linha_csv)) == 222
