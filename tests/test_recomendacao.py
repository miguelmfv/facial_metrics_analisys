"""Teste ponta a ponta da cadeia de dois classificadores (D-79, D-80).

A entrada e real: as 78 linhas de
``arquivos_csv/faces-teste-supervisionado-formato-rosto.csv``, pelas fixtures do
``conftest.py``. Nenhum rosto sintetico.

Duas coisas que este arquivo existe para travar:

1. **a primeira etapa nao se moveu.** A troca da segunda etapa (visagismo ->
   classificador) nao pode alterar em nada a previsao de formato de rosto. A
   linha-base de probabilidades continua conferida digito a digito;
2. **a segunda etapa e de verdade um modelo.** O ranking tem de variar com o
   rosto, estar ordenado por probabilidade e cobrir as classes do artefato --
   e nao ser uma lista chumbada com cara de previsao.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pandas as pd
import pytest

from vizo_rosto import recomendar
from vizo_rosto.entrada import EntradaInvalidaError
from vizo_rosto.modelo import FeatureNaoFinitaError, ModeloFormatoRosto
from vizo_rosto.modelo_armacao import ModeloFormatoArmacao
from vizo_rosto.recomendacao import (
    N_RECOMENDADOS_PADRAO,
    PRINCIPIO,
    ModeloDeArmacaoAusenteError,
    calcular_features,
    montar_aviso,
)

# Valores de referencia, medidos na PRIMEIRA linha de
# arquivos_csv/faces-teste-supervisionado-formato-rosto.csv
# (arquivo 06c919bde2344cd1eadcf61658455438.jpg) com o artefato versionado.
# Conferem, digito a digito, com a linha-base colhida ANTES do refactor da
# rodada 3, em
# contexto/desenvolvimento/dados/linha-base-T2-resultados_previsao-rodada-3-tarefa-1.csv
#
# Se este teste quebrar, uma de duas coisas aconteceu: o features_faciais.py
# mudou de logica (proibido por D-50) ou o modelo de rosto foi substituido
# (proibido por D-48). Nos dois casos e defeito, nao melhoria. D-79 trocou a
# SEGUNDA etapa; estes numeros sao da primeira e tem de continuar iguais.
PRIMEIRA_LINHA_ESPERADA = {
    "formato": "diamante",
    "probabilidades": {
        "diamante": 0.508,
        "oval": 0.313,
        "quadrado": 0.108,
        "redondo": 0.071,
    },
}


def test_ponta_a_ponta_com_linha_real(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    """O caso que fecha a tarefa: entrada conhecida, as duas etapas conferidas."""
    rec = recomendar(linha_csv, modelo, modelo_armacao)

    assert rec.formato_rosto == PRIMEIRA_LINHA_ESPERADA["formato"]
    for classe, esperado in PRIMEIRA_LINHA_ESPERADA["probabilidades"].items():
        assert rec.previsao.probabilidades[classe] == pytest.approx(esperado, abs=5e-4), (
            f"a probabilidade de '{classe}' mudou em relacao a linha-base"
        )

    assert rec.armacoes, "a recomendacao tem de trazer pelo menos um formato de armacao"
    assert len(rec.armacoes) == N_RECOMENDADOS_PADRAO
    assert [a.posicao for a in rec.armacoes] == list(range(1, len(rec.armacoes) + 1))
    assert all(a.formato in modelo_armacao.classes for a in rec.armacoes)
    assert rec.formato_armacao == rec.armacoes[0].formato
    assert rec.aviso and rec.principio
    assert len(rec.features) == 29


def test_o_ranking_cobre_todas_as_classes_e_esta_ordenado(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    """O ranking completo e a saida do classificador, nao um corte arbitrario."""
    rec = recomendar(linha_csv, modelo, modelo_armacao)
    ranking = rec.previsao_armacao.ranking

    assert len(ranking) == len(modelo_armacao.classes)
    assert {item.formato for item in ranking} == set(modelo_armacao.classes)
    probabilidades = [item.probabilidade for item in ranking]
    assert probabilidades == sorted(probabilidades, reverse=True), (
        "o ranking tem de estar em ordem decrescente de probabilidade"
    )
    assert sum(probabilidades) == pytest.approx(1.0, abs=1e-9)
    assert [item.posicao for item in ranking] == list(range(1, len(ranking) + 1))


def test_o_ranking_muda_com_o_rosto(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    df_teste: pd.DataFrame,
) -> None:
    """Se a saida fosse igual para todos, nao haveria modelo nenhum ali.

    Este teste nao afirma que o ranking esta **certo** -- as metricas medidas
    dizem que ele acerta no nivel do acaso --, so que ele responde a entrada.
    Um artefato quebrado que devolvesse sempre a mesma ordem passaria por todos
    os outros testes deste arquivo.
    """
    ordens = {
        tuple(rec.previsao_armacao.formatos)
        for rec in (
            recomendar(df_teste.iloc[i], modelo, modelo_armacao)
            for i in range(min(20, len(df_teste)))
        )
    }
    assert len(ordens) > 1, (
        "as 20 primeiras linhas reais sairam com a MESMA ordem de armacoes; "
        "o classificador nao esta reagindo a entrada"
    )


def test_sem_o_classificador_de_armacao_a_falha_e_explicita(
    modelo: ModeloFormatoRosto, linha_csv: pd.Series
) -> None:
    """Chamador escrito para a cadeia antiga falha dizendo o que fazer.

    Importa para o backend: ele chama ``recomendar(medicao, modelo)`` com dois
    argumentos e so vai passar a chamar com tres depois de carregar o segundo
    artefato. Ate la, tem de quebrar alto, nao devolver meia recomendacao.
    """
    with pytest.raises(ModeloDeArmacaoAusenteError) as erro:
        recomendar(linha_csv, modelo)
    assert "carregar_armacao" in str(erro.value)
    assert "VIZO_ROSTO_MODELO_PATH" in str(erro.value)


def test_n_recomendados_corta_o_contrato_sem_mexer_no_ranking(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    """D-84 e parametro, nao constante chumbada."""
    rec = recomendar(linha_csv, modelo, modelo_armacao, n_recomendados=5)
    assert len(rec.armacoes) == 5
    assert len(rec.previsao_armacao.ranking) == len(modelo_armacao.classes)
    assert len(rec.para_contrato_backend()["ranking_armacoes"]) == 5

    with pytest.raises(ValueError):
        recomendar(linha_csv, modelo, modelo_armacao, n_recomendados=0)


def test_pedir_mais_do_que_existe_nao_inventa_posicao(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    rec = recomendar(linha_csv, modelo, modelo_armacao, n_recomendados=99)
    assert len(rec.armacoes) == len(modelo_armacao.classes)


def test_os_dois_caminhos_de_entrada_dao_a_mesma_recomendacao(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
    medicao_json: dict[str, Any],
) -> None:
    """JSON do backend e linha de CSV tem de convergir, no mesmo rosto real.

    Esta e a prova de que o adaptador de D-46 nao distorce nada: se o backend e
    o CSV divergissem, o numero da monografia e o numero do app seriam
    diferentes. Vale agora para as duas etapas.
    """
    do_csv = recomendar(linha_csv, modelo, modelo_armacao)
    do_json = recomendar(medicao_json, modelo, modelo_armacao)

    assert do_json.previsao == do_csv.previsao
    assert do_json.features == do_csv.features
    assert do_json.previsao_armacao == do_csv.previsao_armacao


def test_recomendacao_e_deterministica(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    primeira = recomendar(linha_csv, modelo, modelo_armacao)
    segunda = recomendar(linha_csv, modelo, modelo_armacao)
    assert segunda.previsao == primeira.previsao
    assert segunda.previsao_armacao == primeira.previsao_armacao


def test_nao_exige_rotulo_na_entrada(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    medicao_json: dict[str, Any],
) -> None:
    """A entrada vem de um usuario real: nao existe classe verdadeira para ela."""
    sem_rotulo = {k: v for k, v in medicao_json.items() if k != "source_type"}
    assert (
        recomendar(sem_rotulo, modelo, modelo_armacao).previsao_armacao
        == recomendar(medicao_json, modelo, modelo_armacao).previsao_armacao
    )


def test_rotulo_errado_na_entrada_nao_muda_a_saida(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    medicao_json: dict[str, Any],
) -> None:
    """Se algum caminho passasse a ler o rotulo, este teste pegaria.

    Vale dobrado agora: o rotulo mentiroso do fixture (``source_type``) e
    justamente o nome da coluna de **armacao** com que o segundo modelo foi
    treinado.
    """
    mentiroso = copy.deepcopy(medicao_json)
    mentiroso["source_type"] = "quadrado"
    assert (
        recomendar(mentiroso, modelo, modelo_armacao).previsao_armacao
        == recomendar(medicao_json, modelo, modelo_armacao).previsao_armacao
    )


def test_funciona_para_as_78_linhas_do_csv_de_teste(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    df_teste: pd.DataFrame,
) -> None:
    """Nenhuma linha real do CSV de teste pode falhar nem sair fora do contrato."""
    for i in range(len(df_teste)):
        rec = recomendar(df_teste.iloc[i], modelo, modelo_armacao)
        assert rec.formato_rosto in modelo.classes
        assert sum(rec.previsao.probabilidades.values()) == pytest.approx(1.0, abs=1e-9)
        assert len(rec.armacoes) == N_RECOMENDADOS_PADRAO
        assert all(a.formato in modelo_armacao.classes for a in rec.armacoes)


def test_medicao_com_contorno_quebrado_e_recusada(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    medicao_json: dict[str, Any],
) -> None:
    quebrada = copy.deepcopy(medicao_json)
    quebrada["contours"]["face"] = quebrada["contours"]["face"][:20]
    with pytest.raises(EntradaInvalidaError):
        recomendar(quebrada, modelo, modelo_armacao)


def test_medicao_que_gera_feature_nao_finita_e_recusada(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    medicao_json: dict[str, Any],
) -> None:
    """D-56: colapsar o contorno do rosto em uma linha torna a largura nao medivel.

    A recusa acontece na primeira etapa, antes de o classificador de armacao
    receber qualquer coisa -- que e o que se quer: nenhum valor imputado entra
    na cadeia.
    """
    quebrada = copy.deepcopy(medicao_json)
    y = quebrada["contours"]["face"][0][1]
    quebrada["contours"]["face"] = [[p[0], y] for p in quebrada["contours"]["face"]]
    with pytest.raises((FeatureNaoFinitaError, EntradaInvalidaError)):
        recomendar(quebrada, modelo, modelo_armacao)


def test_formato_de_entrada_irreconhecivel_explica_as_opcoes(
    modelo: ModeloFormatoRosto, modelo_armacao: ModeloFormatoArmacao
) -> None:
    with pytest.raises(EntradaInvalidaError) as erro:
        recomendar({"altura": 1.7, "peso": 70}, modelo, modelo_armacao)
    assert "entrada.py" in str(erro.value)


def test_para_dicionario_e_serializavel_e_mostra_as_duas_etapas(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    """A saida de auditoria leva a incerteza das DUAS etapas, nao so do rosto."""
    d = recomendar(linha_csv, modelo, modelo_armacao).para_dicionario()
    texto = json.dumps(d, ensure_ascii=False)  # nao levanta => serializavel

    assert "descritivo" in d["formato_rosto"]["natureza"]
    assert "classificador" in d["formatos_armacao"]["natureza"]
    assert len(d["formato_rosto"]["probabilidades"]) == 4
    assert len(d["formatos_armacao"]["ranking"]) == len(modelo_armacao.classes)
    assert d["formatos_armacao"]["ranking"][0]["posicao"] == 1
    assert isinstance(d["formatos_armacao"]["ranking"][0]["probabilidade"], float)
    assert d["formatos_armacao"]["conjunto_entrada"] == modelo_armacao.conjunto_entrada
    assert sum(1 for i in d["formatos_armacao"]["ranking"] if i["no_contrato"]) == (
        N_RECOMENDADOS_PADRAO
    )
    assert d["aviso"]
    assert "features" not in d, "as features nao vao por default"
    assert len(texto) > 0


def test_para_dicionario_inclui_features_so_quando_pedido(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    d = recomendar(linha_csv, modelo, modelo_armacao).para_dicionario(incluir_features=True)
    assert len(d["features"]) == 29


def test_calcular_features_devolve_as_29_na_ordem_do_modelo(
    modelo: ModeloFormatoRosto, linha_csv: pd.Series
) -> None:
    features = calcular_features(linha_csv)
    assert list(features) == list(modelo.features)
    assert all(isinstance(v, float) for v in features.values())


# ===========================================================================
# D-79: o visagismo saiu do caminho de execucao
# ===========================================================================


def test_o_modulo_de_recomendacao_nao_importa_visagismo() -> None:
    """A prova mecanica de D-79, que nao depende de ninguem reler o codigo.

    O modulo ``visagismo`` continua existindo no pacote como registro historico
    -- e por isso ``import vizo_rosto.visagismo`` continua funcionando --, mas
    ``recomendacao`` nao pode mais depender dele. Se alguem reintroduzir a
    tabela no caminho, este teste quebra.
    """
    from vizo_rosto import recomendacao

    assert not hasattr(recomendacao, "visagismo"), (
        "recomendacao voltou a importar visagismo; D-79 tirou a tabela do caminho "
        "de execucao"
    )


def test_o_principio_declara_que_o_criterio_e_aprendido(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    """O texto nao pode continuar prometendo visagismo que o codigo nao aplica."""
    rec = recomendar(linha_csv, modelo, modelo_armacao)
    assert rec.principio == PRINCIPIO
    assert "classificador" in rec.principio.lower()
    assert "visagismo" in rec.principio.lower(), (
        "o principio tem de dizer tambem o que a recomendacao NAO e"
    )


# ===========================================================================
# O aviso: acentuado, honesto e vindo do artefato
# ===========================================================================


def test_o_aviso_vem_do_artefato_e_cita_os_numeros_medidos(
    modelo_armacao: ModeloFormatoArmacao,
) -> None:
    """As metricas no aviso sao as daquele .joblib, nao constantes velhas."""
    aviso = montar_aviso(modelo_armacao)
    medidas = modelo_armacao.metricas_do_escolhido

    assert "0,176" in aviso, "o resultado negativo do trabalho viaja junto (D-63)"
    for valor in (medidas["acuracia"], medidas["top3"]):
        assert f"{valor:.2f}".replace(".", ",") in aviso, (
            f"o aviso nao cita a metrica medida {valor:.2f} do artefato em uso"
        )


def test_o_aviso_esta_acentuado(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    """Achado 5.2 do fechamento da rodada 3, tarefa 3, virado teste.

    O aviso anterior era ASCII puro e chegava a tela do app sem acento nenhum,
    parecendo defeito do aplicativo justamente no texto mais lido. O conserto e
    aqui, na origem, porque o backend grava a string como registro historico
    (D-63) e o app nao pode reescreve-la.
    """
    contrato = recomendar(linha_csv, modelo, modelo_armacao).para_contrato_backend()
    for palavra in ("Recomendação", "armação", "é", "não", "sugestão"):
        assert palavra in contrato["aviso"], f"falta '{palavra}' acentuado no aviso"
    assert "Recomendacao" not in contrato["aviso"]
