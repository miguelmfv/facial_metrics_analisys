"""Testes do classificador de formato de armacao (D-79, D-80, D-81).

Dois eixos:

1. **a carga e desconfiada**: o artefato errado, o artefato truncado e o
   artefato internamente inconsistente tem de falhar na carga, com mensagem
   que diz o que fazer. O erro que isto previne ja aconteceu neste projeto --
   carregar o modelo de armacao achando que era o de rosto, e versionar a saida
   (ver ``legado/execucoes-invalidas/``). Agora que existem dois artefatos de
   verdade, o erro e possivel nas duas direcoes;
2. **a previsao e honesta**: o ranking sai ordenado, soma 1, e as metricas
   medidas viajam dentro do artefato -- inclusive a comparacao que mostra que
   ele nao supera um ranking fixo por frequencia.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest

from vizo_rosto import carregar_armacao, recomendar
from vizo_rosto.modelo import (
    FeatureNaoFinitaError,
    FeaturesIncompativeisError,
    ModeloFormatoRosto,
    ModeloIncompativelError,
)
from vizo_rosto.modelo_armacao import (
    CLASSES_ESPERADAS_ARMACAO,
    CONJUNTOS_ENTRADA,
    PREFIXO_PROB_ROSTO,
    ModeloFormatoArmacao,
    ProbabilidadeDeRostoAusenteError,
)
from vizo_rosto.recomendacao import calcular_features


class SemProba:
    """Estimador de mentira, sem ``predict_proba``.

    No nivel do modulo, e nao dentro do teste, porque ``joblib.dump`` serializa
    a classe por referencia e uma classe local nao tem caminho de importacao.
    """

    classes_ = ("a", "b")


# ===========================================================================
# O artefato versionado
# ===========================================================================


def test_o_artefato_declara_o_que_ele_e(modelo_armacao: ModeloFormatoArmacao) -> None:
    assert modelo_armacao.conjunto_entrada in CONJUNTOS_ENTRADA
    assert modelo_armacao.classes == CLASSES_ESPERADAS_ARMACAO
    assert modelo_armacao.alvo == "source_type"
    assert len(modelo_armacao.features) == len(set(modelo_armacao.features)), (
        "nome de coluna repetido na lista de features do artefato"
    )


def test_o_conjunto_de_entrada_bate_com_as_colunas(
    modelo_armacao: ModeloFormatoArmacao,
) -> None:
    """D-80: o conjunto declarado e as colunas salvas contam a mesma historia."""
    if modelo_armacao.conjunto_entrada == "features":
        assert not modelo_armacao.colunas_de_rosto
    else:
        assert modelo_armacao.colunas_de_rosto
        assert all(
            nome.startswith(PREFIXO_PROB_ROSTO) for nome in modelo_armacao.colunas_de_rosto
        )
    assert modelo_armacao.usa_probabilidades_de_rosto == bool(modelo_armacao.colunas_de_rosto)


def test_o_artefato_carrega_a_procedencia_e_as_metricas(
    modelo_armacao: ModeloFormatoArmacao,
) -> None:
    """Sem isso o aviso ao usuario teria de citar numero chumbado em codigo."""
    assert modelo_armacao.procedencia.get("csv")
    assert modelo_armacao.procedencia.get("linhas", 0) > 0
    assert modelo_armacao.procedencia.get("grupos", 0) > 0

    medidas = modelo_armacao.metricas_do_escolhido
    assert medidas, "o artefato nao diz qual estimador foi versionado"
    for chave in ("acuracia", "acuracia_balanceada", "f1_macro", "top3"):
        assert 0.0 <= medidas[chave] <= 1.0

    assert modelo_armacao.referencias["acuracia_classe_majoritaria"] > 0
    assert modelo_armacao.referencias["top3_ranking_fixo_das_3_mais_frequentes"] > 0


def test_a_comparacao_com_o_ranking_fixo_e_respondida_e_nao_omitida(
    modelo_armacao: ModeloFormatoArmacao,
) -> None:
    """A pergunta honesta do produto: o ranking aprendido vale mais que um fixo?

    O teste **nao** exige que a resposta seja sim -- hoje ela e nao, e isso e o
    resultado do trabalho. Ele exige que o artefato carregue os dois numeros
    para que a resposta possa ser dada sem consultar documento nenhum.
    """
    resposta = modelo_armacao.supera_ranking_fixo
    assert resposta is not None, (
        "o artefato nao tem como comparar o top-3 medido com o do ranking fixo"
    )
    medido = modelo_armacao.metricas_do_escolhido["top3"]
    referencia = modelo_armacao.referencias["top3_ranking_fixo_das_3_mais_frequentes"]
    assert resposta == (medido > referencia)


# ===========================================================================
# A carga desconfiada
# ===========================================================================


def test_carregar_o_modelo_de_rosto_como_armacao_falha_nomeando_o_erro(
    raiz: Path,
) -> None:
    """O erro historico deste projeto, na direcao inversa.

    A recusa acontece antes mesmo da conferencia de classes: o artefato de
    rosto nao tem a chave ``conjunto_entrada``, que so o artefato de armacao
    carrega. Vale registrar que sao **duas** barreiras independentes, e que a
    primeira a disparar e a do contrato do arquivo -- a mensagem nomeia a chave
    que falta e diz como gerar o artefato certo.
    """
    caminho = raiz / "modelos" / "modelo_formato_rosto_sem_linhas_branco.joblib"
    if not caminho.is_file():
        pytest.skip("modelo de rosto nao encontrado")
    with pytest.raises(ModeloIncompativelError) as erro:
        carregar_armacao(caminho)
    mensagem = str(erro.value)
    assert "conjunto_entrada" in mensagem
    assert "treinar_modelo_armacao.py" in mensagem


def test_a_conferencia_de_classes_tambem_pega_o_modelo_de_rosto(raiz: Path) -> None:
    """A segunda barreira, isolada da primeira.

    Reembala o estimador de rosto num pacote formalmente valido de armacao. O
    contrato do arquivo passa; as classes nao. E este o caminho em que o erro
    historico aconteceria hoje: alguem gerar um artefato com o script certo e o
    CSV errado.
    """
    caminho = raiz / "modelos" / "modelo_formato_rosto_sem_linhas_branco.joblib"
    if not caminho.is_file():
        pytest.skip("modelo de rosto nao encontrado")
    pacote_de_rosto = joblib.load(caminho)

    import tempfile

    with tempfile.TemporaryDirectory() as pasta:
        disfarcado = Path(pasta) / "disfarcado.joblib"
        joblib.dump(
            {
                "modelo": pacote_de_rosto["modelo"],
                "features": list(pacote_de_rosto["features"]),
                "conjunto_entrada": "features",
                "alvo": "source_type",
            },
            disfarcado,
        )
        with pytest.raises(ModeloIncompativelError) as erro:
            carregar_armacao(disfarcado)

    assert "diamante" in str(erro.value)
    assert "formatos de ROSTO" in str(erro.value)


def test_arquivo_inexistente_diz_como_gerar_o_artefato(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as erro:
        carregar_armacao(tmp_path / "nao_existe.joblib")
    assert "treinar_modelo_armacao.py" in str(erro.value)


def test_artefato_sem_as_chaves_obrigatorias_e_recusado(
    tmp_path: Path, modelo_armacao: ModeloFormatoArmacao
) -> None:
    caminho = tmp_path / "truncado.joblib"
    joblib.dump({"modelo": modelo_armacao.estimador}, caminho)
    with pytest.raises(ModeloIncompativelError) as erro:
        carregar_armacao(caminho)
    assert "features" in str(erro.value)


def test_artefato_sem_predict_proba_e_recusado(tmp_path: Path) -> None:
    """Sem probabilidade nao existe ranking: a recomendacao E a ordem."""
    caminho = tmp_path / "sem_proba.joblib"
    joblib.dump(
        {"modelo": SemProba(), "features": ["x"], "conjunto_entrada": "features"}, caminho
    )
    with pytest.raises(ModeloIncompativelError) as erro:
        carregar_armacao(caminho)
    assert "predict_proba" in str(erro.value)


def test_conjunto_declarado_que_nao_bate_com_as_colunas_e_recusado(
    tmp_path: Path, modelo_armacao: ModeloFormatoArmacao
) -> None:
    """Artefato internamente inconsistente nao passa como se estivesse bom."""
    caminho = tmp_path / "inconsistente.joblib"
    joblib.dump(
        {
            "modelo": modelo_armacao.estimador,
            "features": list(modelo_armacao.features),  # tem colunas p_rosto_*
            "conjunto_entrada": "features",  # ... mas declara que nao tem
        },
        caminho,
    )
    with pytest.raises(ModeloIncompativelError) as erro:
        carregar_armacao(caminho)
    assert "conjunto" in str(erro.value).lower()


def test_conjunto_de_entrada_desconhecido_e_recusado(
    tmp_path: Path, modelo_armacao: ModeloFormatoArmacao
) -> None:
    caminho = tmp_path / "conjunto_estranho.joblib"
    joblib.dump(
        {
            "modelo": modelo_armacao.estimador,
            "features": list(modelo_armacao.features),
            "conjunto_entrada": "features+astrologia",
        },
        caminho,
    )
    with pytest.raises(ModeloIncompativelError):
        carregar_armacao(caminho)


# ===========================================================================
# A previsao
# ===========================================================================


def test_ranquear_sem_a_etapa_de_rosto_falha_dizendo_a_ordem_das_etapas(
    modelo_armacao: ModeloFormatoArmacao,
    modelo: ModeloFormatoRosto,
    linha_csv: pd.Series,
) -> None:
    if not modelo_armacao.usa_probabilidades_de_rosto:
        pytest.skip("este artefato nao usa probabilidades de rosto")
    features = calcular_features(linha_csv)
    with pytest.raises(ProbabilidadeDeRostoAusenteError) as erro:
        modelo_armacao.ranquear(features)
    assert "duas etapas" in str(erro.value)


def test_feature_faltando_e_recusada_com_as_duas_listas_separadas(
    modelo_armacao: ModeloFormatoArmacao,
    modelo: ModeloFormatoRosto,
    linha_csv: pd.Series,
) -> None:
    features = calcular_features(linha_csv)
    previsao = modelo.prever(features)
    alguma = next(iter(modelo_armacao.colunas_de_face))
    incompleta = {k: v for k, v in features.items() if k != alguma}
    with pytest.raises(FeaturesIncompativeisError) as erro:
        modelo_armacao.ranquear(incompleta, previsao)
    assert alguma in str(erro.value)


def test_feature_nao_finita_e_recusada_em_vez_de_imputada(
    modelo_armacao: ModeloFormatoArmacao,
    modelo: ModeloFormatoRosto,
    linha_csv: pd.Series,
) -> None:
    """D-56 vale na segunda etapa tambem.

    O pipeline do artefato tem um ``SimpleImputer``, que aceitaria o ``NaN`` em
    silencio e devolveria um ranking diferente do verdadeiro. Para **uma**
    medicao nao existe mediana de coluna; a recusa e a resposta correta.
    """
    features = calcular_features(linha_csv)
    previsao = modelo.prever(features)
    alguma = next(iter(modelo_armacao.colunas_de_face))
    estragada = dict(features)
    estragada[alguma] = math.nan
    with pytest.raises(FeatureNaoFinitaError) as erro:
        modelo_armacao.ranquear(estragada, previsao)
    assert alguma in str(erro.value)


def test_o_ranking_e_uma_distribuicao_ordenada(
    modelo_armacao: ModeloFormatoArmacao,
    modelo: ModeloFormatoRosto,
    linha_csv: pd.Series,
) -> None:
    features = calcular_features(linha_csv)
    previsao_armacao = modelo_armacao.ranquear(features, modelo.prever(features))

    assert len(previsao_armacao.ranking) == len(modelo_armacao.classes)
    assert sum(previsao_armacao.probabilidades.values()) == pytest.approx(1.0, abs=1e-9)
    assert previsao_armacao.formato == previsao_armacao.ranking[0].formato
    assert previsao_armacao.confianca == previsao_armacao.ranking[0].probabilidade
    assert previsao_armacao.margem == pytest.approx(
        previsao_armacao.ranking[0].probabilidade - previsao_armacao.ranking[1].probabilidade
    )
    assert previsao_armacao.conjunto_entrada == modelo_armacao.conjunto_entrada
    assert len(previsao_armacao.top(3)) == 3
    with pytest.raises(ValueError):
        previsao_armacao.top(0)


def test_a_ordem_das_features_na_chamada_nao_altera_o_resultado(
    modelo_armacao: ModeloFormatoArmacao,
    modelo: ModeloFormatoRosto,
    linha_csv: pd.Series,
) -> None:
    """A linha e montada por nome; quem chama nao precisa saber a ordem do treino."""
    features = calcular_features(linha_csv)
    previsao = modelo.prever(features)
    invertidas = dict(reversed(list(features.items())))

    assert modelo_armacao.ranquear(invertidas, previsao) == modelo_armacao.ranquear(
        features, previsao
    )


def test_probabilidade_de_rosto_diferente_muda_o_ranking(
    modelo_armacao: ModeloFormatoArmacao,
    modelo: ModeloFormatoRosto,
    df_teste: pd.DataFrame,
) -> None:
    """Prova que a saida da etapa 1 realmente entra na etapa 2 (D-80).

    Se as colunas ``p_rosto_*`` fossem ignoradas pelo estimador, duas chamadas
    com as mesmas features e previsoes de rosto diferentes dariam exatamente o
    mesmo vetor de probabilidades. O teste procura um par de rostos reais em
    que elas diferem.
    """
    if not modelo_armacao.usa_probabilidades_de_rosto:
        pytest.skip("este artefato nao usa probabilidades de rosto")

    features = calcular_features(df_teste.iloc[0])
    primeira = modelo.prever(features)
    outra: Any = None
    for i in range(1, len(df_teste)):
        candidata = modelo.prever(calcular_features(df_teste.iloc[i]))
        if candidata.probabilidades != primeira.probabilidades:
            outra = candidata
            break
    assert outra is not None, "as 78 linhas deram a mesma distribuicao de rosto"

    com_a = modelo_armacao.ranquear(features, primeira).probabilidades
    com_b = modelo_armacao.ranquear(features, outra).probabilidades
    assert com_a != com_b, (
        "trocar a previsao de rosto nao mudou nada: as colunas p_rosto_* nao estao "
        "entrando no classificador de armacao"
    )


def test_o_pacote_inteiro_funciona_com_os_dois_artefatos(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    """O caminho que o backend vai chamar, do jeito que ele vai chamar."""
    rec = recomendar({"contours": _contornos_de(linha_csv)}, modelo, modelo_armacao)
    contrato = rec.para_contrato_backend()
    assert contrato["formato_rosto"] in modelo.classes
    assert all(
        item["formato"] in modelo_armacao.classes for item in contrato["ranking_armacoes"]
    )


def _contornos_de(linha: pd.Series) -> dict[str, list[list[float]]]:
    """Remonta o dicionario de contornos a partir das colunas ``ct_*`` da linha."""
    from vizo_rosto.entrada import CONTORNOS_EXIGIDOS

    return {
        nome: [
            [float(linha[f"ct_{nome}_x{i}"]), float(linha[f"ct_{nome}_y{i}"])]
            for i in range(n)
        ]
        for nome, n in CONTORNOS_EXIGIDOS.items()
    }
