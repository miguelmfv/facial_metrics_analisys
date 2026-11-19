"""Testes da carga e do uso do modelo de formato de rosto."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import pytest

from vizo_rosto.modelo import (
    CLASSES_ESPERADAS,
    FeatureNaoFinitaError,
    FeaturesIncompativeisError,
    ModeloFormatoRosto,
    ModeloIncompativelError,
    carregar,
)
from vizo_rosto.recomendacao import calcular_features


# Estes dublês ficam no nivel do modulo, e nao dentro das funcoes de teste,
# porque o joblib precisa picklar a CLASSE por referencia (modulo + qualname) e
# uma classe definida dentro de uma funcao nao e encontravel por esse caminho.
class EstimadorDeArmacao:
    """Dublê do modelo de ARMACAO, para provar que a guarda de carga o pega."""

    classes_ = ["aviador", "gatinho", "ovalado", "quadrado", "redondo", "retangular"]

    def predict_proba(self, X: Any) -> Any:  # pragma: no cover - nunca chamado
        raise AssertionError("nao deveria chegar a prever")


class EstimadorSoPredict:
    """Dublê de estimador sem predict_proba, que D-49 nao admite."""

    classes_ = list(CLASSES_ESPERADAS)

    def predict(self, X: Any) -> Any:  # pragma: no cover - nunca chamado
        return ["oval"]


def test_carrega_sem_aviso_de_versao(raiz: Path) -> None:
    """O requirements fixa a versao de sklearn gravada no pickle; prove que basta.

    Com ``warnings.simplefilter('error')``, um ``InconsistentVersionWarning``
    quebra o teste. E esse o ponto: este teste e a guarda de regressao contra
    alguem subir a versao de scikit-learn sem refazer a verificacao.
    """
    caminho = raiz / "modelos" / "modelo_formato_rosto_sem_linhas_branco.joblib"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        modelo = carregar(caminho)
    assert modelo.classes == CLASSES_ESPERADAS


def test_contrato_do_artefato(modelo: ModeloFormatoRosto) -> None:
    assert modelo.classes == ("diamante", "oval", "quadrado", "redondo")
    assert len(modelo.features) == 29
    assert modelo.features[0] == "h_w"
    assert modelo.features[-1] == "olhos_larg"
    assert modelo.alvo == "source_type"


def test_preve_e_devolve_as_quatro_probabilidades(
    modelo: ModeloFormatoRosto, linha_csv: Any
) -> None:
    previsao = modelo.prever(calcular_features(linha_csv))
    assert previsao.formato in CLASSES_ESPERADAS
    assert set(previsao.probabilidades) == set(CLASSES_ESPERADAS)
    assert previsao.probabilidades[previsao.formato] == max(previsao.probabilidades.values())
    assert pytest.approx(sum(previsao.probabilidades.values()), abs=1e-9) == 1.0


def test_probabilidades_vem_ordenadas(modelo: ModeloFormatoRosto, linha_csv: Any) -> None:
    valores = list(modelo.prever(calcular_features(linha_csv)).probabilidades.values())
    assert valores == sorted(valores, reverse=True)


def test_confianca_e_margem(modelo: ModeloFormatoRosto, linha_csv: Any) -> None:
    previsao = modelo.prever(calcular_features(linha_csv))
    valores = sorted(previsao.probabilidades.values(), reverse=True)
    assert previsao.confianca == valores[0]
    assert previsao.margem == pytest.approx(valores[0] - valores[1])


def test_top_devolve_os_n_mais_provaveis(modelo: ModeloFormatoRosto, linha_csv: Any) -> None:
    previsao = modelo.prever(calcular_features(linha_csv))
    assert previsao.top(1) == [previsao.formato]
    assert len(previsao.top(2)) == 2
    assert previsao.top(4) == list(previsao.probabilidades)
    with pytest.raises(ValueError):
        previsao.top(0)


def test_previsao_e_deterministica(modelo: ModeloFormatoRosto, linha_csv: Any) -> None:
    """Sem isso nenhum resultado serve para uma monografia."""
    features = calcular_features(linha_csv)
    primeira = modelo.prever(features)
    for _ in range(3):
        assert modelo.prever(features) == primeira


def test_rejeita_feature_faltando(modelo: ModeloFormatoRosto, linha_csv: Any) -> None:
    features = calcular_features(linha_csv)
    del features["ang_queixo"]
    with pytest.raises(FeaturesIncompativeisError) as erro:
        modelo.prever(features)
    msg = str(erro.value)
    assert "ang_queixo" in msg
    assert "Faltando" in msg


def test_rejeita_feature_sobrando(modelo: ModeloFormatoRosto, linha_csv: Any) -> None:
    features = calcular_features(linha_csv)
    features["feature_inventada"] = 1.0
    with pytest.raises(FeaturesIncompativeisError) as erro:
        modelo.prever(features)
    msg = str(erro.value)
    assert "feature_inventada" in msg
    assert "desconhecidas pelo modelo" in msg


def test_mensagem_de_incompatibilidade_proibe_retreinar(
    modelo: ModeloFormatoRosto, linha_csv: Any
) -> None:
    """D-48: a saida facil e errada seria retreinar; a mensagem tem de dizer isso."""
    features = calcular_features(linha_csv)
    del features["h_w"]
    with pytest.raises(FeaturesIncompativeisError) as erro:
        modelo.prever(features)
    assert "Nao retreine" in str(erro.value)


@pytest.mark.parametrize("valor", [float("nan"), float("inf"), float("-inf")])
def test_rejeita_feature_nao_finita_nomeando_a_culpada(
    modelo: ModeloFormatoRosto, linha_csv: Any, valor: float
) -> None:
    """D-56. NaN e inf, nao so NaN.

    O ponto que justifica este teste: com NaN o RandomForest NAO reclama, ele
    responde outra coisa em silencio. Sem esta checagem o erro nao apareceria.
    """
    features = calcular_features(linha_csv)
    features["h_w"] = valor
    with pytest.raises(FeatureNaoFinitaError) as erro:
        modelo.prever(features)
    msg = str(erro.value)
    assert "h_w" in msg, "a mensagem tem de nomear a feature culpada"
    assert "nova captura" in msg, "a mensagem tem de dizer o que fazer"


def test_arquivo_inexistente_orienta_o_caminho(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as erro:
        carregar(tmp_path / "nao-existe.joblib")
    assert "modelos/modelo_formato_rosto_sem_linhas_branco.joblib" in str(erro.value)


def test_rejeita_joblib_que_nao_e_dicionario(tmp_path: Path) -> None:
    caminho = tmp_path / "estimador-cru.joblib"
    joblib.dump([1, 2, 3], caminho)
    with pytest.raises(ModeloIncompativelError) as erro:
        carregar(caminho)
    assert "lista de features" in str(erro.value)


def test_rejeita_joblib_sem_a_chave_features(tmp_path: Path, modelo: ModeloFormatoRosto) -> None:
    caminho = tmp_path / "sem-features.joblib"
    joblib.dump({"modelo": modelo.estimador}, caminho)
    with pytest.raises(ModeloIncompativelError) as erro:
        carregar(caminho)
    assert "features" in str(erro.value)


def test_rejeita_modelo_de_armacao_no_lugar_do_de_rosto(
    tmp_path: Path, modelo: ModeloFormatoRosto
) -> None:
    """A guarda que existe porque o erro JA ocorreu neste projeto.

    A saida da execucao errada chegou a ser versionada: um CSV de 7.416 linhas
    com colunas ``p_rosto_aviador``, ``p_rosto_gatinho``, ``p_rosto_retangular``.
    Este teste garante que, se acontecer de novo, falha na carga.
    """

    caminho = tmp_path / "modelo-de-armacao.joblib"
    joblib.dump(
        {"modelo": EstimadorDeArmacao(), "features": list(modelo.features), "alvo": "source_type"},
        caminho,
    )
    with pytest.raises(ModeloIncompativelError) as erro:
        carregar(caminho)
    msg = str(erro.value)
    assert "ARMACAO" in msg, "a mensagem tem de explicar a causa provavel"
    assert "modelos/modelo_formato_rosto_sem_linhas_branco.joblib" in msg


def test_rejeita_estimador_sem_predict_proba(tmp_path: Path) -> None:
    """D-49 exige as probabilidades das 4 classes, nao so o top1."""

    caminho = tmp_path / "so-predict.joblib"
    joblib.dump(
        {"modelo": EstimadorSoPredict(), "features": ["h_w"], "alvo": "source_type"}, caminho
    )
    with pytest.raises(ModeloIncompativelError) as erro:
        carregar(caminho)
    assert "predict_proba" in str(erro.value)


def test_rejeita_artefato_internamente_inconsistente(
    tmp_path: Path, modelo: ModeloFormatoRosto
) -> None:
    """Estimador ajustado com 29 features, mas lista salva com 3 nomes."""
    caminho = tmp_path / "inconsistente.joblib"
    joblib.dump(
        {"modelo": modelo.estimador, "features": ["h_w", "ang_queixo", "circularidade"]},
        caminho,
    )
    with pytest.raises(ModeloIncompativelError) as erro:
        carregar(caminho)
    assert "29" in str(erro.value) and "3" in str(erro.value)
