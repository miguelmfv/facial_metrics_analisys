"""Orquestracao: medicao -> features -> formato de rosto -> armacoes ranqueadas.

Este modulo e a camada que junta as outras quatro. Ele nao calcula geometria
(isso e :mod:`vizo_rosto.features_faciais`), nao sabe desserializar modelo
(isso e :mod:`vizo_rosto.modelo`) e nao decide criterio estetico (isso e
:mod:`vizo_rosto.visagismo`).

A saida tem duas naturezas diferentes dentro dela, e isso esta marcado campo a
campo:

- o **formato de rosto** e **descritivo**: vem de um modelo aprendido;
- os **formatos de armacao** sao **prescritivos**: vem de uma tabela de regras
  declaradas, com fonte citada, e nao foram aprendidos de dado nenhum.

Nada aqui preve armacao a partir do rosto. O experimento central do TCC
mostrou que essa relacao nao e estatisticamente detectavel (p = 0,176).

O que esta saida **nao** faz: escolher armacoes concretas do catalogo. O
catalogo vive no banco do backend, que este repositorio nao acessa. Esta camada
entrega formato de rosto com probabilidades e formatos de armacao ranqueados
com justificativa; casar isso com produto e trabalho do backend (D-49).

Nenhuma funcao deste modulo imprime: todas devolvem dados.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from . import visagismo
from .entrada import CONTORNOS_EXIGIDOS, EntradaInvalidaError, achatar_medicao, linha_de_csv
from .features_faciais import extract
from .modelo import ModeloFormatoRosto, PrevisaoRosto

__all__ = [
    "ArmacaoRecomendada",
    "Recomendacao",
    "calcular_features",
    "recomendar",
]


@dataclass(frozen=True)
class ArmacaoRecomendada:
    """Um formato de armacao no ranking, com a procedencia da recomendacao.

    Attributes:
        posicao: posicao no ranking, comecando em 1.
        formato: o formato de armacao, um dos 7 do dataset.
        justificativa: o criterio visagista que sustenta a posicao.
        sustentacao: quao bem fundamentada a regra esta
            (``convergente``, ``fonte-unica``, ``divergente``).
        fontes: as fontes citadas, ja resolvidas.
        observacao: ressalva especifica desta regra, se houver.
    """

    posicao: int
    formato: str
    justificativa: str
    sustentacao: str
    fontes: tuple[visagismo.Fonte, ...]
    observacao: str = ""


@dataclass(frozen=True)
class Recomendacao:
    """A saida completa para **uma** medicao.

    Attributes:
        previsao: o formato de rosto previsto e a probabilidade das 4 classes.
            Parte **descritiva**, aprendida.
        armacoes: os formatos de armacao ranqueados. Parte **prescritiva**,
            declarada.
        principio: o principio visagista geral que as fontes aplicam.
        aviso: o aviso de que a tabela e rascunho e de que o projeto nao
            encontrou relacao rosto-armacao. Viaja junto com a recomendacao de
            proposito.
        features: as 29 features calculadas, para auditoria.
    """

    previsao: PrevisaoRosto
    armacoes: tuple[ArmacaoRecomendada, ...]
    principio: str
    aviso: str
    features: dict[str, float]

    @property
    def formato_rosto(self) -> str:
        """Atalho para o formato de rosto previsto."""
        return self.previsao.formato

    def para_dicionario(self, incluir_features: bool = False) -> dict[str, Any]:
        """Converte a recomendacao em tipos nativos, serializaveis em JSON.

        Pensado para quem for consumir isto de fora, inclusive o backend quando
        chegar a hora. As chaves separam explicitamente o que e descritivo do
        que e prescritivo.

        Args:
            incluir_features: se ``True``, inclui as 29 features. Fora de
                depuracao, normalmente nao se quer: sao dados biometricos
                derivados e nao acrescentam nada a decisao.
        """
        saida: dict[str, Any] = {
            "formato_rosto": {
                "natureza": "descritivo (modelo aprendido)",
                "previsto": self.previsao.formato,
                "probabilidades": dict(self.previsao.probabilidades),
                "confianca": self.previsao.confianca,
                "margem_1o_2o": self.previsao.margem,
            },
            "formatos_armacao": {
                "natureza": "prescritivo (regra de visagismo declarada, com fonte)",
                "principio": self.principio,
                "ranking": [
                    {
                        "posicao": a.posicao,
                        "formato": a.formato,
                        "justificativa": a.justificativa,
                        "sustentacao": a.sustentacao,
                        "observacao": a.observacao,
                        "fontes": [
                            {
                                "id": f.id,
                                "titulo": f.titulo,
                                "autoria": f.autoria,
                                "url": f.url,
                                "consultado_em": f.consultado_em,
                                "natureza": f.natureza,
                                "ressalvas": f.ressalvas,
                            }
                            for f in a.fontes
                        ],
                    }
                    for a in self.armacoes
                ],
            },
            "aviso": self.aviso,
        }
        if incluir_features:
            saida["features"] = dict(self.features)
        return saida


def _achatar(medicao: Mapping[str, Any]) -> dict[str, float]:
    """Decide qual adaptador usar e devolve a linha achatada (D-46).

    Reconhece pelas chaves presentes, nao por parametro: o chamador nao deveria
    precisar declarar de onde o dado veio.
    """
    if not isinstance(medicao, Mapping):
        # Pode ser um pandas.Series, que e Mapping-like mas nao Mapping.
        if not hasattr(medicao, "keys"):
            raise EntradaInvalidaError(
                f"A medicao veio como {type(medicao).__name__}. Passe o JSON da "
                "medicao (dict com 'contours'), um dict de contornos, ou uma linha "
                "achatada de CSV (dict ou pandas.Series com colunas 'ct_*')."
            )

    chaves = set(medicao.keys())
    if "contours" in chaves or any(nome in chaves for nome in CONTORNOS_EXIGIDOS):
        return achatar_medicao(medicao)
    if any(str(chave).startswith("ct_") for chave in chaves):
        return linha_de_csv(medicao)

    raise EntradaInvalidaError(
        "Nao reconheci o formato da medicao: ela nao tem a chave 'contours', nem "
        "chaves de contorno no nivel de cima, nem colunas comecando com 'ct_'. "
        f"Chaves encontradas (primeiras 10): {sorted(map(str, chaves))[:10]}. "
        "As duas formas aceitas estao documentadas em entrada.py."
    )


def calcular_features(medicao: Mapping[str, Any]) -> dict[str, float]:
    """Calcula as 29 features de uma medicao, nas duas formas de entrada.

    Args:
        medicao: JSON de medicao do backend, dict de contornos, ou linha
            achatada de CSV.

    Returns:
        As 29 features, por nome, na ordem em que
        :func:`vizo_rosto.features_faciais.extract` as gera.

    Raises:
        EntradaInvalidaError: se a medicao nao puder ser convertida.
    """
    linha = _achatar(medicao)
    # extract devolve numpy.float64; converte para float nativo para que a saida
    # seja serializavel e comparavel sem surpresa.
    return {nome: float(valor) for nome, valor in extract(linha).items()}


def recomendar(
    medicao: Mapping[str, Any],
    modelo: ModeloFormatoRosto,
) -> Recomendacao:
    """Gera a recomendacao completa para **uma** medicao.

    Args:
        medicao: JSON de medicao do backend, dict de contornos, ou linha
            achatada de CSV. **Sem rotulo**: nenhum campo de classe e lido.
        modelo: o modelo carregado por :func:`vizo_rosto.modelo.carregar`.
            Injetado de proposito, em vez de carregado aqui dentro: carregar
            500 arvores a cada chamada seria desperdicio, e um cache de modulo
            seria estado global.

    Returns:
        A recomendacao, com a parte descritiva e a prescritiva separadas.

    Raises:
        EntradaInvalidaError: medicao malformada, contorno faltando ou com
            contagem de pontos errada.
        FeaturesIncompativeisError: o ``features_faciais.py`` atual nao gera o
            que o modelo espera.
        FeatureNaoFinitaError: alguma das 29 features e ``NaN`` ou infinito, e
            portanto a medicao nao e previsivel (D-56).
        FormatoRostoSemRegraError: o modelo previu um formato sem regra
            declarada.
    """
    features = calcular_features(medicao)
    previsao = modelo.prever(features)
    regras = visagismo.ranquear(previsao.formato)

    armacoes = tuple(
        ArmacaoRecomendada(
            posicao=regra.posicao,
            formato=regra.formato_armacao,
            justificativa=regra.justificativa,
            sustentacao=regra.sustentacao,
            fontes=tuple(
                visagismo.FONTES[fid] for fid in regra.fontes if fid in visagismo.FONTES
            ),
            observacao=regra.observacao,
        )
        for regra in regras
    )

    return Recomendacao(
        previsao=previsao,
        armacoes=armacoes,
        principio=visagismo.principio_geral(),
        aviso=visagismo.AVISO_RASCUNHO,
        features=features,
    )
