"""Carga e uso do classificador de formato de ARMACAO (D-79, D-80, D-81).

Este modulo e a segunda etapa da cadeia aprendida::

    medicao -> 29 features -> classificador de ROSTO -> p_rosto_*
            -> classificador de ARMACAO -> ranking dos formatos de armacao

Ele e irmao de :mod:`vizo_rosto.modelo` e repete de proposito as mesmas travas:
contrato de features conferido na previsao, recusa de valor nao finito (D-56) e
conferencia das classes na carga. A repeticao e barata e o que ela evita nao e:
ja aconteceu neste projeto carregar o modelo de armacao achando que era o de
rosto, e a saida daquela execucao chegou a ser versionada (ver
``legado/execucoes-invalidas/``).

**O que este modulo nao faz questao de esconder.** O artefato que ele carrega
preve no nivel do acaso: na rodada anterior o melhor modelo ficou em 0,225 de
acuracia contra 0,242 da classe majoritaria, e em 0,627 de top-3 contra ~0,69
de um ranking **fixo** das tres classes mais frequentes. As metricas medidas
viajam dentro do proprio ``.joblib`` e sao expostas em
:attr:`ModeloFormatoArmacao.metricas`, justamente para que quem usa o ranking
possa dize-lo sem ir atras de documento nenhum. A decisao de usar o
classificador assim mesmo e do autor (D-79); a de nao disfarcar o numero e
deste codigo.

O artefato e produzido por ``treinar_modelo_armacao.py``, na raiz do
repositorio. Nada aqui treina.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

from .modelo import (
    FeatureNaoFinitaError,
    FeaturesIncompativeisError,
    ModeloIncompativelError,
    PrevisaoRosto,
)

__all__ = [
    "CLASSES_ESPERADAS_ARMACAO",
    "CONJUNTOS_ENTRADA",
    "PREFIXO_PROB_ROSTO",
    "FormatoArmacao",
    "ModeloFormatoArmacao",
    "PrevisaoArmacao",
    "ProbabilidadeDeRostoAusenteError",
    "carregar_armacao",
]

#: Prefixo das colunas de probabilidade de formato de rosto dentro da entrada
#: do classificador de armacao. E o mesmo nome que
#: ``legado/pipeline_rosto_armacao.py`` usava, de proposito: o artefato treinado
#: agora e as tabelas da monografia falam a mesma lingua.
PREFIXO_PROB_ROSTO: str = "p_rosto_"

#: Os conjuntos de entrada que o classificador de armacao pode ter sido
#: treinado para receber (D-80). O conjunto usado fica gravado no artefato e e
#: lido na carga: trocar de conjunto e retreinar, nao editar codigo.
#:
#: - ``features``: so as 29 features faciais;
#: - ``probs_rosto``: so as 4 probabilidades de formato de rosto;
#: - ``features+probs_rosto``: as duas coisas. E o default do treino e o
#:   conjunto central da comparacao do pipeline original.
CONJUNTOS_ENTRADA: tuple[str, ...] = ("features", "probs_rosto", "features+probs_rosto")

#: Os 7 formatos de armacao do dataset, em ordem alfabetica -- que e a ordem em
#: que o ``scikit-learn`` devolve ``classes_`` para rotulos de texto.
#:
#: Sem acento e sem cedilha de proposito: sao os nomes das pastas do dataset e
#: os valores gravados na coluna ``source_type``. O texto bonito e assunto da
#: camada de apresentacao, nao do rotulo.
CLASSES_ESPERADAS_ARMACAO: tuple[str, ...] = (
    "aviador",
    "coracao",
    "gatinho",
    "ovalado",
    "quadrado",
    "redondo",
    "retangular",
)


class ProbabilidadeDeRostoAusenteError(ValueError):
    """O modelo exige as probabilidades de rosto e elas nao foram passadas.

    Acontece quando o artefato foi treinado com ``probs_rosto`` ou
    ``features+probs_rosto`` e :meth:`ModeloFormatoArmacao.ranquear` e chamado
    sem a previsao de rosto. E erro de uso, nao de dado: a cadeia de D-80 tem
    duas etapas e a primeira nao e opcional.
    """


@dataclass(frozen=True)
class FormatoArmacao:
    """Um formato de armacao no ranking.

    Attributes:
        posicao: posicao no ranking, comecando em 1.
        formato: o formato de armacao, uma das classes do modelo.
        probabilidade: probabilidade que o estimador deu a esta classe.
            **Nao cruza a fronteira para o backend** (D-66/D-83): existe para
            auditoria, para a monografia e para quem quiser decidir algo com
            base na incerteza.
    """

    posicao: int
    formato: str
    probabilidade: float


@dataclass(frozen=True)
class PrevisaoArmacao:
    """O ranking de formatos de armacao para **uma** medicao.

    Devolve **todas** as classes ranqueadas, nao so as primeiras. Cortar o
    ranking e decisao de quem consome (``n_recomendados`` em
    :func:`vizo_rosto.recomendacao.recomendar`), nao deste modulo.

    Attributes:
        ranking: as classes, da mais provavel para a menos provavel.
        conjunto_entrada: qual conjunto de D-80 o modelo consumiu, copiado do
            artefato para que a saida de auditoria diga como o numero nasceu.
    """

    ranking: tuple[FormatoArmacao, ...]
    conjunto_entrada: str

    @property
    def formato(self) -> str:
        """O primeiro colocado."""
        return self.ranking[0].formato

    @property
    def formatos(self) -> tuple[str, ...]:
        """Os formatos, em ordem de ranking."""
        return tuple(item.formato for item in self.ranking)

    @property
    def probabilidades(self) -> dict[str, float]:
        """Probabilidade de cada formato, da maior para a menor."""
        return {item.formato: item.probabilidade for item in self.ranking}

    @property
    def confianca(self) -> float:
        """Probabilidade do primeiro colocado."""
        return self.ranking[0].probabilidade

    @property
    def margem(self) -> float:
        """Diferenca entre o 1o e o 2o colocado.

        Com este modelo a margem costuma ser pequena, e isso e informacao: um
        ranking cujos dois primeiros estao a milesimos de distancia nao esta
        ordenando nada de robusto.
        """
        if len(self.ranking) < 2:
            return self.ranking[0].probabilidade
        return self.ranking[0].probabilidade - self.ranking[1].probabilidade

    def top(self, n: int = 3) -> tuple[FormatoArmacao, ...]:
        """Os ``n`` primeiros colocados."""
        if n < 1:
            raise ValueError(f"n tem de ser >= 1, recebido {n}.")
        return self.ranking[:n]


@dataclass(frozen=True)
class ModeloFormatoArmacao:
    """O classificador de formato de armacao e o contrato que veio com ele.

    Nao instancie diretamente: use :func:`carregar_armacao`.

    Attributes:
        estimador: o pipeline ``scikit-learn`` desserializado.
        features: os nomes das colunas de entrada, **na ordem do treino**.
            Inclui as colunas ``p_rosto_*`` quando o conjunto as usa.
        conjunto_entrada: um de :data:`CONJUNTOS_ENTRADA`.
        alvo: a coluna de rotulo usada no treino, guardada como procedencia.
        metricas: as metricas medidas por validacao agrupada no treino, por
            nome de estimador. Gravadas dentro do artefato.
        referencias: as referencias do acaso medidas no mesmo dataset,
            inclusive o top-3 de um ranking fixo das 3 classes mais frequentes.
        procedencia: CSV, numero de linhas e grupos, seed, data, versoes.
        caminho: de onde o artefato foi carregado.
    """

    estimador: Any
    features: tuple[str, ...]
    conjunto_entrada: str
    alvo: str
    caminho: Path
    metricas: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    referencias: Mapping[str, float] = field(default_factory=dict)
    procedencia: Mapping[str, Any] = field(default_factory=dict)

    @property
    def classes(self) -> tuple[str, ...]:
        """As classes que o estimador preve, em ordem."""
        return tuple(str(c) for c in self.estimador.classes_)

    @property
    def colunas_de_rosto(self) -> tuple[str, ...]:
        """As colunas de entrada que vem da previsao de rosto."""
        return tuple(nome for nome in self.features if nome.startswith(PREFIXO_PROB_ROSTO))

    @property
    def colunas_de_face(self) -> tuple[str, ...]:
        """As colunas de entrada que vem das 29 features faciais."""
        return tuple(nome for nome in self.features if not nome.startswith(PREFIXO_PROB_ROSTO))

    @property
    def usa_probabilidades_de_rosto(self) -> bool:
        """Se a previsao de rosto e obrigatoria para ranquear."""
        return bool(self.colunas_de_rosto)

    @property
    def metricas_do_escolhido(self) -> Mapping[str, float]:
        """As metricas do estimador que foi de fato versionado."""
        nome = str(self.procedencia.get("modelo_escolhido", "")) or str(
            getattr(self, "_modelo_escolhido", "")
        )
        if nome and nome in self.metricas:
            return self.metricas[nome]
        # Artefato antigo ou montado a mao: sem como saber qual linha e a dele.
        return {}

    @property
    def supera_ranking_fixo(self) -> bool | None:
        """Se o top-3 medido supera o ranking fixo das 3 classes mais frequentes.

        ``None`` quando o artefato nao carrega as duas medidas. Esta e a
        pergunta honesta para um produto que entrega um ranking de 3: se a
        resposta for ``False``, a ordem que o modelo produz nao acerta mais que
        uma lista chumbada por frequencia, e o aviso ao usuario tem de dize-lo.
        """
        medido = self.metricas_do_escolhido.get("top3")
        referencia = self.referencias.get("top3_ranking_fixo_das_3_mais_frequentes")
        if medido is None or referencia is None:
            return None
        return bool(medido > referencia)

    def _montar_linha(
        self,
        features: Mapping[str, float],
        previsao_rosto: PrevisaoRosto | None,
    ) -> pd.DataFrame:
        """Monta a linha de entrada na ordem exata do treino.

        Monta **por nome**, nunca por posicao: o chamador passa as 29 features
        em qualquer ordem e a previsao de rosto como objeto, e a ordem do treino
        e imposta aqui dentro.
        """
        if self.usa_probabilidades_de_rosto and previsao_rosto is None:
            raise ProbabilidadeDeRostoAusenteError(
                f"O modelo em '{self.caminho}' foi treinado com o conjunto "
                f"'{self.conjunto_entrada}' e espera as colunas {list(self.colunas_de_rosto)}, "
                "que vem da previsao de formato de rosto.\n"
                "  O que fazer: rode o classificador de rosto primeiro e passe a "
                "PrevisaoRosto para ranquear(). A cadeia de D-80 tem duas etapas, nesta "
                "ordem."
            )

        valores: dict[str, float] = {}
        faltando_face: list[str] = []
        faltando_rosto: list[str] = []

        for nome in self.features:
            if nome.startswith(PREFIXO_PROB_ROSTO):
                classe = nome[len(PREFIXO_PROB_ROSTO) :]
                probabilidades = previsao_rosto.probabilidades if previsao_rosto else {}
                if classe not in probabilidades:
                    faltando_rosto.append(nome)
                    continue
                valores[nome] = float(probabilidades[classe])
            else:
                if nome not in features:
                    faltando_face.append(nome)
                    continue
                valores[nome] = float(features[nome])

        if faltando_face or faltando_rosto:
            raise FeaturesIncompativeisError(
                "A entrada oferecida nao casa com o que o classificador de armacao foi "
                "treinado para receber.\n"
                f"  Conjunto de entrada do artefato: '{self.conjunto_entrada}' "
                f"({len(self.features)} colunas).\n"
                f"  Features faciais faltando ({len(faltando_face)}): {faltando_face}\n"
                f"  Colunas de probabilidade de rosto faltando ({len(faltando_rosto)}): "
                f"{faltando_rosto}\n"
                "  Causa tipica da segunda lista: o modelo de rosto que gerou as "
                "probabilidades no treino preve classes diferentes das do modelo de rosto "
                "carregado agora. Os dois artefatos sao um par; nao se trocam "
                "separadamente."
            )

        nao_finitas = {
            nome: valor for nome, valor in valores.items() if not math.isfinite(valor)
        }
        if nao_finitas:
            detalhe = ", ".join(f"{nome}={valor}" for nome, valor in sorted(nao_finitas.items()))
            raise FeatureNaoFinitaError(
                f"{len(nao_finitas)} colunas de entrada do classificador de armacao nao sao "
                f"numeros finitos: {detalhe}.\n"
                "  Para uma medicao unica nao existe mediana de coluna com que preencher, e "
                "o estimador aceitaria o valor ausente sem reclamar, devolvendo um ranking "
                "diferente do verdadeiro sem nenhum aviso (D-56).\n"
                "  O que fazer: descarte esta medicao e peca uma nova captura."
            )

        return pd.DataFrame([[valores[nome] for nome in self.features]], columns=list(self.features))

    def ranquear(
        self,
        features: Mapping[str, float],
        previsao_rosto: PrevisaoRosto | None = None,
    ) -> PrevisaoArmacao:
        """Ranqueia os formatos de armacao para **uma** medicao.

        Args:
            features: as 29 features faciais, por nome, em qualquer ordem.
            previsao_rosto: a saida do classificador de rosto. Obrigatoria
                quando o artefato foi treinado com probabilidades de rosto, que
                e o caso do conjunto default de D-80.

        Returns:
            Todas as classes ranqueadas, da mais provavel para a menos.

        Raises:
            ProbabilidadeDeRostoAusenteError: o modelo precisa da etapa de
                rosto e ela nao foi passada.
            FeaturesIncompativeisError: falta coluna que o treino usou.
            FeatureNaoFinitaError: alguma coluna e ``NaN`` ou infinito.
        """
        X = self._montar_linha(features, previsao_rosto)
        probabilidades = self.estimador.predict_proba(X)[0]
        pares = sorted(
            ((classe, float(p)) for classe, p in zip(self.classes, probabilidades)),
            key=lambda par: par[1],
            reverse=True,
        )
        ranking = tuple(
            FormatoArmacao(posicao=i, formato=classe, probabilidade=p)
            for i, (classe, p) in enumerate(pares, start=1)
        )
        return PrevisaoArmacao(ranking=ranking, conjunto_entrada=self.conjunto_entrada)


def carregar_armacao(
    caminho: str | Path,
    classes_esperadas: Sequence[str] | None = CLASSES_ESPERADAS_ARMACAO,
) -> ModeloFormatoArmacao:
    """Carrega o classificador de formato de armacao de um ``.joblib``.

    Args:
        caminho: caminho do artefato. Sem default embutido, pela mesma razao de
            :func:`vizo_rosto.modelo.carregar`.
        classes_esperadas: as classes que o modelo deve prever. O default
            protege contra carregar o modelo de **rosto** por engano, que e o
            espelho do erro que :mod:`vizo_rosto.modelo` protege. Passe
            ``None`` para um artefato treinado com classes excluidas
            (``--excluir coracao``, por exemplo), ciente do risco.

    Returns:
        O modelo, com contrato de features, metricas e procedencia.

    Raises:
        FileNotFoundError: se o arquivo nao existir.
        ModeloIncompativelError: se o artefato nao tiver a estrutura esperada,
            se as classes nao forem as esperadas, ou se ele for internamente
            inconsistente.
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Nao encontrei o classificador de armacao em '{caminho}'. O artefato deste "
            "projeto fica em modelos/modelo_formato_armacao.joblib, relativo a raiz do "
            "repositorio, e e produzido por 'python treinar_modelo_armacao.py'."
        )

    pacote = joblib.load(caminho)

    if not isinstance(pacote, Mapping):
        raise ModeloIncompativelError(
            f"O arquivo '{caminho}' desserializou para {type(pacote).__name__}, mas o "
            "formato esperado e um dicionario com 'modelo', 'features' e "
            "'conjunto_entrada'. Um estimador salvo direto nao carrega a lista de "
            "colunas nem o conjunto de entrada, e sem os dois nao ha como montar a "
            "linha de inferencia na ordem certa."
        )

    faltando = [c for c in ("modelo", "features", "conjunto_entrada") if c not in pacote]
    if faltando:
        raise ModeloIncompativelError(
            f"O arquivo '{caminho}' nao tem as chaves {faltando}. Chaves presentes: "
            f"{sorted(map(str, pacote.keys()))}. Gere o artefato de novo com "
            "'python treinar_modelo_armacao.py'."
        )

    estimador = pacote["modelo"]
    if not hasattr(estimador, "predict_proba"):
        raise ModeloIncompativelError(
            f"O objeto em ['modelo'] de '{caminho}' e um {type(estimador).__name__}, que "
            "nao tem predict_proba. Sem probabilidade nao existe ranking: a recomendacao "
            "e justamente a ordem das classes."
        )

    conjunto = str(pacote["conjunto_entrada"])
    if conjunto not in CONJUNTOS_ENTRADA:
        raise ModeloIncompativelError(
            f"O artefato '{caminho}' declara o conjunto de entrada '{conjunto}', que nao e "
            f"um dos conhecidos: {list(CONJUNTOS_ENTRADA)}."
        )

    features = tuple(str(nome) for nome in pacote["features"])
    tem_rosto = any(nome.startswith(PREFIXO_PROB_ROSTO) for nome in features)
    if (conjunto == "features" and tem_rosto) or (conjunto != "features" and not tem_rosto):
        raise ModeloIncompativelError(
            f"Inconsistencia dentro do artefato '{caminho}': ele declara o conjunto "
            f"'{conjunto}', mas a lista de features {'tem' if tem_rosto else 'nao tem'} "
            f"colunas '{PREFIXO_PROB_ROSTO}*'. O arquivo foi montado a mao ou corrompido; "
            "nao use."
        )

    n_esperado = getattr(estimador, "n_features_in_", None)
    if n_esperado is not None and n_esperado != len(features):
        raise ModeloIncompativelError(
            f"Inconsistencia dentro do artefato '{caminho}': o estimador foi ajustado com "
            f"{n_esperado} colunas, mas a lista salva em ['features'] tem {len(features)} "
            "nomes."
        )

    procedencia = dict(pacote.get("procedencia", {}))
    # 'modelo_escolhido' e de primeiro nivel no artefato; copiado para a
    # procedencia para que metricas_do_escolhido o encontre em um lugar so.
    if "modelo_escolhido" in pacote:
        procedencia.setdefault("modelo_escolhido", pacote["modelo_escolhido"])
    if "metrica_de_selecao" in pacote:
        procedencia.setdefault("metrica_de_selecao", pacote["metrica_de_selecao"])

    modelo = ModeloFormatoArmacao(
        estimador=estimador,
        features=features,
        conjunto_entrada=conjunto,
        alvo=str(pacote.get("alvo", "")),
        caminho=caminho,
        metricas=dict(pacote.get("metricas", {})),
        referencias=dict(pacote.get("referencias_do_acaso", {})),
        procedencia=procedencia,
    )

    if classes_esperadas is not None:
        obtidas = modelo.classes
        if obtidas != tuple(classes_esperadas):
            raise ModeloIncompativelError(
                f"O modelo em '{caminho}' preve as classes {list(obtidas)}, mas eram "
                f"esperadas {list(classes_esperadas)}.\n"
                "  Se as classes obtidas forem formatos de ROSTO (diamante, oval, "
                "quadrado, redondo), este arquivo e o modelo de rosto, nao o de armacao.\n"
                "  Se o artefato foi treinado com --excluir, carregue com "
                "classes_esperadas=None e saiba que o ranking nao cobre as classes "
                "excluidas."
            )

    return modelo
