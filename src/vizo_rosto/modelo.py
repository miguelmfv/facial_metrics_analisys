"""Carga e uso do modelo de formato de rosto.

Este modulo e a **primeira** etapa da cadeia: ele usa um RandomForest treinado
em 313 rostos rotulados a mao para estimar o formato de rosto. Ele nao sabe nada
sobre armacao, e nao deve passar a saber -- a segunda etapa vive em
:mod:`vizo_rosto.modelo_armacao`, e a separacao e o que permite dizer, de cada
numero, de qual modelo ele veio.

Este e o modelo que tem sinal: 0,47 contra 0,25 do acaso em 4 classes. O da
segunda etapa nao tem, e o experimento central do TCC explica por que (teste de
permutacao sobre a relacao rosto-armacao, p global = 0,176). Usa-lo assim mesmo
e decisao registrada do autor (D-79).

O modelo e **reusado congelado** (D-48): nada aqui treina, retreina ou ajusta
hiperparametro. A versao de ``scikit-learn`` tem de ser a mesma gravada no
``.joblib`` (ver ``requirements.txt``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

__all__ = [
    "CLASSES_ESPERADAS",
    "ModeloIncompativelError",
    "FeaturesIncompativeisError",
    "FeatureNaoFinitaError",
    "ModeloFormatoRosto",
    "PrevisaoRosto",
    "carregar",
]

#: As 4 classes que o modelo versionado preve, em ordem.
#:
#: Verificado no artefato: ``modelo.classes_`` e exatamente esta sequencia.
#: A verificacao e estrita de proposito. Ja aconteceu neste projeto carregar o
#: modelo de **armacao** achando que era o de rosto, e a saida daquela execucao
#: chegou a ser versionada (ver ``legado/execucoes-invalidas/``). Comparar as
#: classes na carga transforma aquele erro silencioso em erro imediato.
CLASSES_ESPERADAS: tuple[str, ...] = ("diamante", "oval", "quadrado", "redondo")


class ModeloIncompativelError(ValueError):
    """O arquivo carregado nao e o modelo de formato de rosto esperado."""


class FeaturesIncompativeisError(ValueError):
    """As features oferecidas nao sao as que o modelo espera."""


class FeatureNaoFinitaError(ValueError):
    """Alguma feature e NaN ou infinito, e por isso a medicao nao e previsivel."""


@dataclass(frozen=True)
class PrevisaoRosto:
    """Resultado da previsao de formato de rosto para **uma** medicao.

    Devolve as probabilidades das 4 classes, e nao apenas a classe vencedora.
    Isso e deliberado (D-49): a acuracia do modelo e de 0,47 em 4 classes, o que
    significa que o ``top1`` erra mais da metade das vezes. Quem consome precisa
    poder decidir o que fazer com uma previsao incerta, e para isso precisa ver
    a distribuicao inteira.

    Attributes:
        formato: a classe de maior probabilidade.
        probabilidades: probabilidade de cada uma das 4 classes, da maior para
            a menor.
    """

    formato: str
    probabilidades: dict[str, float]

    @property
    def confianca(self) -> float:
        """Probabilidade da classe prevista."""
        return self.probabilidades[self.formato]

    @property
    def margem(self) -> float:
        """Diferenca entre a 1a e a 2a classe mais provaveis.

        Margem pequena significa previsao disputada: duas classes quase
        empatadas. Util para quem consome decidir se mostra uma recomendacao
        unica ou duas alternativas.
        """
        ordenadas = sorted(self.probabilidades.values(), reverse=True)
        if len(ordenadas) < 2:
            return ordenadas[0] if ordenadas else 0.0
        return ordenadas[0] - ordenadas[1]

    def top(self, n: int = 2) -> list[str]:
        """Os ``n`` formatos mais provaveis, em ordem decrescente."""
        if n < 1:
            raise ValueError(f"n tem de ser >= 1, recebido {n}.")
        return list(self.probabilidades)[:n]


@dataclass(frozen=True)
class ModeloFormatoRosto:
    """O modelo de formato de rosto, com o contrato de features que o acompanha.

    Nao instancie diretamente: use :func:`carregar`, que faz as verificacoes de
    compatibilidade.

    Attributes:
        estimador: o ``RandomForestClassifier`` desserializado.
        features: os nomes das 29 features, **na ordem em que o modelo foi
            treinado**. A ordem importa porque o estimador consome posicao.
        alvo: o nome da coluna de rotulo usada no treino, guardado apenas como
            procedencia. Nada neste modulo exige rotulo na entrada.
        caminho: de onde o artefato foi carregado.
    """

    estimador: Any
    features: tuple[str, ...]
    alvo: str
    caminho: Path

    @property
    def classes(self) -> tuple[str, ...]:
        """As classes que o estimador preve, em ordem."""
        return tuple(str(c) for c in self.estimador.classes_)

    def validar_features(self, features: Mapping[str, float]) -> None:
        """Confere que o conjunto de features oferecido casa com o do treino.

        Raises:
            FeaturesIncompativeisError: se faltar ou sobrar feature. A mensagem
                lista as duas situacoes separadamente, porque elas tem causas
                diferentes: faltando sugere ``features_faciais.py`` mais antigo
                que o modelo, sobrando sugere o contrario.
        """
        oferecidas = set(features)
        faltando = [nome for nome in self.features if nome not in oferecidas]
        sobrando = [nome for nome in oferecidas if nome not in self.features]
        if faltando or sobrando:
            raise FeaturesIncompativeisError(
                "As features calculadas agora nao sao as que o modelo foi treinado "
                f"para receber.\n"
                f"  O modelo espera {len(self.features)} features.\n"
                f"  Faltando ({len(faltando)}): {faltando}\n"
                f"  Geradas agora mas desconhecidas pelo modelo ({len(sobrando)}): "
                f"{sobrando}\n"
                "  Use o MESMO features_faciais.py com que o .joblib foi treinado. "
                "Nao retreine o modelo para resolver isso: o artefato e congelado "
                "(D-48) e retreinar produziria numeros diferentes dos que a "
                "monografia relata."
            )

    def _validar_finitude(self, features: Mapping[str, float]) -> None:
        """Recusa a medicao se alguma feature nao for numero finito (D-56).

        Por que recusar em vez de preencher: para **uma** medicao nao existe
        mediana de coluna com que preencher, e, pior, o RandomForest **nao
        reclama** de ``NaN``. A partir do scikit-learn 1.4 um valor ausente e
        roteado para o filho com mais amostras, sem erro e sem aviso. Medido
        neste artefato: com ``h_w`` em ``NaN``, a probabilidade da classe
        prevista caiu de 0,508 para 0,431 silenciosamente. Uma resposta errada
        com cara de certa e pior que uma recusa.

        A checagem e de **nao-finito**, nao so de ``NaN``: ``inf`` tambem e
        alcancavel, porque as razoes entre larguras dividem por uma largura que
        pode ser zero, e ``inf`` passa por uma checagem de ``NaN``.

        Raises:
            FeatureNaoFinitaError: nomeando cada feature culpada e o seu valor.
        """
        culpadas = {
            nome: valor
            for nome, valor in features.items()
            if nome in set(self.features) and not math.isfinite(float(valor))
        }
        if culpadas:
            detalhe = ", ".join(f"{nome}={valor}" for nome, valor in sorted(culpadas.items()))
            raise FeatureNaoFinitaError(
                f"{len(culpadas)} de {len(self.features)} features nao sao numeros "
                f"finitos, e por isso esta medicao nao e previsivel: {detalhe}.\n"
                "  Causa tipica: a largura do contorno nao pode ser medida em alguma "
                "altura do rosto, o que acontece quando o contorno esta aberto, "
                "autointersectado ou com pontos repetidos.\n"
                "  O que fazer: descarte esta medicao e peca uma nova captura. Nao "
                "preencha o valor faltante com media ou mediana para uma medicao "
                "unica: o modelo aceitaria o NaN sem reclamar e devolveria uma "
                "probabilidade diferente da verdadeira, sem nenhum aviso."
            )

    def prever(self, features: Mapping[str, float]) -> PrevisaoRosto:
        """Preve o formato de rosto de **uma** medicao.

        Args:
            features: as 29 features, por nome. Chaves em qualquer ordem: a
                ordem do treino e imposta aqui dentro.

        Returns:
            A previsao, com a probabilidade das 4 classes.

        Raises:
            FeaturesIncompativeisError: se o conjunto de features nao casar.
            FeatureNaoFinitaError: se alguma feature for ``NaN`` ou infinito.
        """
        self.validar_features(features)
        self._validar_finitude(features)

        # DataFrame com uma linha, colunas na ordem do treino. Usar DataFrame e
        # nao um array cru nao e preferencia de estilo: o estimador foi
        # ajustado com nomes de coluna, entao receber um array sem nomes faz o
        # sklearn emitir aviso de feature names.
        X = pd.DataFrame([[float(features[nome]) for nome in self.features]], columns=list(self.features))
        probabilidades = self.estimador.predict_proba(X)[0]
        classes = self.classes

        pares = sorted(
            ((classe, float(p)) for classe, p in zip(classes, probabilidades)),
            key=lambda par: par[1],
            reverse=True,
        )
        return PrevisaoRosto(formato=pares[0][0], probabilidades=dict(pares))


def carregar(
    caminho: str | Path,
    classes_esperadas: Sequence[str] | None = CLASSES_ESPERADAS,
) -> ModeloFormatoRosto:
    """Carrega o modelo de formato de rosto de um arquivo ``.joblib``.

    Args:
        caminho: caminho do ``.joblib``. Nenhum default embutido de proposito:
            caminho absoluto ou relativo chumbado no codigo foi um problema real
            neste repositorio (o ``--saida`` de ``modelo_formato_rosto.py`` tem
            default ``/resultados/...``, com barra inicial, que no Windows
            resolve na raiz do drive).
        classes_esperadas: as classes que o modelo deve prever. O default
            protege contra carregar o modelo de armacao por engano. Passe
            ``None`` para desligar a verificacao, ciente do risco.

    Returns:
        O modelo, com o contrato de features que foi salvo junto dele.

    Raises:
        FileNotFoundError: se o arquivo nao existir.
        ModeloIncompativelError: se o arquivo nao tiver a estrutura esperada ou
            se as classes nao forem as esperadas.
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Nao encontrei o modelo em '{caminho}'. O artefato versionado deste "
            "projeto fica em "
            "modelos/modelo_formato_rosto_sem_linhas_branco.joblib, relativo a raiz "
            "do repositorio. Confira o diretorio de trabalho."
        )

    pacote = joblib.load(caminho)

    if not isinstance(pacote, Mapping):
        raise ModeloIncompativelError(
            f"O arquivo '{caminho}' desserializou para {type(pacote).__name__}, mas o "
            "formato esperado e um dicionario com as chaves 'modelo', 'features' e "
            "'alvo'. Se este arquivo e um estimador salvo direto, ele nao carrega a "
            "lista de features, e sem ela nao ha como garantir a ordem das colunas."
        )

    faltando = [chave for chave in ("modelo", "features") if chave not in pacote]
    if faltando:
        raise ModeloIncompativelError(
            f"O arquivo '{caminho}' nao tem as chaves {faltando}. Chaves presentes: "
            f"{sorted(pacote.keys())}. Esperado um dicionario salvo como "
            "{'modelo': estimador, 'features': [nomes], 'alvo': coluna}."
        )

    estimador = pacote["modelo"]
    if not hasattr(estimador, "predict_proba"):
        raise ModeloIncompativelError(
            f"O objeto em ['modelo'] de '{caminho}' e um "
            f"{type(estimador).__name__}, que nao tem predict_proba. A recomendacao "
            "precisa das probabilidades das 4 classes, nao so da classe vencedora "
            "(D-49)."
        )

    features = tuple(str(nome) for nome in pacote["features"])
    modelo = ModeloFormatoRosto(
        estimador=estimador,
        features=features,
        alvo=str(pacote.get("alvo", "")),
        caminho=caminho,
    )

    if classes_esperadas is not None:
        obtidas = modelo.classes
        if obtidas != tuple(classes_esperadas):
            raise ModeloIncompativelError(
                f"O modelo em '{caminho}' preve as classes {list(obtidas)}, mas eram "
                f"esperadas {list(classes_esperadas)}.\n"
                "  Se as classes obtidas forem formatos de ARMACAO (retangular, "
                "ovalado, gatinho, aviador, coracao), este arquivo e o modelo de "
                "armacao, nao o de rosto. Isso ja aconteceu neste projeto e a saida "
                "da execucao errada chegou a ser versionada; ver "
                "legado/execucoes-invalidas/README.md.\n"
                "  O que fazer: aponte para "
                "modelos/modelo_formato_rosto_sem_linhas_branco.joblib."
            )

    n_esperado = getattr(estimador, "n_features_in_", None)
    if n_esperado is not None and n_esperado != len(features):
        raise ModeloIncompativelError(
            f"Inconsistencia dentro do proprio artefato '{caminho}': o estimador foi "
            f"ajustado com {n_esperado} features, mas a lista salva em ['features'] "
            f"tem {len(features)} nomes. O arquivo esta corrompido ou foi montado a "
            "mao; nao use."
        )

    return modelo
