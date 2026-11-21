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

Existem DUAS saidas serializaveis, e elas nao sao intercambiaveis:

- :meth:`Recomendacao.para_dicionario` e a saida **completa**, de auditoria:
  leva as probabilidades das 4 classes, confianca, margem e as fontes citadas.
  Serve para a monografia, para conferencia e para depuracao.
- :meth:`Recomendacao.para_contrato_backend` e a saida **para o backend**, e
  ela e deliberadamente mais pobre: nenhum numero de incerteza atravessa
  (D-66). Nao e um resumo da outra por conveniencia de payload; e uma fronteira
  com regra, conferida em tempo de execucao por :func:`_conferir_contrato`.

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
    "CHAVES_CONTRATO",
    "CHAVES_PROIBIDAS_NO_CONTRATO",
    "VERSAO_CONTRATO",
    "ArmacaoRecomendada",
    "ContratoVioladoError",
    "Recomendacao",
    "calcular_features",
    "recomendar",
]

#: Versao do **formato** do dicionario de
#: :meth:`Recomendacao.para_contrato_backend`.
#:
#: Versiona a FORMA do dicionario (quais chaves existem e de que tipo), e nada
#: mais. Em particular **nao** versiona o conteudo da tabela de visagismo: se
#: uma regra de :mod:`vizo_rosto.visagismo` mudar, o ranking gravado no banco
#: do backend muda e este numero continua 1. Quem precisar distinguir
#: recomendacoes geradas antes e depois de uma revisao da tabela nao consegue
#: fazer isso por este campo.
VERSAO_CONTRATO: int = 1

#: As chaves de primeiro nivel do contrato, exatamente estas e so estas.
#:
#: Declarado para documentacao e para a conferencia de tempo de execucao. Os
#: testes **nao** importam esta constante, de proposito: um teste que lesse
#: daqui a lista de chaves esperadas passaria a aceitar qualquer chave nova que
#: alguem acrescentasse aqui, que e justamente o que ele existe para impedir.
CHAVES_CONTRATO: frozenset[str] = frozenset(
    {"formato_rosto", "ranking_armacoes", "principio", "aviso", "versao_contrato"}
)

#: Nomes de chave que **nao podem** aparecer no contrato, em profundidade
#: nenhuma.
#:
#: As quatro primeiras sao as que D-66 proibe nominalmente. As demais estao
#: aqui porque sao os nomes pelos quais a mesma informacao tenderia a voltar
#: com outra roupa.
CHAVES_PROIBIDAS_NO_CONTRATO: frozenset[str] = frozenset(
    {
        # Proibidas nominalmente por D-66.
        "probabilidades",
        "confianca",
        "margem_1o_2o",
        "features",
        # A mesma coisa com outro nome.
        "margem",
        "probabilidade",
        "prob",
        "acuracia",
        "assertividade",
        "certeza",
        "score",
        "pontuacao",
        "percentual",
        "porcentagem",
        "peso",
    }
)


class ContratoVioladoError(AssertionError):
    """A saida para o backend carrega algo que D-66 proibe.

    Herda de ``AssertionError``, e nao de ``ValueError``, de proposito: isto
    nao e erro do dado que entrou, e sim defeito de programacao neste modulo.
    Nenhuma medicao de usuario pode provocar esta excecao; se ela aparecer em
    producao, alguem editou
    :meth:`Recomendacao.para_contrato_backend` e o guarda pegou.
    """


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

    def para_contrato_backend(self) -> dict[str, Any]:
        """Converte a recomendacao no contrato que o backend consome (D-66).

        Esta e a **unica** saida autorizada a cruzar a fronteira para o
        backend. Ela nao leva numero de incerteza nenhum: nem as
        probabilidades das 4 classes, nem a confianca, nem a margem entre a 1a
        e a 2a classe, seja do rosto ou da armacao. O formato de rosto vai como
        um valor unico e os formatos de armacao vao como ranking ordenado, que
        foi o pedido do autor.

        O preco disso fica registrado aqui porque quem ler o codigo precisa
        saber: a acuracia do modelo de rosto e 0,47 em 4 classes, ou seja o
        ``top1`` erra mais da metade das vezes, e sem a confianca o backend
        **nao tem como distinguir** uma previsao folgada de um quase-empate
        entre duas classes. Essa informacao existe e continua disponivel em
        :meth:`para_dicionario`; o que este metodo faz e deixa-la de fora por
        decisao do autor, nao por ela nao existir.

        Returns:
            Dicionario com exatamente as chaves de :data:`CHAVES_CONTRATO`, em
            tipos nativos (``str``, ``int``, ``list``, ``dict``), serializavel
            em JSON e gravavel em ``JSONField`` sem conversao:

            - ``formato_rosto`` (``str``): um dos 4 formatos de rosto;
            - ``ranking_armacoes`` (``list[dict]``): de 3 a 6 itens
              ``{"posicao": int, "formato": str}``, ordenados, com ``posicao``
              comecando em 1 e sem buraco;
            - ``principio`` (``str``): o principio visagista aplicado;
            - ``aviso`` (``str``): o ``AVISO_RASCUNHO``, que viaja junto por
              D-63 para que a condicao de rascunho nao fique so no
              codigo-fonte;
            - ``versao_contrato`` (``int``): :data:`VERSAO_CONTRATO`.

        Raises:
            ContratoVioladoError: se o dicionario montado aqui violar D-66.
                Nao acontece com o codigo como ele esta; existe para que uma
                edicao futura que reintroduza um numero de incerteza falhe na
                primeira chamada, em vez de depender de revisao de codigo.
        """
        contrato: dict[str, Any] = {
            "formato_rosto": self.previsao.formato,
            "ranking_armacoes": [
                {"posicao": int(a.posicao), "formato": a.formato} for a in self.armacoes
            ],
            "principio": self.principio,
            "aviso": self.aviso,
            "versao_contrato": VERSAO_CONTRATO,
        }
        _conferir_contrato(contrato)
        return contrato


def _conferir_contrato(contrato: Mapping[str, Any]) -> None:
    """Confere, em tempo de execucao, que o contrato respeita D-66.

    Tres regras, verificadas em toda a profundidade do dicionario:

    1. as chaves de primeiro nivel sao exatamente :data:`CHAVES_CONTRATO`;
    2. nenhuma chave, em nivel nenhum, esta em
       :data:`CHAVES_PROIBIDAS_NO_CONTRATO`;
    3. nenhum valor e ``float``. Esta e a regra que de fato segura D-66: uma
       porcentagem e um numero fracionario, e o contrato legitimo nao tem
       nenhum. Os unicos numeros que ele carrega sao ``posicao`` e
       ``versao_contrato``, ambos inteiros. Quem acrescentar a confianca com
       qualquer nome falha aqui, mesmo que o nome escape da regra 2.

    O custo e desprezivel: o contrato tem 5 chaves e no maximo 6 itens de
    ranking, contra as 500 arvores que a previsao acabou de percorrer.

    Args:
        contrato: o dicionario a conferir.

    Raises:
        ContratoVioladoError: nomeando o caminho exato do problema.
    """

    def caminhar(no: Any, caminho: str) -> None:
        if isinstance(no, Mapping):
            for chave, valor in no.items():
                onde = f"{caminho}.{chave}" if caminho else str(chave)
                if str(chave).strip().lower() in CHAVES_PROIBIDAS_NO_CONTRATO:
                    raise ContratoVioladoError(
                        f"A chave proibida '{chave}' aparece no contrato do backend, "
                        f"em '{onde}'. D-66 e explicita: nenhum numero de incerteza "
                        "cruza a fronteira, nem do rosto nem da armacao.\n"
                        "  O que fazer: se o backend precisa desse dado, ele NAO sai "
                        "por para_contrato_backend. Use para_dicionario(), que e a "
                        "saida de auditoria e leva probabilidades, confianca e "
                        "margem, e leve a necessidade ao autor antes de mudar a "
                        "fronteira."
                    )
                caminhar(valor, onde)
            return
        if isinstance(no, (list, tuple)):
            for i, item in enumerate(no):
                caminhar(item, f"{caminho}[{i}]")
            return
        # bool e subclasse de int e passaria na checagem de int; o contrato nao
        # tem booleano, entao ele cai na recusa de tipo la embaixo.
        if isinstance(no, float):
            raise ContratoVioladoError(
                f"O valor em '{caminho}' e um float ({no!r}), e o contrato do backend "
                "nao carrega numero fracionario nenhum: os unicos numeros legitimos "
                "nele sao 'posicao' e 'versao_contrato', que sao inteiros. Um float "
                "aqui e, na pratica, uma porcentagem de assertividade atravessando a "
                "fronteira, que e o que D-66 proibe.\n"
                "  O que fazer: remova o campo do contrato. Se o dado for necessario "
                "para auditoria, ele ja existe em para_dicionario()."
            )
        if type(no) not in (str, int, type(None)):
            raise ContratoVioladoError(
                f"O valor em '{caminho}' e {type(no).__name__}, que nao e um dos tipos "
                "nativos que o contrato admite. O backend grava este dicionario em "
                "JSONField e precisa de str, int, list, dict ou None.\n"
                "  O que fazer: converta para tipo nativo antes de montar o contrato. "
                "Atencao a numpy.float64 e numpy.int64, que vem do estimador e nao "
                "sao float e int do Python, e a bool, que o contrato nao usa."
            )

    # A varredura recursiva vem ANTES da conferencia do conjunto de chaves, e
    # a ordem importa para a mensagem: uma chave como 'probabilidades' no topo
    # viola as duas regras ao mesmo tempo, e o diagnostico util e "esta chave
    # e proibida por D-66, use para_dicionario()", nao o generico "o conjunto
    # de chaves nao e o acordado".
    caminhar(dict(contrato), "")

    chaves = set(map(str, contrato.keys()))
    if chaves != set(CHAVES_CONTRATO):
        faltando = sorted(set(CHAVES_CONTRATO) - chaves)
        sobrando = sorted(chaves - set(CHAVES_CONTRATO))
        raise ContratoVioladoError(
            "As chaves de primeiro nivel do contrato do backend nao sao as "
            "acordadas.\n"
            f"  Faltando ({len(faltando)}): {faltando}\n"
            f"  Inesperadas ({len(sobrando)}): {sobrando}\n"
            "  O que fazer: o contrato e acordado com o backend, nao e livre. "
            "Acrescentar chave exige combinar com o backend e subir "
            "VERSAO_CONTRATO; nao basta editar aqui."
        )


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
