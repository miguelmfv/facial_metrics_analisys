"""Orquestracao: medicao -> features -> formato de rosto -> formatos de armacao.

Este modulo e a camada que junta as outras. Ele nao calcula geometria (isso e
:mod:`vizo_rosto.features_faciais`) e nao sabe desserializar modelo (isso e
:mod:`vizo_rosto.modelo` e :mod:`vizo_rosto.modelo_armacao`).

**A cadeia, desde D-79** (22/11/2026), e inteiramente aprendida::

    medicao -> 29 features -> classificador de ROSTO  -> formato de rosto + p_rosto_*
                           -> classificador de ARMACAO -> formatos de armacao ranqueados

E o desenho do pipeline original (``legado/pipeline_rosto_armacao.py``), com a
segunda etapa versionada em artefato em vez de descartada ao fim da validacao.

**O que mudou, e por que isso importa para quem le a saida.** Ate a rodada 3 a
segunda etapa era *prescritiva*: uma tabela de visagismo declarada, com fonte
citada, escolhida justamente porque o experimento central do trabalho nao
encontrou relacao estatisticamente detectavel entre formato de rosto e formato
de armacao (permutacao, p global = 0,176). Por decisao do autor (D-79) o
visagismo saiu do caminho de execucao e a recomendacao voltou a ser aprendida.
:mod:`vizo_rosto.visagismo` continua no pacote como registro historico -- e o
que a monografia cita e o que a rodada 3 documentou --, mas nada aqui o importa.

A consequencia esta medida e viaja junto com a recomendacao: o classificador de
armacao preve no nivel do acaso, e o ranking de 3 formatos que ele produz nao
acerta mais, em top-3, que uma lista fixa das 3 classes mais frequentes do
dataset. :func:`montar_aviso` le esses numeros do proprio artefato e os escreve
no campo ``aviso`` do contrato, para que a ressalva nao dependa de ninguem
lembrar dela.

O que esta saida **nao** faz: escolher armacoes concretas do catalogo. O
catalogo vive no banco do backend, que este repositorio nao acessa (D-49).

Existem DUAS saidas serializaveis, e elas nao sao intercambiaveis:

- :meth:`Recomendacao.para_dicionario` e a saida **completa**, de auditoria:
  leva as probabilidades das duas etapas, confianca, margem e as metricas
  medidas do classificador de armacao. Serve para a monografia e para
  depuracao.
- :meth:`Recomendacao.para_contrato_backend` e a saida **para o backend**, e
  ela e deliberadamente mais pobre: nenhum numero de incerteza atravessa
  (D-66, mantida por D-83). Nao e um resumo da outra por conveniencia de
  payload; e uma fronteira com regra, conferida em tempo de execucao por
  :func:`_conferir_contrato`.

Nenhuma funcao deste modulo imprime: todas devolvem dados.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .entrada import CONTORNOS_EXIGIDOS, EntradaInvalidaError, achatar_medicao, linha_de_csv
from .features_faciais import extract
from .modelo import ModeloFormatoRosto, PrevisaoRosto
from .modelo_armacao import FormatoArmacao, ModeloFormatoArmacao, PrevisaoArmacao

__all__ = [
    "CHAVES_CONTRATO",
    "CHAVES_PROIBIDAS_NO_CONTRATO",
    "N_RECOMENDADOS_PADRAO",
    "PRINCIPIO",
    "VERSAO_CONTRATO",
    "ContratoVioladoError",
    "ModeloDeArmacaoAusenteError",
    "Recomendacao",
    "calcular_features",
    "montar_aviso",
    "recomendar",
]

#: Versao do **formato** do dicionario de
#: :meth:`Recomendacao.para_contrato_backend`.
#:
#: Subiu de 1 para 2 em D-82. As chaves e os tipos sao os mesmos da versao 1 --
#: o backend nao precisa de migracao nem de campo novo --, mas a **procedencia**
#: do conteudo mudou: ``ranking_armacoes`` passou a ser produzido por um
#: classificador treinado, nao por uma tabela de visagismo declarada, e
#: ``principio`` e ``aviso`` dizem outra coisa. Uma recomendacao gravada com
#: versao 1 e uma com versao 2 nao sao comparaveis, e este numero e o unico
#: jeito de distingui-las depois de gravadas.
#:
#: **O backend recusa o que nao conhece**, de proposito: ele tem
#: ``VERSAO_CONTRATO_ACEITA`` em ``vizo_app/recommendationEngine.py``, hoje em
#: 1. Enquanto essa constante nao subir para 2, toda recomendacao gerada por
#: este pacote e recusada na gravacao -- falha alta, nao silenciosa.
VERSAO_CONTRATO: int = 2

#: Quantos formatos de armacao o contrato leva (D-84).
#:
#: O classificador ranqueia todas as classes; o contrato leva as primeiras. Tres
#: e o numero que mantem util o bloco de "nao combinam" do app, que e calculado
#: como o complemento do ranking recebido. E parametro de
#: :func:`recomendar`, nao constante chumbada.
N_RECOMENDADOS_PADRAO: int = 3

#: O texto do campo ``principio`` do contrato.
#:
#: Na versao 1 ele trazia o principio visagista do contraste geometrico. Agora
#: declara o que de fato produz o ranking, que e outra coisa: um criterio
#: estatistico aprendido. Dizer "contraste geometrico" aqui seria descrever um
#: criterio que o codigo nao aplica mais.
PRINCIPIO: str = (
    "Os formatos de armação são ordenados por um classificador treinado em fotos "
    "de catálogo, que recebe as medidas do rosto e a estimativa de formato de "
    "rosto da etapa anterior. O critério é estatístico, aprendido dos dados — não "
    "é uma regra de visagismo nem uma escolha estética declarada."
)


class ContratoVioladoError(AssertionError):
    """A saida para o backend carrega algo que D-66 proibe.

    Herda de ``AssertionError``, e nao de ``ValueError``, de proposito: isto
    nao e erro do dado que entrou, e sim defeito de programacao neste modulo.
    Nenhuma medicao de usuario pode provocar esta excecao.
    """


class ModeloDeArmacaoAusenteError(ValueError):
    """Chamaram :func:`recomendar` sem o classificador de armacao.

    Desde D-79 a cadeia tem duas etapas e a segunda nao e opcional. A excecao
    existe para que um chamador escrito para a assinatura antiga falhe alto, com
    o nome do que falta, em vez de devolver meia recomendacao.
    """


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
#: com outra roupa. A lista ganhou peso com D-79: agora existem probabilidades
#: dos **dois** classificadores querendo atravessar, nao so as do rosto.
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


def _decimal(valor: float, casas: int = 2) -> str:
    """Formata um numero no padrao pt-BR, com virgula decimal."""
    return f"{valor:.{casas}f}".replace(".", ",")


def montar_aviso(modelo_armacao: ModeloFormatoArmacao) -> str:
    """Monta o aviso que viaja com a recomendacao, a partir do artefato em uso.

    O texto e montado, e nao constante, por um motivo concreto: os numeros que
    ele cita sao as metricas **daquele** ``.joblib``. Um artefato retreinado com
    outro recorte produz outros numeros, e um aviso constante continuaria
    afirmando os antigos. Aqui ele acompanha o artefato sozinho.

    O aviso e escrito em portugues **com acentuacao**. A constante anterior
    (``visagismo.AVISO_RASCUNHO``) era ASCII puro, e o texto chegava a tela do
    app sem acento nenhum -- achado 5.2 do fechamento da rodada 3, tarefa 3. Como
    ele e gravado no banco pelo backend (D-63), o conserto so vale para
    recomendacoes novas.

    Args:
        modelo_armacao: o classificador carregado, de onde saem as metricas.

    Returns:
        O texto do campo ``aviso`` do contrato.
    """
    partes = [
        "Recomendação gerada por dois classificadores em sequência: o primeiro "
        "estima o formato do rosto a partir das medidas faciais, o segundo ordena "
        "os formatos de armação."
    ]

    medidas = modelo_armacao.metricas_do_escolhido
    acuracia = medidas.get("acuracia")
    top3 = medidas.get("top3")
    majoritaria = modelo_armacao.referencias.get("acuracia_classe_majoritaria")
    fixo = modelo_armacao.referencias.get("top3_ranking_fixo_das_3_mais_frequentes")

    if acuracia is not None and majoritaria is not None:
        partes.append(
            f"Acurácia medida do classificador de armação: {_decimal(acuracia)} contra "
            f"{_decimal(majoritaria)} de simplesmente responder sempre o formato mais "
            "comum do catálogo."
        )
    if top3 is not None and fixo is not None:
        comparacao = "contra" if top3 <= fixo else "acima dos"
        partes.append(
            f"O acerto dentro de uma lista de três formatos é {_decimal(top3)}, "
            f"{comparacao} {_decimal(fixo)} de uma lista fixa das três armações mais "
            "frequentes, igual para todo rosto."
        )

    partes.append(
        "O experimento deste trabalho não encontrou relação estatisticamente "
        "detectável entre formato de rosto e formato de armação (teste de "
        "permutação, p = 0,176). Trate esta lista como sugestão, não como medida."
    )
    return " ".join(partes)


@dataclass(frozen=True)
class Recomendacao:
    """A saida completa para **uma** medicao.

    As duas partes sao agora da **mesma** natureza -- descritiva, aprendida --,
    e isso e o que D-79 mudou. A saida continua separando-as campo a campo
    porque sao dois modelos diferentes, com desempenhos muito diferentes: o de
    rosto tem sinal real (0,47 contra 0,25 do acaso em 4 classes), o de armacao
    nao tem.

    Attributes:
        previsao: o formato de rosto previsto e a probabilidade das 4 classes.
        previsao_armacao: os formatos de armacao ranqueados, com as
            probabilidades do classificador.
        n_recomendados: quantos formatos o contrato leva (D-84).
        principio: o texto que declara como o ranking foi produzido.
        aviso: a ressalva com os numeros medidos, montada por
            :func:`montar_aviso`. Viaja junto com a recomendacao de proposito
            (D-63).
        features: as 29 features calculadas, para auditoria.
    """

    previsao: PrevisaoRosto
    previsao_armacao: PrevisaoArmacao
    principio: str
    aviso: str
    features: dict[str, float]
    n_recomendados: int = N_RECOMENDADOS_PADRAO

    @property
    def formato_rosto(self) -> str:
        """Atalho para o formato de rosto previsto."""
        return self.previsao.formato

    @property
    def armacoes(self) -> tuple[FormatoArmacao, ...]:
        """Os formatos de armacao que vao para o contrato: os ``n`` primeiros."""
        return self.previsao_armacao.top(self.n_recomendados)

    @property
    def formato_armacao(self) -> str:
        """O primeiro colocado do ranking."""
        return self.previsao_armacao.formato

    def para_dicionario(self, incluir_features: bool = False) -> dict[str, Any]:
        """Converte a recomendacao em tipos nativos, serializaveis em JSON.

        Esta e a saida de **auditoria**: ela leva os numeros de incerteza das
        duas etapas, e por isso nao e a que vai para o backend. O ranking aqui
        vem **inteiro**, nao cortado em ``n_recomendados``: quem audita precisa
        ver tambem o que ficou de fora e por qual diferenca.

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
                "natureza": "descritivo (classificador treinado em fotos de catalogo)",
                "conjunto_entrada": self.previsao_armacao.conjunto_entrada,
                "principio": self.principio,
                "n_no_contrato": int(self.n_recomendados),
                "ranking": [
                    {
                        "posicao": item.posicao,
                        "formato": item.formato,
                        "probabilidade": item.probabilidade,
                        "no_contrato": item.posicao <= self.n_recomendados,
                    }
                    for item in self.previsao_armacao.ranking
                ],
                "confianca": self.previsao_armacao.confianca,
                "margem_1o_2o": self.previsao_armacao.margem,
            },
            "aviso": self.aviso,
        }
        if incluir_features:
            saida["features"] = dict(self.features)
        return saida

    def para_contrato_backend(self) -> dict[str, Any]:
        """Converte a recomendacao no contrato que o backend consome (D-66, D-82).

        Esta e a **unica** saida autorizada a cruzar a fronteira para o
        backend. Ela nao leva numero de incerteza nenhum: nem as probabilidades
        dos formatos de rosto, nem as dos formatos de armacao, nem confianca,
        nem margem. O formato de rosto vai como um valor unico e os formatos de
        armacao vao como ranking ordenado.

        O preco disso fica registrado aqui porque quem ler o codigo precisa
        saber: as duas etapas erram muito -- o rosto acerta 0,47 em 4 classes e
        a armacao fica no nivel do acaso --, e sem os numeros o backend **nao
        tem como distinguir** uma previsao folgada de um empate tecnico. Essa
        informacao existe e continua em :meth:`para_dicionario`; o que este
        metodo faz e deixa-la de fora por decisao do autor. O campo ``aviso``
        carrega a ressalva em texto, que e o que de fato chega ao usuario.

        Returns:
            Dicionario com exatamente as chaves de :data:`CHAVES_CONTRATO`, em
            tipos nativos, serializavel em JSON e gravavel em ``JSONField`` sem
            conversao:

            - ``formato_rosto`` (``str``): um dos 4 formatos de rosto;
            - ``ranking_armacoes`` (``list[dict]``): ``n_recomendados`` itens
              ``{"posicao": int, "formato": str}``, ordenados, com ``posicao``
              comecando em 1 e sem buraco;
            - ``principio`` (``str``): :data:`PRINCIPIO`;
            - ``aviso`` (``str``): o texto de :func:`montar_aviso`, com os
              numeros medidos do artefato em uso;
            - ``versao_contrato`` (``int``): :data:`VERSAO_CONTRATO`, agora 2.

        Raises:
            ContratoVioladoError: se o dicionario montado aqui violar D-66.
        """
        contrato: dict[str, Any] = {
            "formato_rosto": self.previsao.formato,
            "ranking_armacoes": [
                {"posicao": int(item.posicao), "formato": item.formato}
                for item in self.armacoes
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

    A regra 3 passou a valer o dobro com D-79: a segunda etapa agora tambem
    produz probabilidades, e sao elas, nao as do rosto, as mais tentadoras de
    exibir como "% de compatibilidade" -- justamente o numero que o trabalho
    nao tem como sustentar.

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
    modelo_armacao: ModeloFormatoArmacao | None = None,
    n_recomendados: int = N_RECOMENDADOS_PADRAO,
) -> Recomendacao:
    """Gera a recomendacao completa para **uma** medicao (D-79, D-80).

    Roda os dois classificadores em sequencia: o de rosto primeiro, e o de
    armacao em seguida, recebendo as features e as probabilidades que o
    primeiro produziu.

    Args:
        medicao: JSON de medicao do backend, dict de contornos, ou linha
            achatada de CSV. **Sem rotulo**: nenhum campo de classe e lido.
        modelo: o classificador de formato de rosto, carregado por
            :func:`vizo_rosto.modelo.carregar`.
        modelo_armacao: o classificador de formato de armacao, carregado por
            :func:`vizo_rosto.modelo_armacao.carregar_armacao`. **Obrigatorio
            na pratica**: e parametro opcional na assinatura apenas para que um
            chamador escrito para a cadeia antiga falhe com uma mensagem que
            diz o que fazer, em vez de ``TypeError``.
        n_recomendados: quantos formatos o contrato leva (D-84). O ranking
            completo continua em ``previsao_armacao``.

    Returns:
        A recomendacao, com as duas previsoes e o aviso montado do artefato.

    Raises:
        ModeloDeArmacaoAusenteError: se ``modelo_armacao`` nao for passado.
        EntradaInvalidaError: medicao malformada, contorno faltando ou com
            contagem de pontos errada.
        FeaturesIncompativeisError: o ``features_faciais.py`` atual nao gera o
            que algum dos dois modelos espera.
        FeatureNaoFinitaError: alguma feature e ``NaN`` ou infinito, e portanto
            a medicao nao e previsivel (D-56).
    """
    if modelo_armacao is None:
        raise ModeloDeArmacaoAusenteError(
            "recomendar() precisa do classificador de formato de armacao, e ele nao "
            "foi passado.\n"
            "  Desde D-79 a recomendacao e produzida por dois modelos em sequencia; a "
            "tabela de visagismo saiu do caminho de execucao e nao ha mais de onde "
            "tirar o ranking sem o segundo artefato.\n"
            "  O que fazer: carregue-o com "
            "vizo_rosto.carregar_armacao('modelos/modelo_formato_armacao.joblib') e "
            "passe-o como terceiro argumento. Quem chama a partir do backend precisa "
            "tambem de uma setting com o caminho desse artefato, ao lado de "
            "VIZO_ROSTO_MODELO_PATH."
        )
    if n_recomendados < 1:
        raise ValueError(f"n_recomendados tem de ser >= 1, recebido {n_recomendados}.")

    features = calcular_features(medicao)
    previsao = modelo.prever(features)
    previsao_armacao = modelo_armacao.ranquear(features, previsao)

    return Recomendacao(
        previsao=previsao,
        previsao_armacao=previsao_armacao,
        principio=PRINCIPIO,
        aviso=montar_aviso(modelo_armacao),
        features=features,
        n_recomendados=min(n_recomendados, len(previsao_armacao.ranking)),
    )
