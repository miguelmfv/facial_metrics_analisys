"""Tabela PRESCRITIVA de visagismo: formato de rosto -> formatos de armacao.

=============================================================================
AVISO: RASCUNHO NAO REVISADO PELO AUTOR
=============================================================================

Este arquivo e um **rascunho** montado pelo Sextante a partir das fontes que
constam em :data:`FONTES`, e **nao foi revisado nem validado pelo autor do
TCC**. Nenhuma regra aqui deve ser apresentada como resultado do trabalho
enquanto essa revisao nao acontecer. Este arquivo foi feito para ser facil de
editar: toda regra esta em :data:`REGRAS`, cada uma com a sua fonte ao lado, e
mudar uma regra e mudar uma linha.

O QUE ESTA TABELA E, E O QUE ELA NAO E
--------------------------------------

Ela e **prescritiva**: afirma o que a literatura de visagismo *recomenda*.
Ela **nao** e descritiva: nao afirma, e nao pode afirmar, o que as pessoas de
fato usam. Esta distincao e o eixo metodologico do TCC e nao e um detalhe de
redacao.

O experimento central deste trabalho testou se existe relacao entre o formato
de rosto previsto e o formato de armacao que a pessoa usa, e **nao encontrou
nenhuma** (teste de permutacao sobre 2.784 imagens agrupadas por modelo de
armacao, p global = 0,176; acrescentar as probabilidades de rosto ao modelo de
armacao da ganho nulo, dentro do desvio-padrao dos folds). Ou seja: a hipotese
de que as pessoas **ja usam** a armacao que o visagismo recomenda esta
refutada.

Isso nao invalida a regra prescritiva, porque uma regra prescritiva nao e uma
previsao sobre o comportamento observado. Mas tem uma consequencia que precisa
estar escrita: **nada nesta tabela foi aprendido dos dados, e nada nela pode
ser.** Derivar estas regras dos dados e exatamente o que o resultado negativo
impede. Por isso ela e escrita a mao, com fonte, e nao treinada.

O ESTADO REAL DAS FONTES, SEM EMBELEZAMENTO
-------------------------------------------

Esta e a parte que mais importa para a honestidade do trabalho:

1. **Nao existe tabela canonica.** Nao encontrei nenhuma fonte revisada por
   pares, nem norma tecnica, nem obra de referencia acessivel que publique uma
   tabela formato-de-rosto x formato-de-armacao. O que existe sao guias de
   optometristas e de lojas de otica, que **se apresentam como orientacao, nao
   como regra**. Duas das fontes que consultei dizem isso com as proprias
   palavras: "these tips are meant to be helpful, they are not hard and fast
   rules" (F1) e "the suggestions in this guide aren't hard and fast rules"
   (F2).

2. **Philip Hallawell, a referencia brasileira de visagismo, NAO foi
   consultado.** Nao tive acesso a nenhuma obra dele. Toda regra abaixo esta
   marcada em funcao disso, e **nenhuma** regra e atribuida a ele. Se o autor
   tiver acesso a obra, e por ela que esta tabela deve ser refeita. Buscas
   indicaram divergencia entre fontes de terceiros sobre o proprio sistema dele
   (quantos formatos de rosto, de que ano), o que e mais um motivo para nao
   citar de segunda mao.

3. **Duas fontes que eu tentei ler nao abriram** (HTTP 403): o guia da Lenscope
   e o da UnitedHealthcare. Elas **nao** sustentam nada aqui, e nao estao em
   :data:`FONTES`.

4. **A fonte mais credivel nao cobre o rosto diamante.** F1, de optometrista,
   trata quadrado, redondo, coracao, triangular e oval, e **nao menciona
   diamante**. Como diamante e uma das 4 classes que o modelo preve, a regra
   dele se apoia em fonte comercial e em buscas, o que a torna a mais fraca da
   tabela. Esta marcada como tal.

5. **A fonte comercial se contradiz internamente** para o rosto diamante: o
   corpo do texto recomenda oval, sem aro e browline; a tabela-resumo da mesma
   pagina diz oval, sem aro e gatinho; e os links de produto da mesma pagina
   dizem retangular, oval e horn. Registro a divergencia em vez de escolher uma
   das tres em silencio.

LIMITACOES ESTRUTURAIS, DECLARADAS
----------------------------------

- **O modelo preve 4 formatos de rosto; a literatura usa 6 ou 7.** Coracao,
  triangular e oblongo/retangular aparecem nas fontes e **nao** estao mapeados
  aqui, porque o modelo nao os preve. Ver :data:`FORMATOS_ROSTO_FORA_DO_MODELO`.
- **As fontes recomendam atributos que o catalogo nao modela.** Para diamante,
  o termo que mais aparece e "sem aro" (rimless), e tambem "browline" e "horn".
  Nenhum dos tres e um dos 7 formatos do dataset: sao caracteristicas de
  construcao da armacao, nao formato de lente. A recomendacao melhor fundamentada
  para diamante e, portanto, **inexpressavel** no vocabulario disponivel.
- **O formato de armacao ``coracao`` nao tem regra.** Nao encontrei nenhuma
  fonte que recomende armacao em formato de coracao para qualquer formato de
  rosto. As fontes falam de *rosto* em formato de coracao, que e outra coisa.
  Ver :data:`FORMATOS_ARMACAO_SEM_REGRA`. No dataset, ``coracao`` tem so 3
  modelos distintos e e normalmente excluida das analises.
- As fontes falam de proporcao e ajuste (largura da armacao em relacao as macas
  do rosto, altura em relacao a linha das sobrancelhas) que **nao** se reduzem a
  formato. Isso nao esta modelado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "AVISO_RASCUNHO",
    "FONTES",
    "REGRAS",
    "FORMATOS_ROSTO",
    "FORMATOS_ARMACAO",
    "FORMATOS_ARMACAO_SEM_REGRA",
    "FORMATOS_ROSTO_FORA_DO_MODELO",
    "Fonte",
    "RegraVisagismo",
    "FormatoRostoSemRegraError",
    "ranquear",
    "principio_geral",
]

#: Aviso curto, para quem consome a saida exibir ao usuario final. Existe para
#: que a condicao de rascunho viaje junto com a recomendacao, em vez de ficar
#: so no codigo-fonte.
AVISO_RASCUNHO: str = (
    "Recomendacao baseada em regras de visagismo declaradas, nao aprendidas dos "
    "dados deste projeto. A tabela de regras e um rascunho ainda nao revisado pelo "
    "autor, construida a partir de guias de optometria e de otica que se declaram "
    "orientacao e nao regra. Este trabalho nao encontrou relacao estatisticamente "
    "detectavel entre formato de rosto e a armacao que as pessoas usam (p = 0,176), "
    "portanto esta sugestao e um criterio estetico declarado, nao uma previsao."
)

#: Os 4 formatos de rosto que o modelo preve. Verificado no artefato.
FORMATOS_ROSTO: tuple[str, ...] = ("diamante", "oval", "quadrado", "redondo")

#: Os 7 formatos de armacao que o dataset conhece.
FORMATOS_ARMACAO: tuple[str, ...] = (
    "quadrado",
    "retangular",
    "redondo",
    "ovalado",
    "gatinho",
    "aviador",
    "coracao",
)

#: Formatos de armacao que ficaram **sem regra**, com o motivo.
FORMATOS_ARMACAO_SEM_REGRA: dict[str, str] = {
    "coracao": (
        "Nenhuma fonte consultada recomenda armacao em formato de coracao para "
        "qualquer formato de rosto. As fontes tratam de ROSTO em formato de coracao, "
        "que e outra coisa. Sem fonte, nao entra na tabela (D-44). No dataset esta "
        "classe tem apenas 3 modelos distintos e e normalmente excluida das analises."
    ),
}

#: Formatos de rosto que a literatura trata e que **este modelo nao preve**.
#: Registrado como limitacao, conforme D-44.
FORMATOS_ROSTO_FORA_DO_MODELO: dict[str, str] = {
    "coracao": "Coberto por F1 e F2, mas o modelo nao preve esta classe.",
    "triangular": "Coberto por F1 e F2, mas o modelo nao preve esta classe.",
    "oblongo/retangular": (
        "Aparece na tabela-resumo de F2 ('Rectangle'), mas o modelo nao preve esta "
        "classe. F2 e inconsistente aqui: a tabela lista o formato e o corpo do texto "
        "nao tem secao para ele."
    ),
}


@dataclass(frozen=True)
class Fonte:
    """Uma fonte consultada, com o que ela e e o que ela nao e.

    Attributes:
        id: identificador curto usado nas regras.
        titulo: titulo da pagina ou obra.
        autoria: autor ou organizacao responsavel.
        url: endereco consultado, quando aplicavel.
        consultado_em: data da consulta, em ISO.
        natureza: que tipo de fonte e. Determina o peso que a regra merece.
        ressalvas: o que limita o uso desta fonte, nas observacoes de quem
            consultou.
    """

    id: str
    titulo: str
    autoria: str
    url: str
    consultado_em: str
    natureza: str
    ressalvas: str


#: As fontes **efetivamente consultadas**. Cada pagina listada aqui foi lida; o
#: que nao foi lido nao esta aqui.
FONTES: dict[str, Fonte] = {
    "F1": Fonte(
        id="F1",
        titulo="Choosing Glasses for Your Face Shape",
        autoria="Dr. Russel Lazarus - Optometrists Network (optometrists.org)",
        url=(
            "https://www.optometrists.org/general-practice-optometry/optical/"
            "guide-to-optical-frames/9115-2/"
        ),
        consultado_em="2026-11-18",
        natureza=(
            "Guia de rede de optometristas, assinado por optometrista. E a fonte mais "
            "credivel que consegui ler, e ainda assim nao cita estudo nem orgao "
            "profissional: a orientacao e asseverada pelo autor."
        ),
        ressalvas=(
            "Nao cobre o rosto DIAMANTE, que nem e mencionado na pagina. A propria "
            "pagina diz: 'While these tips are meant to be helpful, they are not hard "
            "and fast rules.' Nao foi encontrada data de publicacao."
        ),
    ),
    "F2": Fonte(
        id="F2",
        titulo="How to Pick Glasses for Your Face Shape",
        autoria="Eyebuydirect (eyebuydirect.com)",
        url="https://www.eyebuydirect.com/guides/frames-and-face-shapes",
        consultado_em="2026-11-18",
        natureza=(
            "Guia de marketing de loja online de oculos. Conteudo COMERCIAL: a pagina "
            "tem vitrine, promocao e links para as proprias categorias de produto. "
            "Peso menor que F1 por esse motivo."
        ),
        ressalvas=(
            "Nao cita nenhum estudo, obra, autor ou orgao profissional; sem assinatura "
            "e sem data. Diz de si mesma que as sugestoes 'aren't hard and fast rules' "
            "e que os formatos sao 'just guidelines'. E INTERNAMENTE INCONSISTENTE para "
            "o rosto diamante: corpo do texto diz oval/sem aro/browline, a tabela-resumo "
            "diz oval/sem aro/gatinho, e os links de produto dizem retangular/oval/horn."
        ),
    ),
    "F3": Fonte(
        id="F3",
        titulo=(
            "Agregado de resultados de busca em guias brasileiros de otica "
            "(paginas nao lidas individualmente)"
        ),
        autoria="Diversos guias comerciais de otica e de fabricantes, em portugues",
        url="",
        consultado_em="2026-11-18",
        natureza=(
            "FONTE FRACA, e esta aqui so para registrar DIVERGENCIA, nunca para "
            "sustentar sozinha uma regra. Vem de resultado de busca agregado; as "
            "paginas individuais nao foram abertas e verificadas uma a uma."
        ),
        ressalvas=(
            "Nao use para afirmar nada. Serve para documentar que as fontes brasileiras "
            "divergem entre si sobre o rosto diamante: ha quem indique ovalada ou sem "
            "aro, quem indique quadrada ou oval com aro cheio, e quem indique gatinho e "
            "aviador."
        ),
    ),
}

#: Fontes que eu **tentei** consultar e nao consegui. Registradas para que
#: ninguem as suponha consultadas, e para que a proxima tentativa saiba o que
#: ja falhou.
FONTES_NAO_ACESSIVEIS: dict[str, str] = {
    "Philip Hallawell (obra)": (
        "NAO CONSULTADO. Nenhuma obra de Hallawell foi acessada. Ele e a referencia "
        "brasileira de visagismo e por isso nenhuma regra desta tabela e atribuida a "
        "ele. Se o autor tiver acesso a obra, a tabela deve ser refeita por ela."
    ),
    "Lenscope - Visagismo: tudo sobre essa tecnica": (
        "HTTP 403 na tentativa de leitura em 2026-11-18. Nao sustenta nada aqui."
    ),
    "UnitedHealthcare - How to find the best pair of glasses based on face shape": (
        "HTTP 403 na tentativa de leitura em 2026-11-18. Nao sustenta nada aqui."
    ),
}


@dataclass(frozen=True)
class RegraVisagismo:
    """Uma recomendacao prescritiva, com a sua procedencia colada nela.

    Attributes:
        formato_rosto: o formato de rosto a que a regra se aplica.
        formato_armacao: o formato de armacao recomendado. Sempre um dos 7 de
            :data:`FORMATOS_ARMACAO`.
        posicao: posicao no ranking, comecando em 1.
        justificativa: o criterio visagista, em portugues.
        fontes: ids de :data:`FONTES` que sustentam a regra.
        sustentacao: quao bem fundamentada a regra esta. Um de
            ``"convergente"`` (as duas fontes lidas concordam),
            ``"fonte-unica"``, ``"divergente"`` (as fontes discordam entre si)
            ou ``"nao-fundamentada"``.
        observacao: ressalva especifica desta regra, quando houver.
    """

    formato_rosto: str
    formato_armacao: str
    posicao: int
    justificativa: str
    fontes: tuple[str, ...]
    sustentacao: str
    observacao: str = ""


#: O principio que as fontes aplicam, declarado uma vez para nao se repetir em
#: cada regra.
_PRINCIPIO = (
    "Contraste geometrico: recomenda-se a armacao cuja forma CONTRASTA com a forma "
    "do rosto, para equilibrar o conjunto. Rosto anguloso pede armacao curva, rosto "
    "curvo pede armacao angulosa. E o criterio comum a F1 e F2, e e tambem o unico "
    "ponto em que todas as fontes consultadas concordam."
)


def principio_geral() -> str:
    """Devolve o principio geral que as fontes aplicam."""
    return _PRINCIPIO


# ---------------------------------------------------------------------------
# A TABELA. Uma regra por linha, com a fonte ao lado. Editar aqui.
# ---------------------------------------------------------------------------
REGRAS: dict[str, tuple[RegraVisagismo, ...]] = {
    # -----------------------------------------------------------------------
    # ROSTO QUADRADO - a regra melhor fundamentada da tabela.
    # F1 e F2 concordam, com quase as mesmas palavras.
    # -----------------------------------------------------------------------
    "quadrado": (
        RegraVisagismo(
            formato_rosto="quadrado",
            formato_armacao="redondo",
            posicao=1,
            justificativa=(
                "Suaviza a mandibula marcada e os angulos fortes do rosto quadrado, "
                "por contraste de curva contra reta."
            ),
            # F1: "Square faces are generally balanced out with a round or oval frame
            #      because they provide a natural balance for sharp angles."
            # F2: "choose a rounded, rather than angular, frame. A round or oval
            #      eyeglass frame will soften as well as add contrast to your angular
            #      features."
            fontes=("F1", "F2"),
            sustentacao="convergente",
        ),
        RegraVisagismo(
            formato_rosto="quadrado",
            formato_armacao="ovalado",
            posicao=2,
            justificativa=(
                "Mesmo efeito de suavizacao do redondo, com linha um pouco mais "
                "discreta. As duas fontes citam redondo e oval na mesma frase, sem "
                "ordenar entre os dois."
            ),
            # F1: "...with a round or oval frame..."  F2: "A round or oval eyeglass frame..."
            fontes=("F1", "F2"),
            sustentacao="convergente",
            observacao=(
                "A ordem entre redondo e ovalado e escolha deste rascunho, nao das "
                "fontes: nenhuma das duas diz qual dos dois e preferivel."
            ),
        ),
        RegraVisagismo(
            formato_rosto="quadrado",
            formato_armacao="aviador",
            posicao=3,
            justificativa=(
                "Linha curva na base, citada por guias brasileiros entre as indicadas "
                "para rosto quadrado."
            ),
            fontes=("F3",),
            sustentacao="fonte-unica",
            observacao=(
                "Sustentada APENAS por F3, que e agregado de busca e fonte fraca. F1 e "
                "F2 nao citam aviador para rosto quadrado. Tratar como sugestao "
                "secundaria, nao como regra."
            ),
        ),
        RegraVisagismo(
            formato_rosto="quadrado",
            formato_armacao="gatinho",
            posicao=4,
            justificativa=(
                "F2 recomenda 'cat-eye shades' para rosto quadrado na secao de oculos "
                "de sol, e guias brasileiros o citam entre os indicados."
            ),
            fontes=("F2", "F3"),
            sustentacao="fonte-unica",
            observacao=(
                "Em F2 a mencao esta na secao de OCULOS DE SOL, nao de armacao de "
                "grau. Mantido em ultimo por isso."
            ),
        ),
    ),
    # -----------------------------------------------------------------------
    # ROSTO REDONDO - tambem bem fundamentada: F1 e F2 convergem em angular.
    # -----------------------------------------------------------------------
    "redondo": (
        RegraVisagismo(
            formato_rosto="redondo",
            formato_armacao="retangular",
            posicao=1,
            justificativa=(
                "Acrescenta linha reta e alonga o rosto redondo. E o item mais "
                "especifico em que as duas fontes coincidem."
            ),
            # F1: "rectangular ... known for making round faces appear longer and thinner"
            # F2: "Rectangular frames: Break up the face structure, which can make your
            #      face appear longer and thinner."
            fontes=("F1", "F2"),
            sustentacao="convergente",
        ),
        RegraVisagismo(
            formato_rosto="redondo",
            formato_armacao="quadrado",
            posicao=2,
            justificativa=(
                "Mesmo principio do retangular: angulo e estrutura contra a curva do "
                "rosto."
            ),
            # F1: "bold, angular frames"; "geometric or angular frames provide sharp
            #      lines to balance out your soft look"
            # F2 (tabela): "Round: Rectangle, Square, Geometric | Add angles and
            #      structure to soft curves"
            fontes=("F1", "F2"),
            sustentacao="convergente",
        ),
        RegraVisagismo(
            formato_rosto="redondo",
            formato_armacao="gatinho",
            posicao=3,
            justificativa=(
                "Linha ascendente nas extremidades, que as duas fontes citam "
                "explicitamente para rosto redondo."
            ),
            # F1: "rectangular, geometric, cat-eye, or D-frames"
            # F2: "Upswept frames such as cat-eye"
            fontes=("F1", "F2"),
            sustentacao="convergente",
        ),
        RegraVisagismo(
            formato_rosto="redondo",
            formato_armacao="ovalado",
            posicao=4,
            justificativa=(
                "Aparece nos links de produto de F2 para rosto redondo, contrariando o "
                "proprio texto da pagina, que pede angulo."
            ),
            fontes=("F2",),
            sustentacao="divergente",
            observacao=(
                "DIVERGENCIA INTERNA de F2: o corpo do texto pede angulo e estrutura "
                "para rosto redondo, e os links de produto da mesma pagina oferecem "
                "'Rectangle, Square and Oval'. Pelo principio de contraste, ovalado "
                "contraria a regra. Mantido em ultimo e marcado, em vez de omitido, "
                "para que a divergencia fique registrada."
            ),
        ),
    ),
    # -----------------------------------------------------------------------
    # ROSTO OVAL - as fontes convergem, mas a propria recomendacao e fraca:
    # as duas dizem que quase tudo serve.
    # -----------------------------------------------------------------------
    "oval": (
        RegraVisagismo(
            formato_rosto="oval",
            formato_armacao="quadrado",
            posicao=1,
            justificativa=(
                "Acrescenta definicao a um rosto ja proporcional. As duas fontes "
                "abrem a recomendacao de oval por quadrado/retangular/geometrico."
            ),
            # F1: "Oval faces are balanced well with square, rectangular, or geometric
            #      shaped frames."
            # F2 (tabela): "Oval: Square, Rectangle, Geometric | Add definition to
            #      balanced proportions"
            fontes=("F1", "F2"),
            sustentacao="convergente",
        ),
        RegraVisagismo(
            formato_rosto="oval",
            formato_armacao="retangular",
            posicao=2,
            justificativa=(
                "Citado junto com quadrado nas duas fontes, na mesma enumeracao e sem "
                "ordem entre eles."
            ),
            fontes=("F1", "F2"),
            sustentacao="convergente",
            observacao=(
                "A ordem entre quadrado e retangular e escolha deste rascunho: as "
                "fontes listam os dois juntos, sem preferencia."
            ),
        ),
        RegraVisagismo(
            formato_rosto="oval",
            formato_armacao="gatinho",
            posicao=3,
            justificativa=(
                "F1 o indica para rosto oval como opcao de maior personalidade "
                "('bolder statement')."
            ),
            # F1: "Cat-eye and aviator frames will provide a bolder statement, while
            #      D-frames offer a more classic look."
            fontes=("F1", "F2"),
            sustentacao="fonte-unica",
            observacao=(
                "F1 o trata como escolha estilistica, nao como correcao de proporcao. "
                "Em F2 a mencao esta na secao de oculos de sol."
            ),
        ),
        RegraVisagismo(
            formato_rosto="oval",
            formato_armacao="aviador",
            posicao=4,
            justificativa="Citado por F1 na mesma frase do gatinho, com a mesma funcao.",
            fontes=("F1", "F2"),
            sustentacao="fonte-unica",
            observacao=(
                "Mesma ressalva do gatinho. F2 cita 'aviators' para rosto oval na "
                "secao de oculos de sol."
            ),
        ),
        RegraVisagismo(
            formato_rosto="oval",
            formato_armacao="ovalado",
            posicao=5,
            justificativa=(
                "Nao e recomendado por nenhuma das fontes lidas para rosto oval, mas "
                "ambas afirmam que o rosto oval acomoda quase qualquer formato."
            ),
            # F2: "People with oval faces usually suit many different frame styles."
            fontes=("F2",),
            sustentacao="divergente",
            observacao=(
                "Entra por ser o formato mais versatil, nao por recomendacao direta. "
                "Pelo principio de contraste, armacao ovalada em rosto oval nao "
                "acrescenta definicao. F2 adverte que rosto oval deve evitar armacao "
                "ESTREITA ('steer clear of narrow frames'), o que e criterio de "
                "proporcao e nao de formato, e portanto nao modelavel aqui."
            ),
        ),
        RegraVisagismo(
            formato_rosto="oval",
            formato_armacao="redondo",
            posicao=6,
            justificativa=(
                "Mesma situacao do ovalado: aparece nos links de produto de F2 para "
                "rosto oval, sem recomendacao no texto."
            ),
            fontes=("F2",),
            sustentacao="divergente",
            observacao=(
                "Sustentacao fraca, igual a do ovalado. Listado por completude do "
                "ranking, nao por fundamentacao."
            ),
        ),
    ),
    # -----------------------------------------------------------------------
    # ROSTO DIAMANTE - A REGRA MAIS FRACA DA TABELA. Leia a ressalva.
    # F1, a fonte mais credivel, NAO COBRE este formato.
    # F2 se contradiz tres vezes dentro da propria pagina.
    # -----------------------------------------------------------------------
    "diamante": (
        RegraVisagismo(
            formato_rosto="diamante",
            formato_armacao="ovalado",
            posicao=1,
            justificativa=(
                "Suaviza os angulos e equilibra as macas do rosto salientes, que sao "
                "a caracteristica do rosto diamante. E o UNICO formato que aparece nas "
                "tres listas conflitantes de F2 e tambem no agregado F3."
            ),
            # F2 (corpo): "Oval, rimless, and browline frames are especially flattering
            #      because they add structure without emphasizing sharp facial features."
            # F2 (tabela): "Diamond | Oval, Rimless, Cat-eye"
            # F2 (produto): "Rectangle, Oval and Horn"
            fontes=("F2", "F3"),
            sustentacao="fonte-unica",
            observacao=(
                "F1, a fonte mais credivel, NAO cobre rosto diamante. Esta regra se "
                "apoia em fonte comercial que se contradiz, e sobrevive apenas porque "
                "'oval' e a intersecao das tres listas dela. Prioridade de revisao "
                "pelo autor."
            ),
        ),
        RegraVisagismo(
            formato_rosto="diamante",
            formato_armacao="gatinho",
            posicao=2,
            justificativa=(
                "Valoriza as macas do rosto por contraste com a linha ascendente, "
                "segundo a tabela-resumo de F2."
            ),
            # F2 (tabela): "Diamond | Oval, Rimless, Cat-eye | Highlight cheekbones and
            #      soften angles"
            fontes=("F2", "F3"),
            sustentacao="divergente",
            observacao=(
                "DIVERGENCIA: aparece na tabela-resumo de F2 e no agregado F3, e NAO "
                "aparece no corpo do texto de F2. Buscas tambem encontraram fonte "
                "indicando o oposto, que rosto diamante deve evitar armacao angulosa."
            ),
        ),
        RegraVisagismo(
            formato_rosto="diamante",
            formato_armacao="retangular",
            posicao=3,
            justificativa=(
                "Aparece nos links de produto de F2 para rosto diamante e em parte das "
                "fontes brasileiras agregadas em F3."
            ),
            fontes=("F2", "F3"),
            sustentacao="divergente",
            observacao=(
                "A DIVERGENCIA MAIS FORTE DA TABELA, e o motivo de estar em ultimo. "
                "Os links de produto de F2 oferecem 'Rectangle', enquanto buscas "
                "encontraram fonte afirmando que armacao quadrada, grande, angulosa e "
                "retangular pode NAO funcionar para rosto diamante. Ou seja: ha fonte "
                "a favor e fonte contra. Nao resolvo essa divergencia aqui, e nao "
                "deveria: ela precisa de uma fonte melhor que as duas."
            ),
        ),
    ),
}


class FormatoRostoSemRegraError(KeyError):
    """Nao existe regra declarada para o formato de rosto pedido."""


def ranquear(formato_rosto: str) -> tuple[RegraVisagismo, ...]:
    """Devolve os formatos de armacao recomendados, ja ranqueados.

    Funcao pura: nao le arquivo, nao usa estado global e devolve sempre o mesmo
    resultado para a mesma entrada.

    Args:
        formato_rosto: um dos 4 de :data:`FORMATOS_ROSTO`.

    Returns:
        As regras, da posicao 1 para a ultima. Nunca vazio.

    Raises:
        FormatoRostoSemRegraError: se nao houver regra para o formato pedido.
    """
    chave = str(formato_rosto).strip().lower()
    regras = REGRAS.get(chave)
    if regras is None:
        fora = FORMATOS_ROSTO_FORA_DO_MODELO.get(chave)
        extra = (
            f" O formato '{chave}' e tratado pela literatura, mas nao esta mapeado: "
            f"{fora}"
            if fora
            else ""
        )
        raise FormatoRostoSemRegraError(
            f"Nao existe regra de visagismo declarada para o formato de rosto "
            f"'{formato_rosto}'. Formatos com regra: {sorted(REGRAS)}.{extra} "
            "Para acrescentar um formato, edite REGRAS em visagismo.py, e lembre que "
            "regra sem fonte citada nao entra (D-44)."
        )
    return regras


def fontes_de(regras: Mapping[str, object] | tuple[RegraVisagismo, ...]) -> tuple[Fonte, ...]:
    """Devolve as fontes distintas citadas por um conjunto de regras.

    Args:
        regras: as regras devolvidas por :func:`ranquear`.

    Returns:
        As fontes, sem repeticao, na ordem em que aparecem.
    """
    vistas: list[str] = []
    for regra in regras:  # type: ignore[union-attr]
        if not isinstance(regra, RegraVisagismo):
            continue
        for fid in regra.fontes:
            if fid not in vistas:
                vistas.append(fid)
    return tuple(FONTES[fid] for fid in vistas if fid in FONTES)
