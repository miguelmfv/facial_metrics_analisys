"""Adaptador de entrada: converte uma medicao em uma linha achatada de features.

Este modulo e a fronteira do pacote com o mundo externo. Ele aceita as duas
formas em que uma medicao chega (D-46):

1. o **JSON de uma medicao**, na forma exata em que o backend o guarda, isto e
   ``{"contours": {"face": [[x, y], ...], "leftEye": [...], ...}, ...}``;
2. uma **linha achatada de CSV**, ``dict`` ou ``pandas.Series`` com as chaves
   ``ct_<contorno>_x<i>`` / ``ct_<contorno>_y<i>``, que e o formato dos arquivos
   em ``arquivos_csv/``.

As duas convergem para o mesmo ``dict`` achatado, que e o que
:func:`vizo_rosto.features_faciais.extract` consome.

A entrada **nao tem rotulo**. Nenhuma funcao deste modulo le, exige ou devolve
``source_type``: o dado vem de um usuario real e nao existe classe verdadeira
para ele.

O nome da coluna no CSV e formado por ``ct_{chave_do_json}_x{i}``, exatamente a
mesma regra com que os CSVs foram gerados, o que torna a conversao direta e sem
perda. Verificado em ``VIzo/frontend-mobile/lib/main_batch.dart``, onde o
cabecalho e montado por
``for (var k = 0; k < e.value; k++) ...['ct_${e.key}_x$k', 'ct_${e.key}_y$k']``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

__all__ = [
    "CONTORNOS_EXIGIDOS",
    "EntradaInvalidaError",
    "achatar_medicao",
    "linha_de_csv",
    "nomes_de_coluna",
]


class EntradaInvalidaError(ValueError):
    """A medicao recebida nao pode ser convertida em features.

    Levantada sempre que falta contorno, a contagem de pontos nao bate ou uma
    coordenada nao e numero finito. A mensagem sempre diz **o que fazer**, nao
    apenas o que falhou.
    """


#: Contornos que ``features_faciais.extract`` exige, com a contagem exata de
#: pontos de cada um. Esta tabela e um **contrato**, nao uma preferencia: o
#: ``extract`` indexa pontos por posicao (por exemplo ``face[18]`` para o
#: queixo), de modo que um contorno com menos pontos que o previsto produziria
#: ``KeyError`` ou, pior, mediria o ponto errado.
#:
#: As contagens conferem com ``_contourSizes`` de ``main_batch.dart`` e com o
#: contrato da API em ``contexto/04-contrato-api.md``.
CONTORNOS_EXIGIDOS: dict[str, int] = {
    "face": 36,
    "leftEyebrowTop": 5,
    "leftEyebrowBottom": 5,
    "rightEyebrowTop": 5,
    "rightEyebrowBottom": 5,
    "leftEye": 16,
    "rightEye": 16,
    "upperLipTop": 11,
    "lowerLipBottom": 9,
    "noseBottom": 3,
}


def nomes_de_coluna() -> list[str]:
    """Devolve os nomes de coluna achatados exigidos, na ordem canonica.

    Sao 2 colunas por ponto (``_x`` e ``_y``). Os 10 contornos de
    :data:`CONTORNOS_EXIGIDOS` somam 111 pontos, logo 222 nomes de coluna.
    """
    nomes: list[str] = []
    for contorno, n_pontos in CONTORNOS_EXIGIDOS.items():
        for i in range(n_pontos):
            nomes.append(f"ct_{contorno}_x{i}")
            nomes.append(f"ct_{contorno}_y{i}")
    return nomes


def _coordenada(valor: Any, contorno: str, indice: int, eixo: str) -> float:
    """Converte uma coordenada para ``float``, recusando o que nao for numero finito."""
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        raise EntradaInvalidaError(
            f"O contorno '{contorno}' tem uma coordenada que nao e numero no ponto "
            f"{indice} (eixo {eixo}): {valor!r} (tipo {type(valor).__name__}). "
            f"Esperado um numero em pixels. Verifique se a medicao foi serializada "
            f"com pares [x, y] numericos e nao com texto ou null."
        ) from None
    if not math.isfinite(numero):
        raise EntradaInvalidaError(
            f"O contorno '{contorno}' tem a coordenada {eixo} do ponto {indice} igual a "
            f"{numero}, que nao e um numero finito. Descarte esta medicao e peca uma "
            f"nova captura: nao ha como calcular geometria facial com NaN ou infinito."
        )
    return numero


def _pontos_do_contorno(bruto: Any, contorno: str, n_esperado: int) -> list[tuple[float, float]]:
    """Valida e normaliza a lista de pontos de um contorno."""
    if isinstance(bruto, Mapping) or isinstance(bruto, (str, bytes)):
        raise EntradaInvalidaError(
            f"O contorno '{contorno}' veio como {type(bruto).__name__}, mas deveria ser "
            f"uma lista de {n_esperado} pares [x, y]. Verifique o campo 'contours' da "
            f"medicao."
        )
    if not isinstance(bruto, Iterable):
        raise EntradaInvalidaError(
            f"O contorno '{contorno}' veio como {type(bruto).__name__}, que nao e uma "
            f"lista. Esperado uma lista de {n_esperado} pares [x, y]."
        )

    pontos_brutos = list(bruto)
    if len(pontos_brutos) != n_esperado:
        raise EntradaInvalidaError(
            f"O contorno '{contorno}' veio com {len(pontos_brutos)} pontos, mas o "
            f"calculo das features exige exatamente {n_esperado}. Contorno incompleto "
            f"costuma significar rosto parcialmente fora do quadro ou deteccao de baixa "
            f"qualidade: peca uma nova captura em vez de completar os pontos que "
            f"faltam, porque as features sao medidas em posicoes fixas do contorno "
            f"(por exemplo, o ponto 18 de 'face' e o queixo) e um contorno remendado "
            f"mediria a parte errada do rosto."
        )

    pontos: list[tuple[float, float]] = []
    for i, ponto in enumerate(pontos_brutos):
        if isinstance(ponto, Mapping):
            if "x" not in ponto or "y" not in ponto:
                raise EntradaInvalidaError(
                    f"O ponto {i} do contorno '{contorno}' veio como objeto sem as "
                    f"chaves 'x' e 'y': {ponto!r}. Esperado [x, y] ou "
                    f"{{'x': ..., 'y': ...}}."
                )
            bx, by = ponto["x"], ponto["y"]
        else:
            if isinstance(ponto, (str, bytes)) or not isinstance(ponto, Iterable):
                raise EntradaInvalidaError(
                    f"O ponto {i} do contorno '{contorno}' veio como "
                    f"{type(ponto).__name__} ({ponto!r}), mas deveria ser um par "
                    f"[x, y]."
                )
            par = list(ponto)
            if len(par) != 2:
                raise EntradaInvalidaError(
                    f"O ponto {i} do contorno '{contorno}' tem {len(par)} coordenadas "
                    f"({par!r}), mas deveria ter exatamente 2: [x, y]."
                )
            bx, by = par

        pontos.append(
            (
                _coordenada(bx, contorno, i, "x"),
                _coordenada(by, contorno, i, "y"),
            )
        )
    return pontos


def achatar_medicao(medicao: Mapping[str, Any]) -> dict[str, float]:
    """Converte o JSON de uma medicao do backend em uma linha achatada.

    Aceita tanto a medicao completa, que tem a chave ``contours``, quanto o
    proprio dicionario de contornos. Campos que a medicao traga alem dos
    contornos (razoes, pose, expressao, metadados) sao **ignorados**: as 29
    features sao recalculadas inteiramente a partir dos contornos.

    Args:
        medicao: a medicao, na forma ``{"contours": {"face": [[x, y], ...]}}``
            ou diretamente ``{"face": [[x, y], ...], ...}``.

    Returns:
        Linha achatada com 222 chaves ``ct_<contorno>_x<i>`` / ``_y<i>``, em
        ``float``, pronta para ``features_faciais.extract``.

    Raises:
        EntradaInvalidaError: se faltar contorno, se a contagem de pontos nao
            bater, ou se alguma coordenada nao for numero finito.
    """
    if not isinstance(medicao, Mapping):
        raise EntradaInvalidaError(
            f"A medicao veio como {type(medicao).__name__}, mas deveria ser um "
            f"dicionario. Se o dado chegou como texto JSON, passe-o por "
            f"json.loads() antes."
        )

    bruto = medicao.get("contours")
    if bruto is None:
        # Pode ser que tenham passado o proprio dicionario de contornos.
        if any(chave in medicao for chave in CONTORNOS_EXIGIDOS):
            bruto = medicao
        else:
            presentes = sorted(medicao.keys())[:10]
            raise EntradaInvalidaError(
                "A medicao nao tem a chave 'contours' nem as chaves de contorno no "
                f"nivel de cima. Chaves encontradas: {presentes}. Passe o JSON da "
                "medicao como o backend o guarda, isto e com 'contours' contendo "
                f"{sorted(CONTORNOS_EXIGIDOS)}."
            )

    if not isinstance(bruto, Mapping):
        raise EntradaInvalidaError(
            f"O campo 'contours' veio como {type(bruto).__name__}, mas deveria ser um "
            "dicionario de nome de contorno para lista de pares [x, y]."
        )

    faltando = [nome for nome in CONTORNOS_EXIGIDOS if nome not in bruto]
    if faltando:
        raise EntradaInvalidaError(
            f"Faltam {len(faltando)} contornos exigidos pelo calculo das features: "
            f"{faltando}. Contornos recebidos: {sorted(bruto.keys())}. "
            "Cada um dos exigidos precisa estar presente com a contagem de pontos "
            f"certa ({ {nome: CONTORNOS_EXIGIDOS[nome] for nome in faltando} }). "
            "Se a deteccao do ML Kit nao devolveu algum deles, a medicao esta "
            "incompleta: peca uma nova captura."
        )

    linha: dict[str, float] = {}
    for contorno, n_esperado in CONTORNOS_EXIGIDOS.items():
        pontos = _pontos_do_contorno(bruto[contorno], contorno, n_esperado)
        for i, (x, y) in enumerate(pontos):
            linha[f"ct_{contorno}_x{i}"] = x
            linha[f"ct_{contorno}_y{i}"] = y
    return linha


def linha_de_csv(linha: Mapping[str, Any]) -> dict[str, float]:
    """Valida e normaliza uma linha achatada de CSV.

    Serve para o autor continuar testando com os arquivos de ``arquivos_csv/``.
    Aceita ``dict`` ou ``pandas.Series`` (qualquer mapeamento indexavel por
    nome de coluna). Colunas alem das de contorno sao ignoradas, inclusive
    ``source_type``: **o rotulo nunca e exigido nem usado**.

    Args:
        linha: uma linha com as colunas ``ct_<contorno>_x<i>`` / ``_y<i>``.

    Returns:
        A mesma linha reduzida as 222 colunas de contorno, em ``float``.

    Raises:
        EntradaInvalidaError: se faltar coluna de contorno ou se algum valor
            nao for numero finito.
    """
    try:
        chaves = set(linha.keys())
    except AttributeError:
        raise EntradaInvalidaError(
            f"A linha veio como {type(linha).__name__}, que nao se comporta como um "
            "mapeamento de nome de coluna para valor. Passe um dict ou um "
            "pandas.Series (por exemplo df.iloc[0])."
        ) from None

    exigidas = nomes_de_coluna()
    faltando = [nome for nome in exigidas if nome not in chaves]
    if faltando:
        n_ct = sum(1 for chave in chaves if str(chave).startswith("ct_"))
        raise EntradaInvalidaError(
            f"A linha nao tem {len(faltando)} das {len(exigidas)} colunas de contorno "
            f"exigidas. {len(chaves)} colunas no total, {n_ct} comecando com 'ct_'. "
            f"Primeiras que faltam: {faltando[:8]}. "
            "Se o CSV foi lido com o separador errado, as colunas vem todas grudadas "
            "em uma: confira que a leitura usou sep=';'."
        )

    saida: dict[str, float] = {}
    for nome in exigidas:
        # nome tem a forma ct_<contorno>_<eixo><indice>; a mensagem de erro
        # precisa do contorno e do indice, nao do nome da coluna.
        corpo = nome[len("ct_") :]
        contorno, _, resto = corpo.rpartition("_")
        eixo, indice = resto[0], resto[1:]
        saida[nome] = _coordenada(linha[nome], contorno, int(indice), eixo)
    return saida
