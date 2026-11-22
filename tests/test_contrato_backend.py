"""Testes do contrato que o backend consome (D-66).

ESTE ARQUIVO E UM TESTE DE AUSENCIA, e isso muda como ele e escrito.

Um teste comum confere que a saida tem o que deveria ter. Esse aqui tem de
conferir, alem disso, que ela **nao tem** o que nao pode ter -- e que continuara
nao tendo depois de uma edicao futura feita por alguem que nunca leu D-66. O
pedido do autor foi literal: nenhuma porcentagem de assertividade chega ao
backend, nem do rosto nem da armacao.

Por isso aqui existem QUATRO barreiras independentes, e nao uma. Cada uma pega
um jeito diferente de a porcentagem voltar:

1. **Conjunto exato de chaves.** Pega quem acrescenta ``probabilidades`` com
   esse nome. As chaves esperadas estao escritas a mao neste arquivo, e nao
   importadas de ``recomendacao.CHAVES_CONTRATO``: se o teste lesse a lista do
   proprio modulo que ele audita, bastaria acrescentar a chave nos dois lugares
   para o teste voltar a passar, e ele deixaria de servir para o que existe.
2. **Nenhum ``float``, em profundidade nenhuma.** Pega quem acrescenta a mesma
   informacao com outro nome (``nivel``, ``w``, ``x``), porque toda porcentagem
   e um numero fracionario e o contrato legitimo nao tem nenhum.
3. **Nenhum numero dentro de texto.** Pega quem transforma a confianca em
   string para escapar da barreira 2 (``"formato_rosto": "oval (51%)"``). Os
   dois campos de texto livre sao comparados **por igualdade** com o que o
   pacote declara (``PRINCIPIO`` e ``montar_aviso``), e todo outro texto tem de
   ser um identificador minusculo sem digito.

   Desde D-79 o ``aviso`` e **montado** a partir das metricas do artefato em
   uso, em vez de ser constante. Por isso a comparacao e com
   ``montar_aviso(modelo_armacao)`` e nao com um literal: um literal aqui
   congelaria numeros que o proximo retreino muda, e o teste passaria a
   reprovar o artefato certo.
4. **O guarda de tempo de execucao.** Confere que
   ``recomendacao._conferir_contrato`` realmente recusa, em vez de so existir.

Dado de entrada: as 78 linhas reais de
``arquivos_csv/faces-teste-supervisionado-formato-rosto.csv``, pelas fixtures de
``conftest.py``. Nenhum rosto sintetico, e nenhuma medicao do autor copiada para
dentro de arquivo versionado -- geometria facial de pessoa real nao entra em
repositorio com remote publico.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import pytest

from vizo_rosto import recomendar
from vizo_rosto.modelo import CLASSES_ESPERADAS as FORMATOS_ROSTO
from vizo_rosto.modelo import ModeloFormatoRosto
from vizo_rosto.modelo_armacao import CLASSES_ESPERADAS_ARMACAO as FORMATOS_ARMACAO
from vizo_rosto.modelo_armacao import ModeloFormatoArmacao
from vizo_rosto.recomendacao import (
    N_RECOMENDADOS_PADRAO,
    PRINCIPIO,
    ContratoVioladoError,
    _conferir_contrato,
    montar_aviso,
)

# ---------------------------------------------------------------------------
# O contrato, transcrito A MAO do §4 do briefing da rodada 3, tarefa 2.
# Mexer nestes tres literais e mexer no acordo com o backend, nao no teste.
# ---------------------------------------------------------------------------
CHAVES_ESPERADAS = {
    "formato_rosto",
    "ranking_armacoes",
    "principio",
    "aviso",
    "versao_contrato",
}

CHAVES_ESPERADAS_NO_ITEM_DO_RANKING = {"posicao", "formato"}

#: A versao do contrato, escrita a mao aqui pelo mesmo motivo das chaves. Subiu
#: para 2 em D-82, quando a segunda etapa deixou de ser a tabela de visagismo e
#: passou a ser um classificador. O backend tem a contrapartida disto em
#: `vizo_app/recommendationEngine.py` (`VERSAO_CONTRATO_ACEITA`) e **recusa** o
#: que nao conhece: enquanto la estiver 1, nenhuma recomendacao v2 e gravada.
VERSAO_ESPERADA = 2

#: Nomes que D-66 proibe nominalmente. Lista propria, independente da que o
#: modulo mantem, pelo motivo explicado no cabecalho.
CHAVES_PROIBIDAS = {
    "probabilidades",
    "confianca",
    "margem_1o_2o",
    "features",
}

#: Qualquer nome de chave que *cheire* a numero de incerteza. Mais larga que a
#: lista nominal de proposito: a informacao proibida tende a voltar com nome
#: novo, nao com o mesmo nome.
PADRAO_DE_CHAVE_SUSPEITA = re.compile(
    r"prob|conf|margem|score|percent|porcent|acur|certez|assert|pontua|peso|"
    r"nivel|rank_?valor",
    re.IGNORECASE,
)

#: Um numero decimal em qualquer notacao, inclusive cientifica e com virgula.
PADRAO_DE_DECIMAL = re.compile(r"\d+[.,]\d|\d[eE][-+]?\d")

#: Texto que um valor de identificador pode ter: minusculas e sublinhado, sem
#: digito, sem pontuacao e sem sinal de porcentagem.
PADRAO_DE_IDENTIFICADOR = re.compile(r"^[a-z_]+$")


def _percorrer(no: Any, caminho: str = "") -> list[tuple[str, Any, Any]]:
    """Achata o contrato em ``(caminho, chave, valor)`` para cada folha e chave.

    Funcao pura. Existe para que as asserts falem de **todo** o dicionario, em
    qualquer profundidade, e nao apenas do primeiro nivel: uma chave proibida
    escondida dentro de um item do ranking e tao proibida quanto no topo.
    """
    achatado: list[tuple[str, Any, Any]] = []
    if isinstance(no, dict):
        for chave, valor in no.items():
            onde = f"{caminho}.{chave}" if caminho else str(chave)
            achatado.append((onde, chave, valor))
            achatado.extend(_percorrer(valor, onde))
    elif isinstance(no, (list, tuple)):
        for i, item in enumerate(no):
            achatado.extend(_percorrer(item, f"{caminho}[{i}]"))
    return achatado


@pytest.fixture
def contrato(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> dict[str, Any]:
    """O contrato gerado a partir de uma linha real do CSV de teste."""
    return recomendar(linha_csv, modelo, modelo_armacao).para_contrato_backend()


# ===========================================================================
# BARREIRA 1 -- conjunto exato de chaves
# ===========================================================================


def test_o_contrato_tem_exatamente_as_cinco_chaves_acordadas(
    contrato: dict[str, Any]
) -> None:
    """Chave a mais quebra aqui, chave a menos tambem.

    Este e o teste que o briefing pediu em S2. Se alguem acrescentar
    ``probabilidades`` ao contrato, ``sobrando`` deixa de ser vazio e este
    assert falha com o nome da chave intrusa na mensagem.
    """
    obtidas = set(contrato)
    sobrando = obtidas - CHAVES_ESPERADAS
    faltando = CHAVES_ESPERADAS - obtidas
    assert not sobrando, (
        f"chave(s) nao acordada(s) no contrato do backend: {sorted(sobrando)}. "
        "O contrato do §4 tem 5 chaves e so elas. Se o backend precisa de mais "
        "um dado, isso e mudanca de contrato: combine com o backend e suba "
        "VERSAO_CONTRATO."
    )
    assert not faltando, f"chave(s) acordada(s) ausente(s): {sorted(faltando)}"


def test_cada_item_do_ranking_tem_exatamente_posicao_e_formato(
    contrato: dict[str, Any]
) -> None:
    """A barreira 1 vale dentro do ranking, nao so no topo.

    Acrescentar ``{"posicao": 1, "formato": "redondo", "confianca": 0.4}``
    quebra aqui.
    """
    for i, item in enumerate(contrato["ranking_armacoes"]):
        assert set(item) == CHAVES_ESPERADAS_NO_ITEM_DO_RANKING, (
            f"ranking_armacoes[{i}] tem as chaves {sorted(item)}, mas o contrato "
            f"acordou exatamente {sorted(CHAVES_ESPERADAS_NO_ITEM_DO_RANKING)}"
        )


def test_nenhuma_chave_proibida_por_d66_em_profundidade_nenhuma(
    contrato: dict[str, Any]
) -> None:
    """As quatro chaves que D-66 proibe pelo nome."""
    for onde, chave, _ in _percorrer(contrato):
        assert str(chave).lower() not in CHAVES_PROIBIDAS, (
            f"a chave proibida por D-66 '{chave}' aparece em '{onde}'"
        )


def test_nenhum_nome_de_chave_sugere_numero_de_incerteza(
    contrato: dict[str, Any]
) -> None:
    """Mais largo que a lista nominal: pega a mesma coisa com nome novo."""
    for onde, chave, _ in _percorrer(contrato):
        assert not PADRAO_DE_CHAVE_SUSPEITA.search(str(chave)), (
            f"o nome de chave '{chave}', em '{onde}', sugere numero de "
            "assertividade. D-66 nao proibe quatro nomes, proibe a informacao: "
            "renomear nao resolve."
        )


# ===========================================================================
# BARREIRA 2 -- nenhum float
# ===========================================================================


def test_o_contrato_nao_carrega_nenhum_float(contrato: dict[str, Any]) -> None:
    """Toda porcentagem e fracionaria; o contrato legitimo nao tem fracao.

    Esta barreira nao depende de nome nenhum, e por isso e a que resiste a
    renomeacao. Os unicos numeros legitimos do contrato sao ``posicao`` e
    ``versao_contrato``, inteiros.
    """
    for onde, _, valor in _percorrer(contrato):
        assert not isinstance(valor, float), (
            f"'{onde}' carrega o float {valor!r}. O contrato do backend nao leva "
            "numero fracionario: um float aqui e uma porcentagem de "
            "assertividade cruzando a fronteira que D-66 fechou."
        )


def test_todo_numero_do_contrato_e_int_do_python_e_nao_do_numpy(
    contrato: dict[str, Any]
) -> None:
    """``numpy.int64`` nao e ``int``, e o JSONField do backend recusa.

    Regra de tipo exato, nao ``isinstance``: ``numpy.int64`` passaria num
    ``isinstance(v, int)`` em algumas plataformas e ``bool`` passa sempre.
    """
    for onde, _, valor in _percorrer(contrato):
        if isinstance(valor, (dict, list, tuple, str)):
            continue
        assert type(valor) is int, (
            f"'{onde}' e {type(valor).__name__} ({valor!r}), mas so int nativo e "
            "str sao aceitos como folha do contrato"
        )


# ===========================================================================
# BARREIRA 3 -- nenhum numero escondido dentro de texto
# ===========================================================================


def test_os_campos_de_texto_livre_sao_os_declarados_pelo_pacote_sem_alteracao(
    contrato: dict[str, Any], modelo_armacao: ModeloFormatoArmacao
) -> None:
    """Comparacao por igualdade, para que nada possa ser anexado ao texto.

    ``aviso`` e ``principio`` sao os dois campos de texto longo do contrato, e
    seriam o esconderijo obvio para uma confianca interpolada
    (``aviso + f" (confianca {c:.0%})"``). Exigir igualdade exata fecha esse
    caminho, e de passagem prova D-63: o aviso chega ao backend integro, nao
    resumido nem reescrito.

    O aviso **legitimamente** cita numeros -- as metricas medidas do artefato e
    o p = 0,176 do resultado negativo. E por isso que ele e comparado com o que
    `montar_aviso` produz para este mesmo artefato: o que nao pode e um numero
    que nao venha de la.
    """
    assert contrato["aviso"] == montar_aviso(modelo_armacao)
    assert contrato["principio"] == PRINCIPIO


def test_os_valores_de_identificador_nao_tem_digito_nem_sinal_de_porcentagem(
    contrato: dict[str, Any]
) -> None:
    """``"oval"`` passa; ``"oval (51%)"`` e ``"oval:0.51"`` quebram."""
    identificadores = {"formato_rosto", "formato"}
    for onde, chave, valor in _percorrer(contrato):
        if chave in identificadores:
            assert isinstance(valor, str) and PADRAO_DE_IDENTIFICADOR.match(valor), (
                f"'{onde}' vale {valor!r}, e um valor de formato tem de ser so "
                "minusculas e sublinhado. Digito ou '%' aqui e porcentagem "
                "disfarcada de rotulo."
            )


def test_o_json_serializado_nao_contem_decimal_fora_dos_textos_livres(
    contrato: dict[str, Any]
) -> None:
    """Varredura final sobre o payload inteiro, do jeito que ele viaja.

    Os dois textos livres contem numeros legitimos (as metricas medidas e o
    p = 0,176 do resultado negativo, que D-63 manda viajar), entao sao
    removidos antes da varredura em vez de a varredura ser enfraquecida.
    """
    sem_texto_livre = {
        chave: valor
        for chave, valor in contrato.items()
        if chave not in ("aviso", "principio")
    }
    texto = json.dumps(sem_texto_livre, ensure_ascii=False)
    achados = PADRAO_DE_DECIMAL.findall(texto)
    assert not achados, (
        f"o contrato serializado contem decimal(is) {achados} em {texto!r}. "
        "Nenhum numero fracionario pode chegar ao backend (D-66)."
    )


# ===========================================================================
# BARREIRA 4 -- o guarda de tempo de execucao funciona de verdade
# ===========================================================================


def test_o_guarda_recusa_a_chave_probabilidades_acrescentada_depois(
    contrato: dict[str, Any]
) -> None:
    """Simula exatamente o cenario do briefing: alguem poe 'probabilidades' ali.

    Prova que a recusa nao depende deste arquivo de teste existir: ela acontece
    em tempo de execucao, dentro do pacote, na primeira chamada.
    """
    adulterado = dict(contrato)
    adulterado["probabilidades"] = {"oval": 0.51, "quadrado": 0.49}
    with pytest.raises(ContratoVioladoError) as erro:
        _conferir_contrato(adulterado)
    assert "probabilidades" in str(erro.value)
    assert "para_dicionario" in str(erro.value), (
        "a mensagem tem de dizer o que fazer, nao apenas que falhou"
    )


def test_o_guarda_recusa_confianca_renomeada_para_escapar_da_lista_de_nomes(
    contrato: dict[str, Any]
) -> None:
    """A barreira do float nao depende do nome da chave."""
    adulterado = dict(contrato)
    adulterado["nota_interna"] = 0.5083
    with pytest.raises(ContratoVioladoError):
        _conferir_contrato(adulterado)


def test_o_guarda_recusa_chave_proibida_dentro_de_um_item_do_ranking(
    contrato: dict[str, Any]
) -> None:
    """A conferencia e recursiva, nao so de primeiro nivel."""
    adulterado = json.loads(json.dumps(contrato))
    adulterado["ranking_armacoes"][0]["confianca"] = 0.4
    with pytest.raises(ContratoVioladoError) as erro:
        _conferir_contrato(adulterado)
    assert "ranking_armacoes[0].confianca" in str(erro.value), (
        "a mensagem tem de apontar o caminho exato dentro do dicionario"
    )


def test_o_guarda_recusa_numpy_float_que_e_o_vazamento_mais_provavel(
    contrato: dict[str, Any]
) -> None:
    """``numpy.float64`` vem do estimador e nao e ``float`` do Python.

    E o vazamento mais provavel na pratica, porque ele atravessaria uma
    checagem ingenua de ``isinstance(v, float)``... exceto que ``numpy.float64``
    herda de ``float``. Fica o teste para fixar isso como garantido, e nao como
    suposto.
    """
    import numpy as np

    adulterado = dict(contrato)
    adulterado["versao_contrato"] = np.float64(1.0)
    with pytest.raises(ContratoVioladoError):
        _conferir_contrato(adulterado)


def test_o_guarda_recusa_chave_que_falta(contrato: dict[str, Any]) -> None:
    """Contrato incompleto e tao violacao quanto contrato inflado."""
    adulterado = {k: v for k, v in contrato.items() if k != "aviso"}
    with pytest.raises(ContratoVioladoError) as erro:
        _conferir_contrato(adulterado)
    assert "aviso" in str(erro.value)


# ===========================================================================
# O conteudo: o contrato nao e so bem-formado, ele esta certo
# ===========================================================================


def test_o_formato_de_rosto_e_um_valor_unico_dos_quatro_do_modelo(
    contrato: dict[str, Any]
) -> None:
    """D-61: um valor, nao uma lista de rostos ranqueada."""
    assert isinstance(contrato["formato_rosto"], str)
    assert contrato["formato_rosto"] in FORMATOS_ROSTO


def test_o_ranking_de_armacoes_e_ordenado_sem_buraco_e_sem_repeticao(
    contrato: dict[str, Any]
) -> None:
    ranking = contrato["ranking_armacoes"]
    assert len(ranking) == N_RECOMENDADOS_PADRAO, (
        f"o ranking tem {len(ranking)} itens; D-84 manda levar "
        f"{N_RECOMENDADOS_PADRAO} ao contrato. O ranking completo das 7 classes "
        "continua disponivel em previsao_armacao, fora do contrato."
    )
    assert [item["posicao"] for item in ranking] == list(range(1, len(ranking) + 1))
    formatos = [item["formato"] for item in ranking]
    assert len(set(formatos)) == len(formatos), f"formato repetido no ranking: {formatos}"
    assert all(f in FORMATOS_ARMACAO for f in formatos)


def test_a_versao_do_contrato_e_a_acordada(contrato: dict[str, Any]) -> None:
    """D-82: a versao subiu, e o backend precisa saber disso.

    O numero esta escrito a mao neste teste de proposito. Subir
    `VERSAO_CONTRATO` no pacote sem subir `VERSAO_CONTRATO_ACEITA` no backend
    faz toda recomendacao ser recusada na gravacao; o teste quebrando aqui e o
    lembrete de que as duas pontas mudam juntas.
    """
    assert type(contrato["versao_contrato"]) is int
    assert contrato["versao_contrato"] == VERSAO_ESPERADA


def test_o_contrato_sobrevive_a_ida_e_volta_por_json(contrato: dict[str, Any]) -> None:
    """O backend grava isto em JSONField: o round-trip tem de ser identico."""
    assert json.loads(json.dumps(contrato, ensure_ascii=False)) == contrato


def test_o_contrato_e_deterministico_para_a_mesma_medicao(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    """Seed fixa, modelos congelados: duas chamadas dao o mesmo dicionario."""
    primeiro = recomendar(linha_csv, modelo, modelo_armacao).para_contrato_backend()
    segundo = recomendar(linha_csv, modelo, modelo_armacao).para_contrato_backend()
    assert primeiro == segundo


def test_as_duas_formas_de_entrada_dao_o_mesmo_contrato(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
    medicao_json: dict[str, Any],
) -> None:
    """JSON do backend e linha de CSV, mesmo rosto real, mesmo contrato."""
    do_csv = recomendar(linha_csv, modelo, modelo_armacao).para_contrato_backend()
    do_json = recomendar(medicao_json, modelo, modelo_armacao).para_contrato_backend()
    assert do_json == do_csv


# ===========================================================================
# As 78 linhas reais: nenhuma delas pode escapar de nenhuma barreira
# ===========================================================================


def test_as_78_linhas_reais_respeitam_o_contrato_inteiro(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    df_teste: pd.DataFrame,
) -> None:
    """Um caso que passa pode ser sorte; 78 nao sao.

    Repete as quatro barreiras sobre cada rosto real do CSV de teste. O tamanho
    do ranking agora e constante (D-84), mas o conteudo muda a cada rosto, e um
    contrato conferido em um rosto so nao foi conferido.
    """
    formatos_vistos: set[str] = set()
    for i in range(len(df_teste)):
        c = recomendar(df_teste.iloc[i], modelo, modelo_armacao).para_contrato_backend()
        formatos_vistos.add(c["formato_rosto"])

        assert set(c) == CHAVES_ESPERADAS, f"linha {i}: chaves {sorted(c)}"
        for onde, chave, valor in _percorrer(c):
            assert str(chave).lower() not in CHAVES_PROIBIDAS, f"linha {i}: {onde}"
            assert not PADRAO_DE_CHAVE_SUSPEITA.search(str(chave)), f"linha {i}: {onde}"
            assert not isinstance(valor, float), f"linha {i}: {onde} = {valor!r}"
        assert c["aviso"] == montar_aviso(modelo_armacao), f"linha {i}"
        assert [it["posicao"] for it in c["ranking_armacoes"]] == list(
            range(1, len(c["ranking_armacoes"]) + 1)
        ), f"linha {i}"

    assert len(formatos_vistos) >= 2, (
        f"as 78 linhas sairam com um unico formato de rosto ({formatos_vistos}); "
        "o teste nao exercitou rankings de tamanhos diferentes"
    )


def test_para_dicionario_continua_levando_a_incerteza_que_o_contrato_nao_leva(
    modelo: ModeloFormatoRosto,
    modelo_armacao: ModeloFormatoArmacao,
    linha_csv: pd.Series,
) -> None:
    """As duas saidas tem publicos diferentes, e isso nao mudou com D-79.

    A monografia precisa das probabilidades das duas etapas; o backend nao pode
    receber nenhuma. Se alguem "arrumar" ``para_dicionario`` para obedecer a
    D-66, a auditoria do TCC fica sem os numeros e este teste quebra.
    """
    d = recomendar(linha_csv, modelo, modelo_armacao).para_dicionario()
    assert len(d["formato_rosto"]["probabilidades"]) == 4
    assert isinstance(d["formato_rosto"]["confianca"], float)
    assert isinstance(d["formato_rosto"]["margem_1o_2o"], float)
    assert isinstance(d["formatos_armacao"]["confianca"], float)
    assert isinstance(d["formatos_armacao"]["margem_1o_2o"], float)
    assert len(d["formatos_armacao"]["ranking"]) == len(modelo_armacao.classes)
    assert all(
        isinstance(item["probabilidade"], float)
        for item in d["formatos_armacao"]["ranking"]
    )
