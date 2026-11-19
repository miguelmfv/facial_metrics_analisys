"""Testes da tabela prescritiva de visagismo (D-44).

Estes testes nao julgam se uma regra de visagismo e boa: isso depende de
revisao do autor e de literatura, nao de codigo. Eles garantem as propriedades
estruturais que D-44 exige e que, se violadas, tornariam a tabela
cientificamente indefensavel: toda regra tem fonte, nenhuma regra inventa
formato que o catalogo nao tem, e as limitacoes estao declaradas.
"""

from __future__ import annotations

import pytest

from vizo_rosto import visagismo
from vizo_rosto.visagismo import (
    AVISO_RASCUNHO,
    FONTES,
    FORMATOS_ARMACAO,
    FORMATOS_ARMACAO_SEM_REGRA,
    FORMATOS_ROSTO,
    REGRAS,
    FormatoRostoSemRegraError,
    ranquear,
)

SUSTENTACOES_VALIDAS = {"convergente", "fonte-unica", "divergente", "nao-fundamentada"}


def test_as_quatro_classes_do_modelo_tem_regra() -> None:
    """Se o modelo preve a classe, tem de haver recomendacao para ela."""
    assert set(REGRAS) == set(FORMATOS_ROSTO)
    for formato in FORMATOS_ROSTO:
        assert ranquear(formato), f"'{formato}' nao tem nenhuma regra"


def test_so_devolve_formatos_de_armacao_que_existem_no_dataset() -> None:
    """A tabela nao pode inventar formato que o catalogo nao conhece."""
    for formato_rosto in FORMATOS_ROSTO:
        for regra in ranquear(formato_rosto):
            assert regra.formato_armacao in FORMATOS_ARMACAO, (
                f"'{regra.formato_armacao}' nao e um dos 7 formatos do dataset "
                f"{FORMATOS_ARMACAO}"
            )


def test_toda_regra_tem_pelo_menos_uma_fonte() -> None:
    """D-44: regra sem fonte nao entra na tabela."""
    for formato_rosto in FORMATOS_ROSTO:
        for regra in ranquear(formato_rosto):
            assert regra.fontes, (
                f"a regra {formato_rosto} -> {regra.formato_armacao} esta sem fonte; "
                "D-44 proibe"
            )


def test_toda_fonte_citada_existe_e_esta_descrita() -> None:
    """Nao pode haver id de fonte citado que nao esteja em FONTES."""
    for formato_rosto in FORMATOS_ROSTO:
        for regra in ranquear(formato_rosto):
            for fid in regra.fontes:
                assert fid in FONTES, (
                    f"a regra {formato_rosto} -> {regra.formato_armacao} cita a fonte "
                    f"'{fid}', que nao existe em FONTES"
                )
                fonte = FONTES[fid]
                assert fonte.titulo and fonte.autoria
                assert fonte.consultado_em, "toda fonte tem de dizer quando foi consultada"
                assert fonte.natureza, "toda fonte tem de declarar que tipo de fonte e"


def test_posicoes_sao_sequenciais_e_sem_empate() -> None:
    for formato_rosto in FORMATOS_ROSTO:
        regras = ranquear(formato_rosto)
        posicoes = [r.posicao for r in regras]
        assert posicoes == list(range(1, len(regras) + 1)), (
            f"as posicoes de '{formato_rosto}' nao sao 1..n: {posicoes}"
        )


def test_nao_repete_formato_de_armacao_dentro_de_um_rosto() -> None:
    for formato_rosto in FORMATOS_ROSTO:
        formatos = [r.formato_armacao for r in ranquear(formato_rosto)]
        assert len(formatos) == len(set(formatos)), (
            f"'{formato_rosto}' recomenda o mesmo formato duas vezes: {formatos}"
        )


def test_sustentacao_declarada_e_de_um_tipo_conhecido() -> None:
    for formato_rosto in FORMATOS_ROSTO:
        for regra in ranquear(formato_rosto):
            assert regra.sustentacao in SUSTENTACOES_VALIDAS


def test_regra_divergente_explica_a_divergencia() -> None:
    """D-44: onde a literatura divergir, registre, nao escolha em silencio."""
    for formato_rosto in FORMATOS_ROSTO:
        for regra in ranquear(formato_rosto):
            if regra.sustentacao == "divergente":
                assert regra.observacao, (
                    f"a regra {formato_rosto} -> {regra.formato_armacao} esta marcada "
                    "como divergente mas nao explica a divergencia"
                )


def test_toda_regra_tem_justificativa_em_texto() -> None:
    for formato_rosto in FORMATOS_ROSTO:
        for regra in ranquear(formato_rosto):
            assert len(regra.justificativa) > 20, (
                f"{formato_rosto} -> {regra.formato_armacao} sem justificativa util"
            )


def test_coracao_fica_sem_regra_e_o_motivo_esta_declarado() -> None:
    """Nenhuma fonte recomenda armacao em formato de coracao; nao se inventa regra."""
    assert "coracao" in FORMATOS_ARMACAO_SEM_REGRA
    assert FORMATOS_ARMACAO_SEM_REGRA["coracao"]
    recomendados = {
        r.formato_armacao for formato in FORMATOS_ROSTO for r in ranquear(formato)
    }
    assert "coracao" not in recomendados


def test_limitacoes_de_cobertura_estao_declaradas() -> None:
    """Os formatos de rosto que a literatura trata e o modelo nao preve."""
    assert visagismo.FORMATOS_ROSTO_FORA_DO_MODELO
    for nome, motivo in visagismo.FORMATOS_ROSTO_FORA_DO_MODELO.items():
        assert nome not in FORMATOS_ROSTO, (
            f"'{nome}' esta listado como fora do modelo, mas o modelo o preve"
        )
        assert motivo


def test_fontes_inacessiveis_estao_registradas_e_nao_sao_citadas() -> None:
    """O que nao foi lido nao pode sustentar regra, e tem de constar como nao lido."""
    assert "Philip Hallawell (obra)" in visagismo.FONTES_NAO_ACESSIVEIS
    citadas = {
        fid for formato in FORMATOS_ROSTO for r in ranquear(formato) for fid in r.fontes
    }
    for nome in visagismo.FONTES_NAO_ACESSIVEIS:
        assert nome not in citadas


def test_aviso_de_rascunho_diz_o_essencial() -> None:
    """O aviso tem de viajar com a saida e tem de ser inequivoco."""
    texto = AVISO_RASCUNHO.lower()
    assert "rascunho" in texto
    assert "nao revisad" in texto or "ainda nao revisad" in texto
    assert "0,176" in AVISO_RASCUNHO, "o aviso tem de citar o resultado negativo"
    assert "nao aprendida" in texto or "nao aprendidas" in texto


def test_docstring_do_modulo_marca_o_rascunho() -> None:
    doc = visagismo.__doc__ or ""
    assert "RASCUNHO NAO REVISADO PELO AUTOR" in doc


def test_principio_geral_esta_declarado() -> None:
    assert "ontrast" in visagismo.principio_geral()


def test_formato_desconhecido_da_erro_que_orienta() -> None:
    with pytest.raises(FormatoRostoSemRegraError) as erro:
        ranquear("hexagonal")
    msg = str(erro.value)
    assert "visagismo.py" in msg
    assert "D-44" in msg


def test_formato_fora_do_modelo_da_erro_que_explica_a_limitacao() -> None:
    with pytest.raises(FormatoRostoSemRegraError) as erro:
        ranquear("triangular")
    assert "nao esta mapeado" in str(erro.value)


def test_ranquear_normaliza_caixa_e_espaco() -> None:
    assert ranquear("  OVAL  ") == ranquear("oval")


def test_ranquear_e_funcao_pura() -> None:
    """Mesma entrada, mesmo resultado, sem efeito colateral."""
    assert ranquear("quadrado") is ranquear("quadrado") or ranquear("quadrado") == ranquear(
        "quadrado"
    )


def test_fontes_de_devolve_sem_repeticao() -> None:
    fontes = visagismo.fontes_de(ranquear("quadrado"))
    ids = [f.id for f in fontes]
    assert ids == list(dict.fromkeys(ids))
    assert ids, "as regras de 'quadrado' citam fontes"


def test_diamante_esta_marcado_como_o_mais_fraco() -> None:
    """A fonte mais credivel nao cobre diamante; isso tem de estar visivel."""
    sustentacoes = {r.sustentacao for r in ranquear("diamante")}
    assert "convergente" not in sustentacoes, (
        "nenhuma regra de diamante pode se declarar convergente: F1 nao cobre diamante"
    )
    assert any(r.observacao for r in ranquear("diamante"))
