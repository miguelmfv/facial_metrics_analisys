"""vizo_rosto: formato de rosto e formatos de armacao a partir de uma medicao facial.

Desde **D-79** (22/11/2026) o pacote executa o **pipeline original**: dois
classificadores em sequencia, os dois aprendidos dos dados.

1. **Formato de rosto** (``diamante``, ``oval``, ``quadrado``, ``redondo``), com
   um RandomForest treinado em 313 rostos rotulados a mao. Desempenho
   documentado: 0,47 +/- 0,08 em validacao cruzada 10-fold, contra 0,25 do
   acaso em 4 classes. Sinal real, modesto.
2. **Formatos de armacao** ranqueados, com um classificador treinado nas 2.832
   fotos de catalogo, que recebe as 29 features **mais** as probabilidades de
   rosto da etapa 1 (D-80).

**O que o segundo modelo vale, dito aqui e nao so na monografia.** Ele preve no
nivel do acaso: 0,23 de acuracia contra 0,24 de responder sempre o formato mais
frequente, e 0,60 de acerto dentro de uma lista de tres contra 0,68 de uma lista
fixa das tres armacoes mais comuns. O experimento central do trabalho testou a
relacao rosto-armacao em 2.784 imagens, com validacao agrupada por modelo de
armacao, e nao a encontrou (permutacao, p global = 0,176). Usar o classificador
assim mesmo e decisao do autor (D-79); o aviso que viaja com cada recomendacao
declara esses numeros, lidos do proprio artefato.

:mod:`vizo_rosto.visagismo` continua no pacote como **registro historico** da
rodada 3 -- e o que a monografia cita --, mas nenhum caminho de execucao o
importa desde D-79.

Exemplo de uso::

    from vizo_rosto import carregar, carregar_armacao, recomendar

    rosto = carregar("modelos/modelo_formato_rosto_sem_linhas_branco.joblib")
    armacao = carregar_armacao("modelos/modelo_formato_armacao.joblib")
    rec = recomendar(medicao_json, rosto, armacao)

    rec.formato_rosto                      # 'oval'
    rec.formato_armacao                    # 'retangular'
    [a.formato for a in rec.armacoes]      # os 3 que vao para o backend
    rec.previsao_armacao.probabilidades    # as 7 classes, da maior para a menor
    rec.para_dicionario()                  # saida completa, de auditoria
    rec.para_contrato_backend()            # saida para o backend, sem porcentagem
"""

from __future__ import annotations

from . import entrada, features_faciais, modelo, modelo_armacao, recomendacao, visagismo
from .entrada import CONTORNOS_EXIGIDOS, EntradaInvalidaError, achatar_medicao, linha_de_csv
from .modelo import (
    CLASSES_ESPERADAS,
    FeatureNaoFinitaError,
    FeaturesIncompativeisError,
    ModeloFormatoRosto,
    ModeloIncompativelError,
    PrevisaoRosto,
    carregar,
)
from .modelo_armacao import (
    CLASSES_ESPERADAS_ARMACAO,
    CONJUNTOS_ENTRADA,
    PREFIXO_PROB_ROSTO,
    FormatoArmacao,
    ModeloFormatoArmacao,
    PrevisaoArmacao,
    ProbabilidadeDeRostoAusenteError,
    carregar_armacao,
)
from .recomendacao import (
    CHAVES_CONTRATO,
    CHAVES_PROIBIDAS_NO_CONTRATO,
    N_RECOMENDADOS_PADRAO,
    PRINCIPIO,
    VERSAO_CONTRATO,
    ContratoVioladoError,
    ModeloDeArmacaoAusenteError,
    Recomendacao,
    calcular_features,
    montar_aviso,
    recomendar,
)
from .visagismo import (
    AVISO_RASCUNHO,
    FORMATOS_ARMACAO,
    FORMATOS_ROSTO,
    FormatoRostoSemRegraError,
    RegraVisagismo,
)

__all__ = [
    # submodulos
    "entrada",
    "features_faciais",
    "modelo",
    "modelo_armacao",
    "recomendacao",
    "visagismo",
    # entrada
    "CONTORNOS_EXIGIDOS",
    "EntradaInvalidaError",
    "achatar_medicao",
    "linha_de_csv",
    # modelo de rosto
    "CLASSES_ESPERADAS",
    "FeatureNaoFinitaError",
    "FeaturesIncompativeisError",
    "ModeloFormatoRosto",
    "ModeloIncompativelError",
    "PrevisaoRosto",
    "carregar",
    # modelo de armacao
    "CLASSES_ESPERADAS_ARMACAO",
    "CONJUNTOS_ENTRADA",
    "PREFIXO_PROB_ROSTO",
    "FormatoArmacao",
    "ModeloFormatoArmacao",
    "PrevisaoArmacao",
    "ProbabilidadeDeRostoAusenteError",
    "carregar_armacao",
    # recomendacao
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
    # visagismo: registro historico da rodada 3, fora do caminho de execucao
    "AVISO_RASCUNHO",
    "FORMATOS_ARMACAO",
    "FORMATOS_ROSTO",
    "FormatoRostoSemRegraError",
    "RegraVisagismo",
]
