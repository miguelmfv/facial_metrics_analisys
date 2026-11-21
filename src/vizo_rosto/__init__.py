"""vizo_rosto: formato de rosto a partir de uma medicao facial, e recomendacao
prescritiva de formatos de armacao.

Este pacote faz duas coisas de naturezas diferentes, e a separacao entre elas e
o eixo metodologico do TCC:

1. **Descritivo**, aprendido dos dados: estima o formato de rosto
   (``diamante``, ``oval``, ``quadrado``, ``redondo``) com um RandomForest
   treinado em 313 rostos rotulados a mao. Desempenho documentado:
   0,47 +/- 0,08 em validacao cruzada 10-fold, contra 0,25 do acaso em 4
   classes.
2. **Prescritivo**, declarado: ranqueia formatos de armacao segundo regras de
   visagismo com fonte citada, em :mod:`vizo_rosto.visagismo`.

O que este pacote NAO faz, e nao deve passar a fazer: prever a armacao que uma
pessoa usa a partir do rosto dela. O experimento central deste trabalho testou
essa relacao em 2.784 imagens, com validacao agrupada por modelo de armacao, e
nao a encontrou (teste de permutacao, p global = 0,176). A recomendacao e um
criterio estetico declarado, nunca uma previsao aprendida.

Exemplo de uso::

    from vizo_rosto import carregar, recomendar

    modelo = carregar("modelos/modelo_formato_rosto_sem_linhas_branco.joblib")
    rec = recomendar(medicao_json, modelo)

    rec.formato_rosto                 # 'oval'
    rec.previsao.probabilidades       # as 4 classes, da maior para a menor
    [a.formato for a in rec.armacoes] # formatos de armacao ranqueados
    rec.para_dicionario()             # saida completa, de auditoria
    rec.para_contrato_backend()       # saida para o backend, sem porcentagem
"""

from __future__ import annotations

from . import entrada, features_faciais, modelo, recomendacao, visagismo
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
from .recomendacao import (
    CHAVES_CONTRATO,
    CHAVES_PROIBIDAS_NO_CONTRATO,
    VERSAO_CONTRATO,
    ArmacaoRecomendada,
    ContratoVioladoError,
    Recomendacao,
    calcular_features,
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
    "recomendacao",
    "visagismo",
    # entrada
    "CONTORNOS_EXIGIDOS",
    "EntradaInvalidaError",
    "achatar_medicao",
    "linha_de_csv",
    # modelo
    "CLASSES_ESPERADAS",
    "FeatureNaoFinitaError",
    "FeaturesIncompativeisError",
    "ModeloFormatoRosto",
    "ModeloIncompativelError",
    "PrevisaoRosto",
    "carregar",
    # recomendacao
    "CHAVES_CONTRATO",
    "CHAVES_PROIBIDAS_NO_CONTRATO",
    "VERSAO_CONTRATO",
    "ArmacaoRecomendada",
    "ContratoVioladoError",
    "Recomendacao",
    "calcular_features",
    "recomendar",
    # visagismo
    "AVISO_RASCUNHO",
    "FORMATOS_ARMACAO",
    "FORMATOS_ROSTO",
    "FormatoRostoSemRegraError",
    "RegraVisagismo",
]
