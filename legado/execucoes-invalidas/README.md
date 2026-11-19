# Execuções inválidas, preservadas de propósito

Esta pasta guarda saídas de execuções que **não valem como resultado**. Elas não foram
apagadas porque documentar um erro encontrado vale mais que escondê-lo: o histórico de como
o defeito foi detectado é parte da metodologia do trabalho.

**Nada aqui deve ser citado como resultado, nem usado como entrada de nada.**

---

## `review_com_prob_rosto.csv` (5,4 MB, 7.416 linhas)

**O que deveria ser:** a saída do `pipeline_rosto_armacao.py`, isto é, cada imagem de review
com a probabilidade de cada **formato de rosto** atribuída pelo modelo de formato de rosto.

**O que é de fato:** a saída de uma execução em que o `.joblib` passado em `--modelo-rosto`
era o modelo de **formato de armação**, não o de rosto — rodado sobre o pool **não rotulado**.

### Como isso foi estabelecido

Três evidências independentes, todas verificáveis no próprio arquivo:

1. **As colunas de probabilidade são formatos de armação.** O cabeçalho traz
   `p_rosto_aviador`, `p_rosto_coracao`, `p_rosto_gatinho`, `p_rosto_ovalado`,
   `p_rosto_quadrado`, `p_rosto_redondo`, `p_rosto_retangular` — **sete** colunas. O modelo
   de formato de rosto tem **quatro** classes (`diamante`, `oval`, `quadrado`, `redondo`) e
   nunca poderia produzir sete. `aviador`, `gatinho`, `coracao`, `ovalado` e `retangular` não
   são formatos de rosto: são formatos de armação.

2. **A coluna `rosto_top1` contém formatos de armação.** Distribuição medida:

   | valor | ocorrências |
   |---|---|
   | retangular | 2.181 |
   | ovalado | 1.267 |
   | redondo | 1.133 |
   | gatinho | 1.007 |
   | quadrado | 932 |
   | aviador | 624 |
   | coracao | 272 |

3. **São 7.416 linhas, exatamente o total de `arquivos_csv/faces-nao-classificado.csv`** —
   o pool sem rótulo. A execução autoritativa do experimento rosto → armação usa o CSV de
   review e reporta **2.784** linhas (ver
   `resultados/Resultado Algoritmos com todos dados - novo.txt`).

### Por que o arquivo importa, apesar de inválido

Este é exatamente o erro que as duas travas `checar_modelo_rosto` do
`pipeline_rosto_armacao.py` existem para impedir — elas abortam a execução se as classes do
"modelo de rosto" incluírem nomes que só existem como armação. A existência deste arquivo
**prova que o erro ocorreu antes das travas existirem**, e que as travas foram escritas em
resposta a um problema real, não por precaução abstrata.

Vale como nota de método na monografia: o pipeline carrega duas verificações defensivas cuja
motivação é documentada, e esta é a documentação.

### Procedência

- Detectado em 18/11/2026 pelo Sextante, durante a reorganização do repositório
  (branch `refactor/estrutura-e-recomendacao`), ao investigar por que havia dois arquivos
  com o mesmo nome e tamanhos diferentes.
- Verificado de forma independente pelo Maestro, pelos mesmos três critérios acima.
- Movido para cá por decisão do autor (**D-53**), registrada em
  `contexto/desenvolvimento/tarefa-analise-rodada-3-tarefa-1.md`.
- O arquivo não foi alterado: é byte a byte o que estava em `resultados/`.

---

## Arquivos relacionados, que **não** estão aqui

| Arquivo | Onde está | Situação |
|---|---|---|
| `review_com_prob_rosto--excluindo-coracao-gatinho.csv` | `resultados/` | **Legítimo** (4 classes de rosto), mas de uma execução com `--excluir coracao gatinho`: 2.550 linhas, não as 2.784 da execução autoritativa. Renomeado por **D-54** para o nome dizer de qual experimento veio |
| `resultados_previsao--nao-reproduzivel-com-artefato-atual.csv` | `resultados/` | Não se reproduz com o `.joblib` e o `features_faciais.py` atuais: 21 das 78 classes previstas divergem. Mantido por **D-55** porque sua acurácia (0,500) pode estar citada em texto |
| `resultados_previsao--linha-base-2026-11-18.csv` | `resultados/` | **A versão reproduzível**, gerada em 18/11/2026 pelo artefato e pelo código que estão neste repositório. Acurácia 0,513, top-2 0,756, sha256 `005f100f5cdef39e…` |
