# facial_metrics_analisys

Análise de métricas faciais e recomendação de formatos de armação, para o TCC
**Vizo** (Análise e Desenvolvimento de Sistemas, IFRS).

O aplicativo mede 29 características faciais **no próprio aparelho**, com ML Kit,
e envia só números — **nunca a imagem** — para uma API Django. Este repositório
é a parte de análise de dados e de modelagem: recebe uma medição, estima o
formato de rosto e devolve formatos de armação ranqueados.

---

## O resultado que você precisa conhecer antes de ler o código

**O resultado central deste TCC é negativo.** Testamos se existe relação entre o
formato de rosto de uma pessoa e o formato de armação que ela usa, em 2.784
imagens, com validação agrupada por modelo de armação, e **não encontramos
nenhuma relação estatisticamente detectável** (teste de permutação, p global =
**0,176**). Acrescentar as probabilidades de formato de rosto ao modelo de
armação dá **ganho nulo**, dentro do desvio-padrão dos folds.

Isso tem uma consequência direta sobre o que este código faz e não faz:

| | Natureza | Como se obtém |
|---|---|---|
| Formato de **rosto** | **descritivo** | RandomForest treinado em 313 rostos rotulados à mão |
| Formatos de **armação** | **prescritivo** | regra de visagismo declarada, com fonte citada |

**Nenhum modelo neste repositório prevê armação a partir do rosto, e nenhum
deve passar a prever** — é exatamente a hipótese que o experimento refutou. O
que os dados refutam é que as pessoas **já usem** a armação que o visagismo
recomenda; a regra prescritiva continua sendo uma regra prescritiva, e por isso
ela é escrita à mão, com fonte, em `src/vizo_rosto/visagismo.py`.

> ⚠️ A tabela de visagismo é um **rascunho ainda não revisado pelo autor**.
> Leia o cabeçalho de `visagismo.py` antes de usar qualquer regra dela: a fonte
> mais credível que consegui ler **não cobre o rosto diamante**, e a fonte
> comercial se contradiz três vezes dentro da própria página.

---

## Instalação

Python 3.13. As versões são **fixas** e isso não é zelo excessivo — ver
"Verificação do artefato".

```bash
cd facial_metrics_analisys
python -m venv .venv

# Windows (PowerShell)
.venv/Scripts/python -m pip install -r requirements.txt

# Linux / macOS
.venv/bin/python -m pip install -r requirements.txt
```

Para rodar os testes, instale também as dependências de desenvolvimento:

```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

Este repositório **não é instalado como pacote** (não há `pip install .`), porque
nenhuma dependência de build foi autorizada. Fora dos testes, aponte o
`PYTHONPATH` para `src/`:

```bash
PYTHONPATH=src .venv/Scripts/python -c "import vizo_rosto; print(vizo_rosto.FORMATOS_ROSTO)"
```

Nos testes isso é automático: `pyproject.toml` declara `pythonpath = ["src"]`.

---

## Como usar

```python
from vizo_rosto import carregar, recomendar

modelo = carregar("modelos/modelo_formato_rosto_sem_linhas_branco.joblib")

# A entrada aceita as duas formas, e elas dão o MESMO resultado:
#   1) o JSON de uma medição, como o backend o guarda
#   2) uma linha achatada de CSV (dict ou pandas.Series com colunas ct_*)
rec = recomendar(medicao, modelo)

rec.formato_rosto                   # 'oval'
rec.previsao.probabilidades         # as 4 classes, da maior para a menor
rec.previsao.margem                 # distância entre a 1ª e a 2ª classe
[a.formato for a in rec.armacoes]   # formatos de armação ranqueados
rec.armacoes[0].fontes              # a fonte da regra aplicada
rec.para_dicionario()               # tudo em tipos nativos, para JSON
```

A entrada **não tem rótulo** e nenhum caminho do código exige um: o dado vem de
um usuário real e não existe classe verdadeira para ele. Se a medição trouxer um
campo de rótulo por acaso, ele é ignorado — há teste garantindo isso.

### O que a saída significa

```
formato_rosto.previsto        descritivo:  a classe mais provável
formato_rosto.probabilidades descritivo:  as 4 classes, sempre todas
formatos_armacao.ranking     prescritivo: regra declarada, com fonte
aviso                        a condição de rascunho e o resultado negativo
```

**As 4 probabilidades são devolvidas sempre, de propósito.** A acurácia do
modelo é 0,47 em 4 classes: o `top1` erra mais da metade das vezes. Quem consome
precisa poder ver uma previsão disputada (use `rec.previsao.margem`) e decidir o
que fazer com ela, em vez de receber uma única classe com falsa firmeza.

**O que a saída não faz:** escolher armações concretas do catálogo. O catálogo
vive no banco do backend, que este repositório não acessa. Casar formato com
produto é trabalho do backend, com esta saída na mão.

---

## Estrutura

```
src/vizo_rosto/
  features_faciais.py   as 29 features. LÓGICA CONGELADA: ver abaixo
  entrada.py            adaptador: JSON de medição | linha de CSV -> linha achatada
  modelo.py             carrega o .joblib, valida features, prevê formato de rosto
  visagismo.py          a tabela prescritiva, com fontes  <- RASCUNHO
  recomendacao.py       orquestra tudo e devolve a saída
tests/                  pytest
modelos/                o .joblib em uso, versionado
legado/                 os scripts originais, inalterados. Ver legado/README.md
arquivos_csv/           os datasets
resultados/             saídas salvas das execuções
```

A **lógica** de `features_faciais.py` é congelada: nenhuma fórmula, constante,
índice de ponto ou ordem de feature muda. O arquivo mudou de lugar, o conteúdo
não — o `git log --follow` mostra renomeação pura.

### As 29 features, em uma linha cada grupo

proporção altura/largura · 8 posições verticais (os "terços" do visagismo) ·
7 larguras em referências anatômicas · 5 razões entre larguras (testa × maçãs ×
mandíbula) · 5 ângulos de queixo e mandíbula · 2 de curvatura (área e
circularidade) · distância interocular.

Antes de qualquer medida, `extract` **corrige o roll**: rotaciona os pontos para
deixar a linha dos olhos horizontal. Sem isso, um rosto fotografado inclinado
teria altura e largura medidas em eixos errados.

As features mais importantes do modelo são `ang_queixo_largo` (0,156), `h_w`
(0,151) e `circularidade` (0,134) — as três somam 44% da importância, e são
justamente os critérios que a literatura de visagismo aponta. O modelo aprendeu
algo coerente com a teoria, não um artefato.

---

## Verificação do artefato, e por que as versões são fixas

O `.joblib` é **reusado congelado**: nada neste repositório retreina o modelo.
Retreinar produziria números diferentes dos que a monografia relata.

A versão de `scikit-learn` está fixada em **1.9.1** porque é a versão gravada
dentro do próprio pickle, legível nos bytes no campo `_sklearn_version`.
Desserializar um estimador em versão diferente dá, na melhor das hipóteses, um
aviso de incompatibilidade, e na pior um modelo silenciosamente diferente.

Verificado neste ambiente, com avisos tratados como erro:

| Verificação | Resultado |
|---|---|
| carrega sem erro e **sem aviso** | sim |
| classes | `['diamante', 'oval', 'quadrado', 'redondo']` |
| 29 features do `.joblib` × as que o código gera | idênticas em **nome e ordem** |

Há teste automatizado para isso (`tests/test_modelo.py`), com
`warnings.simplefilter("error")`: ele quebra se alguém subir a versão de
`scikit-learn` sem refazer a verificação.

### Desempenho

| Medida | Valor | Acaso |
|---|---|---|
| CV 10-fold nos 313 rostos de treino | **0,47 ± 0,08** | 0,25 |
| Acurácia nos 78 rostos do teste supervisionado | **0,513** | 0,25 |
| Top-2 nos mesmos 78 | **0,756** | 0,50 |

São **medidas diferentes** e não se substituem: a primeira é validação cruzada
no conjunto de treino, a segunda é um conjunto de teste separado.

Seed fixa em 42 em tudo que tem aleatoriedade.

---

## Medição não finita é recusada, e isso é deliberado

Se qualquer uma das 29 features sair `NaN` ou infinita, a medição é **recusada**
com uma mensagem que nomeia a feature culpada. Não há preenchimento com média
nem com mediana.

O motivo não é purismo, é uma medição:

```
linha intacta        -> diamante [0.508 0.313 0.108 0.071]
com h_w = NaN        -> diamante [0.431 0.399 0.100 0.070]   <- SEM erro, SEM aviso
com testa_maca = inf -> ValueError genérico, lá dentro do sklearn
```

A partir do scikit-learn 1.4, um valor ausente é roteado para o filho com mais
amostras, **em silêncio**. Com `h_w` ausente — a 2ª feature mais importante — a
probabilidade caiu de 0,508 para 0,431 e nada avisaria. Uma resposta errada com
cara de certa é pior que uma recusa.

A checagem é de **não-finito** (`NaN` **e** ±`inf`), não só de `NaN`: `inf` é
alcançável, porque as razões entre larguras dividem por uma largura que pode ser
zero, e uma checagem de `NaN` não o pega.

Nas três bases reais (313, 78 e 2.832 linhas), **nenhuma** linha produz `NaN`.

---

## Testes

```bash
.venv/Scripts/python -m pytest
```

68 testes. O caso ponta a ponta usa uma linha **real** de
`arquivos_csv/faces-teste-supervisionado-formato-rosto.csv`, nunca um rosto
sintético: dado sintético esconde exatamente os problemas que importam, e isso
já aconteceu neste projeto — um payload de exemplo tinha, como contorno de
rosto, uma **elipse gerada**.

Além do caminho feliz, os testes cobrem: contorno faltando, contagem de pontos
errada, coordenada não numérica, feature não finita, modelo de armação carregado
no lugar do de rosto, e a equivalência entre as duas formas de entrada.

---

## Reprodutibilidade

Os scripts originais, que geraram os números citados na monografia, estão em
`legado/`, inalterados. **Leia `legado/README.md`** para o procedimento de
execução depois da mudança de estrutura.

Três ressalvas sobre arquivos de resultado, documentadas porque afetam o que pode
ser citado. As movimentações decididas em **D-53, D-54 e D-55 foram aplicadas** no
commit `af614d6`: cada um dos três arquivos mudou de lugar ou de nome para que o
nome diga a procedência. **Nenhum foi apagado nem alterado** — os três renomeios
saíram a 100% de similaridade. Ver
`contexto/desenvolvimento/15-fechamento-analise-rodada-3-tarefa-1.md`.

- `resultados/resultados_previsao--nao-reproduzivel-com-artefato-atual.csv`
  **não se reproduz** com o modelo que está
  hoje no repositório: 21 das 78 classes previstas divergem da linha-base atual,
  com diferença máxima de probabilidade de 0,197. A causa é um `.joblib` que não
  existe mais **ou** uma fórmula antiga de feature, e as duas hipóteses **não
  são separáveis** com o que está versionado, porque a checagem de
  compatibilidade do `avaliar_modelo.py` compara **nomes** de feature, não
  valores. Acurácia daquele arquivo: 0,500. Da linha-base de hoje: 0,513. A
  versão **reproduzível** está ao lado, em
  `resultados/resultados_previsao--linha-base-2026-11-18.csv` (sha256
  `005f100f5cdef39e…`), gerada pelo artefato e pelo código deste repositório.
- `legado/execucoes-invalidas/review_com_prob_rosto.csv` é a saída de uma execução em que o
  modelo de **armação** foi usado no lugar do de **rosto**, rodada sobre o pool
  não rotulado. A prova: as colunas de probabilidade são `p_rosto_aviador`,
  `p_rosto_gatinho`, `p_rosto_retangular` e companhia — sete **formatos de
  armação** sob um prefixo que diz "rosto" —, `rosto_top1` contém formatos de
  armação, e são 7.416 linhas, exatamente o total de
  `arquivos_csv/faces-nao-classificado.csv`. É o erro que as duas travas
  `checar_modelo_rosto` do `pipeline_rosto_armacao.py` existem para impedir.
  **Não cite este arquivo como resultado.** `modelo.py` passou a recusar esse
  engano na carga, e há teste para isso. A pasta tem um `README.md` próprio com o
  diagnóstico completo e a procedência.
- `resultados/review_com_prob_rosto--excluindo-coracao-gatinho.csv` (antes na
  raiz) é legítimo (tem as 4 classes de rosto),
  mas são 2.550 linhas, de uma execução com `--excluir coracao gatinho`
  (2.832 − 48 − 234 = 2.550). **Não é** a execução de 2.784 linhas que sustenta
  o p global = 0,176, registrada em
  `resultados/Resultado Algoritmos com todos dados - novo.txt`.
