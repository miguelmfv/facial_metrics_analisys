# `legado/` — os scripts originais, inalterados

Os scripts deste diretório são **byte a byte idênticos** aos que estavam na raiz
do repositório antes da reorganização. Foram movidos com `git mv`, sem uma única
linha alterada, e **não devem ser alterados**.

## Por que preservá-los sem tocar

Os caminhos e os nomes destes scripts estão **citados nos documentos do TCC**, e
os `.txt` em `resultados/` referenciam os comandos exatos com que foram gerados.
Alterá-los comprometeria a reprodutibilidade declarada na monografia; apagá-los,
mais ainda. Inclusive o `visualizar_modelo.py`, que está **quebrado**, foi
preservado — ver a ressalva no fim.

## O que mudou de lugar, e o que isso exige

| Antes | Agora |
|---|---|
| `features_faciais.py` (raiz) | `src/vizo_rosto/features_faciais.py` |
| `modelo_formato_rosto_sem_linhas_branco.joblib` (raiz) | `modelos/modelo_formato_rosto_sem_linhas_branco.joblib` |
| os 5 scripts (raiz) | `legado/` |

Os scripts fazem `from features_faciais import build`. Como o módulo agora vive
dentro do pacote, é preciso dizer ao Python onde encontrá-lo — e isso se faz
**de fora**, por variável de ambiente, justamente para não alterar os scripts:

```bash
PYTHONPATH=src/vizo_rosto
```

Note que o `PYTHONPATH` aponta para o **diretório do pacote**, não para `src/`.
É de propósito: os scripts importam `features_faciais` como módulo de primeiro
nível, não como `vizo_rosto.features_faciais`.

O `.joblib` passou a ser passado por **argumento de linha de comando**, que já
era como os scripts o recebiam — então aqui não há nada a contornar, apenas o
caminho novo a escrever.

## Como rodar — comandos verificados

Execute **da raiz do repositório** (não de dentro de `legado/`), porque os
caminhos de CSV são relativos à raiz.

### Windows (PowerShell)

```powershell
$env:PYTHONPATH = "src/vizo_rosto"
.venv/Scripts/python legado/avaliar_modelo.py `
    modelos/modelo_formato_rosto_sem_linhas_branco.joblib `
    arquivos_csv/faces-teste-supervisionado-formato-rosto.csv
```

### Linux, macOS, ou Git Bash no Windows

```bash
PYTHONPATH=src/vizo_rosto .venv/Scripts/python legado/avaliar_modelo.py \
    modelos/modelo_formato_rosto_sem_linhas_branco.joblib \
    arquivos_csv/faces-teste-supervisionado-formato-rosto.csv
```

**Este comando foi executado e verificado.** Saída, idêntica à de antes da
reorganização:

```
Rostos avaliados: 78
Acurácia: 0.513
Top-2 accuracy: 0.756
```

E mais que isso: o `resultados_previsao.csv` gerado por ele é **byte a byte
idêntico** (sha256 `005f100f5cdef39e41...`) ao que o mesmo script produziu na
raiz antes de qualquer `git mv`. A movimentação não mudou comportamento nenhum.

### Os outros scripts

O mesmo `PYTHONPATH` serve para todos. O `pipeline_rosto_armacao.py` também faz
`from comparar_modelos import ...`, e isso continua funcionando sem configuração
extra porque os dois ficam lado a lado em `legado/`: o Python põe o diretório do
script executado no início do `sys.path`.

```bash
# treino do modelo de formato de rosto
#   ATENÇÃO: o default de --saida é "/resultados/..." com barra inicial, que no
#   Windows resolve na RAIZ DO DRIVE. Passe --saida sempre.
PYTHONPATH=src/vizo_rosto .venv/Scripts/python legado/modelo_formato_rosto.py \
    arquivos_csv/faces-formato-rosto-classificado.csv \
    --alvo source_type --saida resultados/modelo_novo.joblib

# comparação de algoritmos para prever formato de armação
PYTHONPATH=src/vizo_rosto .venv/Scripts/python legado/comparar_modelos.py \
    --csv arquivos_csv/faces-frame-type-frame-id.csv \
    --drop file_name --excluir coracao --features derivadas

# o experimento central: existe relação rosto -> armação?
PYTHONPATH=src/vizo_rosto .venv/Scripts/python legado/pipeline_rosto_armacao.py \
    --csv arquivos_csv/faces-frame-type-frame-id.csv \
    --modelo-rosto modelos/modelo_formato_rosto_sem_linhas_branco.joblib \
    --drop file_name --excluir coracao
```

> Só o `avaliar_modelo.py` foi executado na verificação desta tarefa. Os outros
> três **não** foram rodados; o procedimento de `PYTHONPATH` é o mesmo, mas isso
> é dedução a partir do import que cada um faz, não medição.

## Onde os scripts escrevem

| Script | Escreve |
|---|---|
| `avaliar_modelo.py` | `resultados_previsao.csv` **no diretório de trabalho atual** |
| `pipeline_rosto_armacao.py` | `review_com_prob_rosto.csv` no diretório atual |
| `modelo_formato_rosto.py` | onde `--saida` apontar |

Rodando da raiz, as duas primeiras saídas caem na raiz e estão no `.gitignore`.
**Confira `git status` antes de commitar** depois de rodar qualquer um deles.

## Uma ressalva honesta: `visualizar_modelo.py` está quebrado

Ele **não roda**, e não foi corrigido nesta tarefa de propósito (nada em
`legado/` é alterado). Dois defeitos independentes, pela leitura do código:

1. carrega `modelo_formato_rosto-t1.joblib`, um arquivo que **não existe** neste
   repositório;
2. passa o `dict` inteiro para `plot_tree` e acessa `model.estimators_` e
   `model.feature_importances_` no `dict`. Só funcionaria com `model["modelo"]`.

Era um script exploratório abandonado. Está preservado como registro.

## O que NÃO fazer

- Não edite nenhum arquivo deste diretório, nem para corrigir o que está errado.
  Se precisar de uma versão corrigida, escreva um módulo novo no pacote.
- Não apague nada daqui.
- `resultados/review_com_prob_rosto.csv` é saída de execução **comprovadamente
  errada** (modelo de armação no lugar do de rosto). Não cite aquele arquivo como
  resultado: leia a seção "Reprodutibilidade" do README da raiz.
