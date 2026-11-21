# O contrato com o backend

**Rodada 3, tarefa 2** · branch `feature/contrato-backend` · 21/11/2026
Autor deste documento: Sextante. Decisões de referência: D-57, D-61 a D-66.

Este documento descreve o que o pacote `vizo_rosto` entrega ao backend Django,
como o backend passa a importá-lo, e o custo medido disso. Todo número aqui foi
obtido por execução, e o comando que o produziu está citado ao lado.

---

## 1. A saída: `Recomendacao.para_contrato_backend()`

Método novo em `src/vizo_rosto/recomendacao.py`. O
`para_dicionario()` que já existia **não foi alterado** — as duas saídas têm
públicos diferentes e coexistem de propósito (§3).

```python
from vizo_rosto import carregar, recomendar

modelo = carregar(caminho_do_joblib)          # uma vez, no startup (§5)
contrato = recomendar(medicao, modelo).para_contrato_backend()
```

Retorno, com exatamente estas cinco chaves e nenhuma outra:

```python
{
    "formato_rosto": "oval",
    "ranking_armacoes": [
        {"posicao": 1, "formato": "quadrado"},
        {"posicao": 2, "formato": "retangular"},
        # ... de 3 a 6 itens, conforme o formato de rosto
    ],
    "principio": "Contraste geometrico: recomenda-se a armacao cuja forma ...",
    "aviso": "Recomendacao baseada em regras de visagismo declaradas, ...",
    "versao_contrato": 1,
}
```

| chave | tipo | garantias |
|---|---|---|
| `formato_rosto` | `str` | um de `diamante`, `oval`, `quadrado`, `redondo`. Valor único, não ranking (D-61) |
| `ranking_armacoes` | `list[dict]` | 3 a 6 itens; `posicao` começa em 1, sem buraco e sem repetição; `formato` é um dos 7 do dataset, nunca `coracao` |
| `principio` | `str` | idêntico a `visagismo.principio_geral()` |
| `aviso` | `str` | idêntico a `visagismo.AVISO_RASCUNHO`, 460 caracteres, íntegro (D-63) |
| `versao_contrato` | `int` | `recomendacao.VERSAO_CONTRATO` |

Tudo em tipos nativos (`str`, `int`, `list`, `dict`). Grava em `JSONField` sem
conversão e sobrevive a `json.dumps` → `json.loads` idêntico.

### Proibido neste dicionário

`probabilidades`, `confianca`, `margem_1o_2o`, `features` — e, por D-66, **a
informação**, não apenas esses quatro nomes. Renomear não resolve. O método
confere a própria saída antes de devolvê-la e levanta `ContratoVioladoError`
(subclasse de `AssertionError`, porque é defeito de programação e não erro do
dado de entrada) se a violação acontecer. Nenhuma medição de usuário pode
provocar essa exceção.

---

## 2. As quatro barreiras contra a porcentagem voltar

O pedido do autor é fácil de atender hoje e fácil de furar daqui a seis meses. A
proteção, então, é em camadas independentes — `tests/test_contrato_backend.py`,
22 testes:

| # | barreira | pega quem |
|---|---|---|
| 1 | conjunto exato de chaves, no topo e em cada item do ranking | acrescenta `probabilidades` com esse nome |
| 2 | nenhum `float`, em profundidade nenhuma; todo número é `int` nativo | renomeia para algo fora da lista (`nivel`, `w`, `nota_interna`) |
| 3 | `aviso` e `principio` conferidos **por igualdade** com as constantes; valores de formato têm de casar `^[a-z_]+$` | embute a confiança em texto (`"oval (51%)"`, `aviso + " Confianca: 0.51"`) |
| 4 | `_conferir_contrato` recusa em tempo de execução, dentro do pacote | tudo acima, já em produção, sem depender de ninguém rodar o pytest |

As chaves esperadas estão **escritas à mão** no arquivo de teste, e não
importadas de `recomendacao.CHAVES_CONTRATO`. Se o teste lesse a lista do módulo
que ele audita, bastaria acrescentar a chave nos dois lugares para o teste voltar
a passar — e ele deixaria de servir para aquilo que existe.

### Que as barreiras funcionam, verificado por mutação

Não basta afirmar que o teste quebraria. Foram aplicadas três mutações ao código
e a suíte foi executada em cada uma:

| mutação | resultado medido |
|---|---|
| **A** — `"probabilidades": dict(self.previsao.probabilidades)` no contrato | **3 falhas + 18 erros**; o guarda estoura já na fixture |
| **B** — a mesma chave **e** `_conferir_contrato` removido da chamada | **8 falhas** — o arquivo de teste é autossuficiente, não depende do guarda |
| **C** — `formato_rosto` virando `f"{formato} ({confianca:.0%})"` e `aviso + f" Confianca: {c:.2f}"` (strings, nenhuma chave nova, nenhum `float`) | **4 falhas** |

A mutação C é a que importa: é o escape que passaria por qualquer verificação
baseada em nome de chave ou em tipo numérico. O código foi restaurado ao estado
limpo depois de cada mutação; a suíte final passa com 90 testes.

---

## 3. Por que existem duas saídas, e não uma

`para_dicionario()` (de auditoria) continua levando probabilidades das 4 classes,
confiança, margem e as fontes citadas. `para_contrato_backend()` não leva nada
disso. Isso **não** é duplicação por conveniência de payload:

- a monografia precisa dos números, e o teste
  `test_para_dicionario_continua_levando_tudo_o_que_levava` quebra se alguém
  "corrigir" `para_dicionario` para obedecer a D-66;
- o backend não pode recebê-los, por decisão do autor.

O preço de D-66 está registrado no docstring do método porque quem lê o código
precisa saber dele: **a acurácia do modelo de rosto é 0,47 em 4 classes** — o
`top1` erra mais da metade das vezes — e, sem a confiança, o backend não tem como
distinguir uma previsão folgada de um quase-empate entre duas classes. A
informação existe e continua acessível por `para_dicionario()`; ela está fora do
contrato por decisão, não por ausência.

---

## 4. Instalação no backend (D-57)

Chamada de função no próprio processo do backend. Não há segundo serviço, não há
HTTP, não há token.

### 4.1 O que entra em `requirements.txt`

As quatro diretas, nas **mesmas versões** em que o artefato foi verificado:

```
scikit-learn==1.9.1
numpy==2.5.5
pandas==3.1.1
joblib==1.6.0
```

`scikit-learn==1.9.1` não é negociável: é a versão gravada dentro do próprio
`.joblib`, legível no campo `_sklearn_version` dos bytes do pickle.
Desserializar em outra versão dá, na melhor das hipóteses, um
`InconsistentVersionWarning`, e na pior um modelo silenciosamente diferente. O
modelo não é retreinado (D-48), então a versão do leitor tem de acompanhar a do
escritor.

**`matplotlib` não entra.** Ele está no `requirements.txt` deste repositório
porque os scripts de `legado/` desenham gráficos; nada no caminho
medição → features → previsão → ranking o importa. São 32 MiB que o backend não
precisa carregar.

Transitivas, que o pip resolve sozinho e que o backend vai pagar de todo jeito:
`scipy`, `narwhals`, `threadpoolctl`, `cloudpickle`, `python-dateutil`.

### 4.2 O custo em tamanho de imagem — medido, não estimado

Medição sobre os **wheels manylinux reais** para `cp313`, que é a plataforma do
backend (`FROM python:3.13` no `Dockerfile`), baixados com
`pip download --only-binary=:all: --platform manylinux_2_28_x86_64
--python-version 3.13 --abi cp313`, descompactados e compilados com
`python -m compileall`, que é o que `pip install` faz por padrão.

| pacote | sem `.pyc` | `.pyc` | total |
|---|---|---|---|
| scipy | 106,68 MiB | 25,19 MiB | **131,88 MiB** |
| pandas | 41,46 MiB | 31,49 MiB | 72,95 MiB |
| numpy | 50,32 MiB | 11,30 MiB | 61,63 MiB |
| scikit-learn | 29,57 MiB | 14,31 MiB | 43,88 MiB |
| narwhals | 2,02 MiB | 2,56 MiB | 4,58 MiB |
| joblib | 0,88 MiB | 0,97 MiB | 1,85 MiB |
| python-dateutil | 0,31 MiB | 0,32 MiB | 0,63 MiB |
| threadpoolctl | 0,11 MiB | — | 0,11 MiB |
| cloudpickle | 0,07 MiB | 0,05 MiB | 0,12 MiB |
| **subtotal pacotes** | **231,42 MiB** | **86,21 MiB** | **317,62 MiB** |
| `modelo_formato_rosto_sem_linhas_branco.joblib` | | | 0,85 MiB |
| fonte de `src/vizo_rosto/` (6 arquivos `.py`) | | | 0,08 MiB |
| **CRESCIMENTO TOTAL** | | | **318,55 MiB** (0,334 GB) |

**O número:** a imagem do backend cresce **318,55 MiB**. Hoje
`vizo-backend:latest` tem **1,98 GB** (`docker images`), então o resultado fica
em **≈ 2,31 GB — crescimento de ~17 %**. Em transferência de registry, os wheels
somam **70,03 MiB** comprimidos.

Três observações sobre esse número:

- **`scipy` é 41 % do crescimento** (131,88 MiB) e o projeto nunca o importa
  diretamente: é dependência transitiva do `scikit-learn`. Não é removível.
- **`PIP_NO_COMPILE=1` economiza 86,21 MiB** (−27 %), caindo para 232,35 MiB. O
  preço é o primeiro import de cada módulo ficar mais lento, e em container
  com filesystem só-leitura o `.pyc` não é regravado. É uma troca disponível, não
  uma recomendação.
- `tzdata` **não** conta. O `pandas` só o exige em `win32` e `emscripten`; no
  container Linux ele não é puxado, e o backend já pina `tzdata==2025.3` por
  conta própria. **Não há conflito de versão** entre o que o `vizo_rosto` precisa
  e o que o backend já tem — o único pacote em comum é `typing_extensions`
  (4.16.0 nos dois) e nenhuma das quatro diretas o exige.

### 4.3 Como o pacote chega lá dentro

`vizo_rosto` **não é instalável por pip**: o `pyproject.toml` deste repositório
não tem `[project]` nem `[build-system]`, de propósito (D-51 — nenhuma
dependência de build foi autorizada). Um `pip install .` aqui falha, e deve
falhar. As opções, então, são duas:

1. **Copiar `src/vizo_rosto/` para dentro da imagem** e pôr o diretório que o
   contém no `PYTHONPATH`. É o que o repositório já documenta como caminho de
   uso fora dos testes, e não exige autorização nova.
2. Autorizar `[build-system]` no `pyproject.toml` e instalar por pip. **Depende
   de decisão do autor**, porque mexe em D-51. Não fiz.

Em qualquer uma das duas, a versão do pacote e a do `.joblib` precisam andar
juntas, e não há hoje nada no pacote que carregue um número de versão
(`vizo_rosto` não tem `__version__`). Ver §7.

---

## 5. De onde o backend carrega o `.joblib`, e o que acontece se ele não estiver lá

### 5.1 De onde

`carregar()` **não tem caminho padrão embutido**, e isso é deliberado: caminho
absoluto chumbado no código já foi problema real neste repositório. Quem chama
informa o caminho. Para o backend, isso significa:

- o arquivo `modelos/modelo_formato_rosto_sem_linhas_branco.joblib` (0,85 MiB) é
  copiado para a imagem, ou montado em volume;
- o caminho vem de **setting do Django lido de variável de ambiente**, no mesmo
  padrão que o backend já usa para `RECOMMENDATION_SERVICE_TOKEN`.
  Sugestão de nome: `VIZO_ROSTO_MODELO_PATH`.

### 5.2 Uma vez no startup, não por requisição — e por quê

Medido neste repositório (`python -I`, Windows, SSD, Python 3.13.5):

| etapa | tempo |
|---|---|
| `import vizo_rosto` (arrasta sklearn, pandas, scipy) | **1 196 ms** |
| `carregar()` do `.joblib` (500 árvores) | **2 185 ms** |
| 1ª recomendação completa | 82 ms |
| recomendação, média de 50 | **41 ms** |

Carregar o modelo custa **2,2 s**. Se isso acontecer dentro da view, **todo**
`POST /measurement/` paga 2,2 s por uma coisa que não muda. O modelo é imutável e
sem estado: carregue uma vez por worker, em `AppConfig.ready()`, e injete o
objeto em quem recomenda — é justamente para isso que `recomendar()` recebe o
modelo por parâmetro em vez de carregá-lo lá dentro. O custo real por requisição
passa a ser os **41 ms** da recomendação.

Efeito colateral a conhecer: o startup de cada worker do gunicorn fica ~3,4 s
mais lento (1,2 s de import + 2,2 s de carga).

### 5.3 Se o arquivo não estiver lá

`carregar()` levanta `FileNotFoundError` **antes** de tentar desserializar,
com mensagem acionável. Saída real:

```
FileNotFoundError: Nao encontrei o modelo em 'modelos/nao-existe.joblib'. O
artefato versionado deste projeto fica em
modelos/modelo_formato_rosto_sem_linhas_branco.joblib, relativo a raiz do
repositorio. Confira o diretorio de trabalho.
```

Dois avisos para quem integrar:

- **`FileNotFoundError` não está na tabela de erros do §4 do briefing.** A tabela
  cobre `EntradaInvalidaError`, `FeatureNaoFinitaError`, `ModeloIncompativelError`,
  `FeaturesIncompativeisError` e `FormatoRostoSemRegraError`. O arquivo ausente é
  uma quinta situação, e é a **mais provável de todas** na primeira implantação.
  Classificação correta: **defeito de implantação**, o mesmo balde de
  `ModeloIncompativelError` — 500 e log, nunca erro atribuído ao cliente.
- Se o modelo for carregado no startup (§5.2), o arquivo ausente **derruba o
  worker na subida**, e não na primeira requisição. Isso é melhor: falha barulhenta
  e imediata em vez de uma medição aceita que não gera recomendação. Mas é uma
  escolha consciente, e interage com D-64: D-64 protege a medição contra falha
  **do recomendador**, não contra backend que não sobe.

O outro erro de implantação possível é apontar para o modelo errado:
`carregar()` compara `estimador.classes_` com as 4 classes esperadas e levanta
`ModeloIncompativelError` se não baterem. Essa verificação existe porque carregar
o modelo de **armação** achando que era o de rosto já aconteceu neste projeto, e
a saída daquela execução chegou a ser versionada.

---

## 6. Verificação feita, e com o quê

```
.venv/Scripts/python -m pytest tests/ -q
90 passed in 10.27s
```

68 testes que já existiam, mais 22 novos em `tests/test_contrato_backend.py`.
Entrada: as **78 linhas reais** de
`arquivos_csv/faces-teste-supervisionado-formato-rosto.csv`, pelas fixtures de
`conftest.py`. Nenhum rosto sintetizado, e **nenhuma geometria facial do autor
copiada para dentro de arquivo versionado** — este repositório tem remote
público.

O contrato foi conferido também nas quatro medições reais que o autor deixou na
raiz do repositório, **sem commitá-las** e sem copiar o conteúdo delas para
nenhum arquivo versionado:

| medição | `formato_rosto` | `ranking_armacoes` |
|---|---|---|
| `medicao-teste.json` | quadrado | redondo, ovalado, aviador, gatinho |
| `medicao-teste-2.json` | quadrado | redondo, ovalado, aviador, gatinho |
| `medicao-teste-3.json` | quadrado | redondo, ovalado, aviador, gatinho |
| `medicao-teste-4.json` | oval | quadrado, retangular, gatinho, aviador, ovalado, redondo |

Confere com o que o Maestro havia obtido pelo pipeline completo. Nas quatro, as
cinco chaves saíram exatas, o aviso chegou íntegro (460 caracteres) e o
round-trip por JSON foi idêntico.

### O que isto **não** prova

- **Não prova que o backend funciona.** Nada aqui executa Django, migração ou
  endpoint. Prova que o dicionário entregue na fronteira está correto e que
  nenhuma porcentagem o atravessa.
- **Não prova que o modelo acerta.** A acurácia continua 0,47 ± 0,08 em CV
  10-fold. O contrato transporta uma previsão que erra mais da metade das vezes,
  e isso não mudou nesta tarefa.
- **Não prova que a recomendação é correta no mérito.** As regras de visagismo
  são um rascunho não revisado pelo autor (D-44), e o resultado central do TCC é
  que **não existe** relação estatisticamente detectável entre formato de rosto e
  a armação que as pessoas usam (teste de permutação, p global = 0,176). A
  recomendação é prescritiva e declarada; o `aviso` que viaja no contrato existe
  justamente para que isso não se perca no caminho até o usuário.
- **Não prova o número de tamanho de imagem por `docker build`.** Os 318,55 MiB
  vêm da medição direta dos wheels manylinux `cp313` descompactados e
  bytecompilados, não de uma imagem construída — o Docker está desligado e
  construir não estava autorizado. O 1,98 GB da imagem atual é leitura de
  `docker images`.

---

## 7. Pontos em aberto, para decisão do autor

Nenhum destes foi implementado. Estão aqui porque são achados, não tarefas.

1. **`versao_contrato` versiona a forma, e o que vai mudar é o conteúdo.** Ele
   distingue formatos de dicionário. Mas a mudança que o projeto **planeja** é
   outra: o autor vai revisar a tabela de visagismo (D-44). Quando isso
   acontecer, o ranking gravado no banco muda e `versao_contrato` continua `1`,
   porque a forma não mudou. Resultado: recomendações geradas antes e depois da
   revisão ficam indistinguíveis no banco. O que resolveria é um segundo campo,
   versionando a **tabela de regras** — e isso mexe no contrato do §4, que foi
   fixado para permitir trabalho em paralelo, então é decisão do Maestro e do
   autor, não minha.
2. **O contrato não leva a justificativa nem a fonte.** O §4 traz só `posicao` e
   `formato`. A `justificativa` (o critério visagista daquela posição) e as
   `fontes` ficam de fora, então o app pode exibir "armação redonda" sem dizer
   por quê e sem citar de onde. Nenhuma das duas é número, então D-66 não as
   proíbe — é escolha de escopo. Tensiona com D-44, que exige fonte em cada
   regra. Seria contrato versão 2, aditivo.
3. **O `requirements.txt` do backend está em UTF-16LE**, com BOM `ff fe` (812
   bytes para 18 linhas). Acrescentar as quatro linhas em UTF-8 corrompe o
   arquivo e o `pip install` do `Dockerfile` falha. Quem editar precisa preservar
   a codificação ou converter o arquivo inteiro de uma vez. **É território do
   Alicerce** — não toquei, só li. Vale avisar antes de ele editar.
4. **`vizo_rosto` não tem `__version__`.** Sem isso, não há como registrar no
   banco qual versão do pacote gerou uma recomendação, nem casar pacote com
   `.joblib`. Relacionado ao ponto 1.
5. **Instalar por pip exigiria mexer em D-51.** Ver §4.3. Hoje o caminho é copiar
   o diretório e ajustar `PYTHONPATH`.
