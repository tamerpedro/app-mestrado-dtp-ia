# Avaliacao comparativa dos motores de sugestao de riscos

Este pacote mede a qualidade das sugestoes de risco em contratacoes reais de
TIC, comparando quatro motores:

1. **`rules`** -- o motor atual de regras do app (`suggestion_score` /
   `suggest_risks` em `src/suggestions.py`), usado aqui **somente leitura**.
   Nada em `src/` e alterado por este pacote.
2. **`bm25`** -- BM25 Okapi, determinístico e auditável, implementado em
   Python puro (`evaluation/engines/bm25_engine.py`), sem dependencias
   externas.
3. **`embedding`** -- similaridade de cosseno sobre embeddings de um
   **modelo fixo, executado localmente** (`evaluation/engines/embedding_engine.py`).
4. **`llm`** -- um LLM **restrito ao repositorio**: só pode "escolher" IDs
   de risco que existem na biblioteca carregada; qualquer ID fora disso e
   descartado e contabilizado na taxa de IDs invalidos. **Desligado por
   padrao** e só aceita endpoint local, a menos que `allow_external=True`
   seja setado explicitamente (nunca por padrao).

`evaluation/` tem dependencias proprias (`evaluation/requirements.txt`,
`evaluation/requirements-embeddings.txt`), separadas de `requirements.txt`
do app, e não altera o comportamento do app nem de `src/suggestions.py`.
Tambem inclui uma interface Streamlit **separada** de `app.py`
(`evaluation/ui.py`) para cadastrar casos, anotar o gabarito e rodar a
avaliacao sem usar a linha de comando -- ver "Interface Streamlit" abaixo.

## Entrada dos motores

Todos os motores recebem **apenas os campos de `ContractContext`**
(`objeto`, `tipo_contratacao`, `valor_estimado`, `criticidade`, `prazo`,
`modalidade`, `contexto`) e a lista de riscos da biblioteca. O gabarito de
cada caso é usado só para calcular métricas, nunca como entrada de um
motor.

## Formato de um caso (`evaluation/cases/*.json`)

```json
{
  "case_id": "identificador unico do caso",
  "context": {
    "objeto": "...", "tipo_contratacao": "...", "valor_estimado": 0.0,
    "criticidade": "...", "prazo": "...", "modalidade": "...", "contexto": "..."
  },
  "gabarito_ids": ["R001", "R006"],
  "used_to_build_base": false,
  "provenance_notes": "opcional: de onde veio o caso, como o gabarito foi obtido",
  "context_predates_risk_map": true,
  "annotator_count": 2
}
```

- `gabarito_ids`: IDs de `data/riscos_base.csv` presentes no mapa de riscos
  real daquela contratacao. Pode conter IDs que nao existem na biblioteca
  carregada (ex.: risco relevante que a biblioteca ainda nao cobre) -- isso
  e reportado, nao descartado, pois limita o recall maximo alcancavel do
  caso por qualquer motor.
- `used_to_build_base`: **`true`** se este caso alimentou a construcao **ou
  a revisao/ajuste** de `data/riscos_base.csv`, de qualquer forma -- nao so
  "um risco foi copiado literalmente para a biblioteca". Conta tambem se o
  caso foi usado para: revisar o *texto* de um risco existente (titulo,
  causa, consequencia, acoes); calibrar suas *palavras-chave*; calibrar seus
  *pesos* (`probabilidade_padrao` / `impacto_padrao`); ou calibrar os
  *limiares* do motor de regras (`minimum_score`, limite por categoria) ou
  parametros equivalentes de outro motor. Qualquer uma dessas formas de
  influencia vaza o gabarito do caso para dentro dos motores. Esses casos
  sao automaticamente excluidos da avaliacao principal -- ver "Controle de
  vazamento" abaixo. Em caso de dúvida sobre se um caso se qualifica, marque
  `true` (o padrão seguro é excluir, nunca incluir por engano).
- `context_predates_risk_map` (opcional, recomendado): `true` se o `context`
  foi extraido de documentos **anteriores** ao mapa de riscos final (termo
  de referencia, estudo tecnico preliminar, edital) -- nunca do proprio mapa
  ou de material produzido depois dele. Ver "Proveniencia do contexto e do
  gabarito" abaixo.
- `annotator_count` (opcional, recomendado): quantas pessoas anotaram o
  gabarito de forma independente e sem acesso as sugestoes de nenhum motor.
  Preferencialmente `>= 2`.

Os dois ultimos campos sao opcionais (casos antigos continuam carregando sem
eles) mas fortemente recomendados: sua ausencia gera um aviso informativo em
`ProvenanceReport.methodology_warnings`, que aparece no relatorio mas nao
bloqueia a avaliacao.

Veja `evaluation/cases/synthetic/case_example_001.json` para um exemplo
completo (fictício).

### Onde ficam os casos reais

- `evaluation/cases/real/` -- casos reais, **nao versionados**
  (`.gitignore` ignora todo o conteudo exceto `.gitkeep`). Coloque ali um
  `.json` por caso, no formato acima, com dados de contratacoes reais.
- `evaluation/cases/synthetic/` -- casos sinteticos/ficticios, versionados;
  usados nos testes automatizados e como exemplo de formato. **Nunca** usar
  dados reais aqui.

## Controle de vazamento (proveniencia)

Um caso com `used_to_build_base: true` nunca entra na avaliacao principal.
Isso cobre nao so "um risco do mapa real foi copiado para a biblioteca", mas
qualquer caso usado para revisar o texto, as palavras-chave, os pesos ou os
limiares de um risco -- qualquer uma dessas formas de influencia faz o motor
"aprender", via a biblioteca ou seus parâmetros, o próprio gabarito do caso,
inflando artificialmente a métrica (ver docstring de `EvalCase.used_to_build_base`
em `cases/schema.py`). `evaluation/cases/provenance.py` faz esse filtro e
também registra:

- quais casos foram excluídos por esse motivo (`excluded_leaked_cases`);
- quais IDs de gabarito não existem na biblioteca carregada
  (`cases_with_unknown_gabarito_ids`), para transparência sobre o teto de
  recall de cada caso;
- avisos informativos de metodologia de anotação não seguida/informada por
  caso (`methodology_warnings`) -- ver seção "Proveniência do contexto e do
  gabarito" abaixo.

## Proveniência do contexto e do gabarito

Dois cuidados metodológicos, para evitar vazamento por retrospectiva
(*hindsight bias*) e ancoragem na anotação:

1. **O `context` de um caso deve vir de documentos anteriores ao mapa de
   riscos final** (termo de referência, estudo técnico preliminar, edital
   etc.), nunca do próprio mapa de riscos ou de material produzido depois
   dele. Se o contexto já incorpora conhecimento do mapa final, qualquer
   motor "vê" pistas que não existiam no momento real da decisão, inflando
   a qualidade aparente das sugestões. Registre isso em
   `context_predates_risk_map: true` no caso.
2. **O gabarito deve ser anotado sem acesso às saídas de nenhum motor**
   (nem do motor atual de regras, nem dos demais) -- quem anota a lista de
   IDs relevantes não deve ver sugestões automáticas antes de decidir,
   senão a anotação ancora no que o motor já sugeriu. Preferencialmente,
   use **dois anotadores independentes** e registre `annotator_count: 2`
   (ou mais) -- isso permite checar concordância entre anotadores, o que
   este pacote não calcula automaticamente hoje, mas que o campo deixa
   registrado para análise manual.

Casos sem esses campos continuam sendo aceitos (compatibilidade com casos
já coletados), mas geram aviso em `ProvenanceReport.methodology_warnings`,
incluído no relatório (`summary.md` e `report.json`).

## Motores em detalhe

### `rules` -- motor atual

`evaluation/engines/rules_engine.py` usa `suggestion_score` (sem
modificação) para pontuar todos os riscos e ordena com o mesmo critério de
desempate de `suggest_risks` -- `(score, categoria, id)` decrescente -- mas
**sem** aplicar `minimum_score` nem o limite por categoria
(`MAX_SUGGESTIONS_PER_CATEGORY`), que no app são políticas de quantas
sugestões *mostrar*, não a qualidade do ranking em si. Isso faz o ranking
comparável aos outros motores (que também devolvem o ranking completo) e é
o que entra no bootstrap pareado como linha de base.

Além desse ranking completo, **todo relatório também inclui `rules_production`**:
o mesmo motor `rules`, mas com os cortes de produção aplicados
(`rules_engine.rank_as_shown_in_production`, com `minimum_score` e o limite
por categoria) -- ou seja, exatamente o que o usuário vê hoje no app. É
calculado automaticamente sempre que `rules` roda (`runner.PRODUCTION_RULES_ENGINE`)
e aparece como uma linha adicional na tabela do relatório, mas **fica fora**
do bootstrap pareado (não é um motor candidato a substituir o atual).

> Atenção ao interpretar precisão@k de `rules_production`: como o motor
> pode devolver menos de *k* sugestões (por causa dos cortes), precisão@k é
> sempre calculada dividindo por *k* (não pelo tamanho da lista devolvida) --
> ver `metrics/ranking.py::precision_at_k` -- para não inflar artificialmente
> a métrica de um motor que decidiu sugerir pouco.

### Texto por risco, idêntico nos três motores de ranking por texto

`bm25`, `embedding` e `llm` usam exatamente o mesmo texto por risco --
`text_utils.risk_document_text`: título + causa + consequência +
palavras-chave + tipos de contratação + categoria -- e a mesma consulta a
partir do contexto -- `text_utils.context_query_text`: objeto + contexto +
tipo de contratação + criticidade + modalidade + prazo. Isso garante que os
três vejam exatamente a mesma informação sobre cada risco; nenhum motor tem
acesso a um resumo mais rico (ou mais pobre) que outro. `bm25_engine.py` e
`embedding_engine.py` tokenizam esse texto (minúsculas, sem acento, tokens
alfanuméricos); `llm_engine.py` o usa como está, por candidato, no prompt.

### `bm25`

Parâmetros `k1`/`b` em `config.BM25Config` (padrão `k1=1.5`, `b=0.75`),
registrados em todo relatório.

### `embedding`

Comparação por similaridade de cosseno entre embeddings do texto acima. O
backend de produção (`SentenceTransformerBackend`) usa um **modelo fixo,
pinado em um hash de commit** (nunca em uma branch como `"main"`, cujo
conteúdo pode mudar sob o mesmo nome) -- `config.DEFAULT_EMBEDDING_MODEL_NAME`
+ `DEFAULT_EMBEDDING_MODEL_REVISION`, hoje
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2@e8f8c211226b894fcb81acc59f3b34ba3efd5f42`
-- executado **localmente** via `sentence-transformers`
(`evaluation/requirements-embeddings.txt`, dependência opcional e pesada --
não é instalada por padrão). Nenhum texto sai da máquina. Trocar o modelo
exige editar essas duas constantes em `config.py` e registrar o motivo no
commit -- nunca silenciosamente.

Para testes automatizados e uso offline sem baixar o modelo, existe
`HashingEmbeddingBackend`: um backend local, determinístico, sem
dependências, baseado em *hashing trick* sobre os tokens. **Não é um
embedding semântico real** e não deve ser usado para gerar os números finais
da avaliação -- apenas para exercitar a lógica de ranking por similaridade
nos testes. `python -m evaluation --embedding-backend hashing` deixa isso
explícito; o padrão da CLI é `sentence-transformers`.

### `llm`

Desligado por padrão (`config.LLMConfig.enabled=False`). Quando habilitado,
monta um prompt com os campos do `ContractContext` e a lista de candidatos
(id + o mesmo texto de risco usado pelo BM25/embeddings) e pede ao modelo
uma lista JSON de IDs ordenada por relevância. Qualquer ID que não esteja
entre os candidatos é descartado do ranking usado nas métricas de
qualidade, mas contabilizado na **taxa de IDs inválidos**
(`metrics/validity.py`).

Endpoint padrão: `http://localhost:11434/v1/chat/completions` (compatível
com a API de chat da OpenAI -- Ollama, llama.cpp server, vLLM etc.). Um
endpoint que não resolva para `localhost`/`127.0.0.1`/`::1`/`*.localhost` é
**rejeitado** a menos que `allow_external=True` seja definido
explicitamente (`evaluation/config.py`, `LLMConfigError`). Nunca envia dados
a um serviço externo por padrão.

**Modelo e timeout padrão.** `config.LLMConfig.model` é `"llama3.1:8b"` e
`config.LLMConfig.timeout_s` é `300` segundos -- um modelo local rodando em
CPU, sem GPU dedicada, pode demorar bem mais que os 30s de um timeout HTTP
"normal" para gerar a resposta completa; um timeout curto derrubava a
chamada antes do modelo terminar e inflava a contagem de falhas abaixo sem
o modelo ter de fato falhado. Ambos são editáveis: via `LLMConfig(model=...,
timeout_s=...)` no código, via `--llm-model`/`--llm-endpoint` na CLI
(`timeout_s` só pela CLI indiretamente, construindo um `RunConfig` próprio),
ou pelos campos "Modelo" e "Timeout (s)" na aba Avaliação da interface
(`evaluation/ui.py`).

**Identificador exato do modelo.** `config.LLMConfig.model` é só a tag que
*pedimos* ao servidor (ex.: `"llama3.1:8b"`); o servidor pode resolvê-la
para uma versão/digest mais específica. `LocalHTTPLLMClient` captura o
campo `"model"` da própria resposta da API (compatível com a da OpenAI) em
`last_model_id`, e `llm_engine.rank()` o registra em
`RankResult.reported_model_id`. O relatório traz isso em dois níveis: por
caso (`CaseEngineRecord.llm_reported_model_id`) e agregado
(`EngineAggregate.distinct_llm_reported_model_ids`, os valores distintos
vistos na execução) -- assim o relatório audita o modelo que *de fato*
respondeu, não só o nome configurado. Um cliente customizado que não exponha
`last_model_id` simplesmente não preenche esse campo (fica `None`).

**Resposta vazia ou JSON inválido nunca é silenciosa.** Se o servidor
devolver uma resposta vazia, ou um texto que não dá para interpretar como
uma lista JSON de IDs, o ranking daquele caso fica vazio -- mas isso é
*sempre* registrado, não confundido com "o motor decidiu que não há risco
relevante" (que seria uma resposta válida `"[]"`). `llm_engine._parse_ids`
devolve essa distinção (`RankResult.response_parse_failed`); o runner conta
quantos casos tiveram essa falha por execução
(`EngineAggregate.llm_response_parse_failure_count`) e marca cada caso
afetado (`CaseEngineRecord.llm_response_parse_failed`). O relatório
(`summary.md`, `report.json`) e a aba Avaliação da interface mostram essa
contagem com um aviso explícito -- nunca aparecem só como "0 riscos
encontrados" indistinguível de uma resposta legítima.

**Rodando com Ollama localmente.** O Ollama trunca o contexto em 2048
tokens por padrão, o que é pouco para o prompt desta avaliação (contexto da
contratação + até 62 candidatos com título, causa, consequência e
palavras-chave) -- a resposta pode sair cortada ou vazia bem antes do
modelo "decidir" algo, contando como falha de resposta na métrica acima.
Suba o servidor com uma janela de contexto maior, por exemplo:

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

(no Windows, via PowerShell: `$env:OLLAMA_CONTEXT_LENGTH=8192; ollama serve`).
Ajuste o valor para cima se `config.LLMConfig.max_candidates` for aumentado
ou se os riscos da biblioteca tiverem textos muito mais longos.

## Métrica principal e métricas exploratórias

Antes de rodar com dados reais, **`config.PRIMARY_METRIC = "recall_at_10"`
(Revocação@10) foi definida como a métrica principal** -- a única usada
para decidir se um motor candidato substitui o motor atual de regras. A
escolha é pré-registrada (decidida antes de olhar os números) para evitar
"metric shopping" -- escolher depois a métrica que favorece o resultado
desejado. A justificativa: numa biblioteca curada e pequena (62 riscos),
*não* sugerir um risco relevante (falso negativo) é o erro mais caro; o
excesso de sugestões é filtrado pela revisão humana de qualquer forma, e
Revocação@10 é a métrica mais sensível a esse tipo de erro na janela que o
app efetivamente mostra.

Todas as demais métricas abaixo são **exploratórias**: ajudam a entender o
comportamento de cada motor (ex.: um motor pode ter Revocação@10 parecida
mas Precisão@5 muito pior, sinal de que ele acerta mas enterra os riscos
certos mais abaixo no ranking), mas não decidem a troca. O relatório
(`summary.md`, `report.json`) rotula explicitamente qual é qual --
`report["primary_metric"]` / `report["exploratory_metrics"]` no JSON.

## Métricas (por caso e por motor)

Calculadas em `evaluation/metrics/`:

- **Precisão@k** e **Revocação@k**, para `k = 5` e `k = 10`
  (`metrics/ranking.py`). Precisão@k é sempre dividida por `k` (não pelo
  tamanho da lista devolvida), para não inflar a métrica de um motor que
  devolve menos de `k` itens (caso de `rules_production`, ver abaixo).
- **nDCG@10** (ganho binário, desconto logarítmico padrão).
- **MRR** (posição do primeiro item relevante, sem corte).
- **Taxa de IDs inválidos** -- fração dos IDs devolvidos pelo motor antes da
  validação que não existem na biblioteca carregada (`metrics/validity.py`).
  Sempre 0 para `rules`/`bm25`/`embedding` (só escolhem entre riscos reais);
  mede a disciplina do motor `llm`.
- **Reprodutibilidade** -- roda o mesmo motor `reproducibility_repeats`
  vezes (padrão 3) sobre o mesmo caso e compara os rankings: taxa de
  correspondência exata e sobreposição de Jaccard do top-10
  (`metrics/reproducibility.py`). Motores determinísticos devem chegar a
  1.0; o motor `llm` evidencia aqui se está variando entre chamadas.
- **Tempo de execução** -- medido com `time.perf_counter()` ao redor de cada
  chamada de ranking.

## Comparação com o motor atual: bootstrap pareado

`evaluation/metrics/bootstrap.py` implementa um bootstrap pareado por caso
(reamostragem com reposição dos *casos*, preservando o pareamento com o
motor `rules`) para cada métrica em `runner.BOOTSTRAP_METRICS`
(precisão@5/@10, revocação@5/@10, nDCG@10, MRR) -- sempre com a métrica
principal (`recall_at_10`) primeiro, as demais em seguida, marcadas como
exploratórias no relatório (`summary.md`). `rules_production` fica fora
dessa comparação (ver seção `rules` acima). Produz a diferença média
observada, um intervalo de confiança (padrão 95%, `config.RunConfig.bootstrap_confidence`)
e uma aproximação bootstrap do p-valor bicaudal. Implementado só com a
biblioteca padrão (`random`, `statistics`) -- sem numpy.

## Registro de versões, parâmetros e sementes

Todo relatório (`evaluation/run_metadata.py`) inclui: versão do Python,
plataforma, versões de pacotes relevantes instalados (`requests`,
`sentence-transformers`, `torch`, `pytest`), commit git do app no momento da
execução, e a configuração completa (`RunConfig`) -- sementes, parâmetros do
BM25, modelo/revision (hash de commit) de embeddings, a tag configurada do
LLM (sem credenciais) e a métrica principal. O identificador *exato* do
modelo LLM que respondeu (podendo diferir da tag configurada) é registrado
separadamente, por caso e agregado -- ver "Identificador exato do modelo"
na seção `llm` acima.

## Rodando a avaliação

```bash
# Nucleo (motores de regras e BM25, metricas, bootstrap): sem instalacao extra.
# Para o motor de embeddings "de producao":
pip install -r evaluation/requirements-embeddings.txt
# Para o motor LLM com o cliente HTTP padrao:
pip install -r evaluation/requirements.txt

# Coloque seus casos reais em evaluation/cases/real/*.json (ver formato acima), depois:
python -m evaluation
```

Sem casos em `evaluation/cases/real/`, use `--include-synthetic` para uma
demonstração com o caso de exemplo (não produz números representativos,
é só 1 caso fictício). O relatório é gravado em
`evaluation/reports/<timestamp>/` (`report.json`, `per_case_metrics.csv`,
`summary.md`) -- a pasta inteira é ignorada pelo git.

```bash
python -m evaluation --include-synthetic --embedding-backend hashing
```

Para habilitar o motor LLM contra um servidor local já em execução (ver
"Rodando com Ollama localmente" acima para `OLLAMA_CONTEXT_LENGTH`):

```bash
python -m evaluation --enable-llm --llm-endpoint http://localhost:11434/v1/chat/completions --llm-model llama3.1:8b
```

## Interface Streamlit (`evaluation/ui.py`)

App Streamlit **separado** de `app.py`/`src/` (nao altera nada ali) para
cadastrar casos, anotar o gabarito e rodar a avaliacao sem usar a linha de
comando. Reutiliza o schema (`cases/schema.py`), o loader (`cases/loader.py`)
e o runner (`runner.py`) existentes -- nao duplica validacao nem a logica
dos motores.

```bash
streamlit run evaluation/ui.py --server.address 127.0.0.1
```

`--server.address 127.0.0.1` restringe o servidor a conexoes locais. **Isto
nao e cosmetico: nunca rode esta interface numa instancia publica, acessivel
pela rede, ou com `--server.address 0.0.0.0`, quando houver dados reais em
`evaluation/cases/real/` ou em trafego com o motor LLM.** A interface nao
tem autenticacao nem controle de acesso -- qualquer um que alcance o
endereço/porta ve e edita os casos. Dados reais de contratacoes sao
informacao sensivel (ver `.gitignore`, que ja os mantem fora do controle de
versao); o mesmo cuidado vale para onde a interface roda.

Três abas:

1. **Casos** -- cadastra, edita e lista contratações (`ContractContext`,
   `case_id`, `used_to_build_base`, `context_predates_risk_map`,
   `annotator_count`, `provenance_notes`). Salvar grava um JSON por caso em
   `evaluation/cases/real/`, validado pelo mesmo `cases.schema.case_from_dict`
   usado pelo loader/runner -- um caso invalido (ex.: sem gabarito ainda)
   nunca é escrito em disco; a interface mostra a mensagem de validação e
   mantém o rascunho em memória para continuar editando.
2. **Gabarito** -- marca, para o caso carregado, quais riscos de
   `data/riscos_base.csv` aparecem no mapa real, com busca por ID, título ou
   categoria. **Nunca mostra pontuação ou sugestão de nenhum motor** --
   só os riscos da biblioteca e checkboxes, para manter a anotação cega (ver
   "Proveniência do contexto e do gabarito" acima).
3. **Avaliação** -- escolhe os motores (`rules` sempre inclui
   `rules_production` automaticamente, como no resto do protocolo), roda
   `evaluation.runner.run_evaluation` sobre os casos reais cadastrados e
   exibe: o resumo por motor (com a métrica principal destacada), os
   intervalos de confiança do bootstrap pareado, os avisos de metodologia de
   anotação e, por caso e motor, os riscos encontrados e perdidos no top 10.
   Ao escolher o backend `hashing` para embeddings, a interface mostra um
   aviso de que os números dessa execução não são finais nem
   representativos. O motor LLM continua desligado por padrão e só aceita
   endpoint local, a menos que "permitir endpoint externo" seja marcado
   explicitamente -- a mesma validação de `LLMConfig` (ver seção `llm`
   acima), não duplicada na interface. Quando habilitado, o campo "Timeout
   (s)" permite ajustar `LLMConfig.timeout_s` (padrão 300) sem editar
   código. Se algum caso tiver resposta vazia ou JSON inválido do LLM, a
   interface mostra a contagem de falhas por motor e marca o caso
   especificamente na lista de riscos encontrados/perdidos (ver "Resposta
   vazia ou JSON inválido nunca é silenciosa" na seção `llm` acima).

## Testes

```bash
pip install -r requirements.txt   # pytest e streamlit (requisitos do app, ja usados pelos testes de evaluation)
pytest evaluation/tests
```

Os testes usam riscos e casos sintéticos (`evaluation/tests/conftest.py`,
`evaluation/cases/synthetic/`) e dublês locais para o backend de embeddings
(`HashingEmbeddingBackend`) e o cliente LLM (`FakeLLMClient`), então não
exigem rede, modelo baixado nem servidor LLM rodando.
`evaluation/tests/test_ui.py` testa só a lógica pura de `ui.py` (validação,
persistência, filtro de riscos, montagem de `RunConfig`) -- as funções que
chamam a API do Streamlit (`render_*`/`main`) dependem de um runtime de app
que não existe sob pytest e não são exercitadas por esses testes; foram
verificadas manualmente rodando a interface de ponta a ponta (cadastro de
caso, anotação de gabarito, execução da avaliação).
