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
  "provenance_notes": "opcional: de onde veio o caso, como o gabarito foi obtido"
}
```

- `gabarito_ids`: IDs de `data/riscos_base.csv` presentes no mapa de riscos
  real daquela contratacao. Pode conter IDs que nao existem na biblioteca
  carregada (ex.: risco relevante que a biblioteca ainda nao cobre) -- isso
  e reportado, nao descartado, pois limita o recall maximo alcancavel do
  caso por qualquer motor.
- `used_to_build_base`: **`true`** se este caso foi usado para construir ou
  revisar `data/riscos_base.csv` (por exemplo, um dos seus riscos foi
  copiado para a biblioteca). Esses casos sao automaticamente excluidos da
  avaliacao principal -- ver "Controle de vazamento" abaixo.

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

Um caso com `used_to_build_base: true` nunca entra na avaliacao principal:
se um risco do mapa real de um caso foi copiado para a biblioteca, qualquer
motor que também veja essa biblioteca "aprenderia" o próprio gabarito do
caso, inflando artificialmente a métrica. `evaluation/cases/provenance.py`
faz esse filtro e também registra:

- quais casos foram excluídos por esse motivo (`excluded_leaked_cases`);
- quais IDs de gabarito não existem na biblioteca carregada
  (`cases_with_unknown_gabarito_ids`), para transparência sobre o teto de
  recall de cada caso.

## Motores em detalhe

### `rules` -- motor atual

`evaluation/engines/rules_engine.py` usa `suggestion_score` (sem
modificação) para pontuar todos os riscos e ordena com o mesmo critério de
desempate de `suggest_risks` -- `(score, categoria, id)` decrescente -- mas
**sem** aplicar `minimum_score` nem o limite por categoria
(`MAX_SUGGESTIONS_PER_CATEGORY`), que no app são políticas de quantas
sugestões *mostrar*, não a qualidade do ranking em si. Isso faz o ranking
comparável aos outros motores (que também devolvem o ranking completo).

Para obter os números "como o usuário realmente vê" (com os cortes de
produção), use `rules_engine.rank_as_shown_in_production(...)` -- disponível
mas não usado no cálculo principal.

### `bm25`

Documento de cada risco = título + causa + consequência + palavras-chave +
tipos de contratação + categoria. Consulta = objeto + contexto + tipo de
contratação + criticidade + modalidade + prazo. Tokenização simples
(minúsculas, sem acento, tokens alfanuméricos) em `evaluation/text_utils.py`.
Parâmetros `k1`/`b` em `config.BM25Config` (padrão `k1=1.5`, `b=0.75`),
registrados em todo relatório.

### `embedding`

Mesmos textos de documento/consulta do BM25, mas comparados por
similaridade de cosseno entre embeddings. O backend de produção
(`SentenceTransformerBackend`) usa um **modelo fixo e versionado**
(`config.DEFAULT_EMBEDDING_MODEL_NAME` + `DEFAULT_EMBEDDING_MODEL_REVISION`,
hoje `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2@main`),
executado **localmente** via `sentence-transformers`
(`evaluation/requirements-embeddings.txt`, dependência opcional e pesada --
não é instalada por padrão). Nenhum texto sai da máquina.

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
(id, título, categoria, palavras-chave) e pede ao modelo uma lista JSON de
IDs ordenada por relevância. Qualquer ID que não esteja entre os candidatos
é descartado do ranking usado nas métricas de qualidade, mas contabilizado
na **taxa de IDs inválidos** (`metrics/validity.py`).

Endpoint padrão: `http://localhost:11434/v1/chat/completions` (compatível
com a API de chat da OpenAI -- Ollama, llama.cpp server, vLLM etc.). Um
endpoint que não resolva para `localhost`/`127.0.0.1`/`::1`/`*.localhost` é
**rejeitado** a menos que `allow_external=True` seja definido
explicitamente (`evaluation/config.py`, `LLMConfigError`). Nunca envia dados
a um serviço externo por padrão.

## Métricas (por caso e por motor)

Calculadas em `evaluation/metrics/`:

- **Precisão@k** e **Revocação@k**, para `k = 5` e `k = 10`
  (`metrics/ranking.py`).
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
(precisão@5/@10, revocação@5/@10, nDCG@10, MRR). Produz a diferença média
observada, um intervalo de confiança (padrão 95%, `config.RunConfig.bootstrap_confidence`)
e uma aproximação bootstrap do p-valor bicaudal. Implementado só com a
biblioteca padrão (`random`, `statistics`) -- sem numpy.

## Registro de versões, parâmetros e sementes

Todo relatório (`evaluation/run_metadata.py`) inclui: versão do Python,
plataforma, versões de pacotes relevantes instalados (`requests`,
`sentence-transformers`, `torch`, `pytest`), commit git do app no momento da
execução, e a configuração completa (`RunConfig`) -- sementes, parâmetros do
BM25, modelo/revision de embeddings, configuração do LLM (sem credenciais).

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

Para habilitar o motor LLM contra um servidor local já em execução:

```bash
python -m evaluation --enable-llm --llm-endpoint http://localhost:11434/v1/chat/completions --llm-model llama3.1:8b-instruct
```

## Testes

```bash
pip install -r requirements.txt   # pytest (requisito do app, ja usado pelos testes de evaluation)
pytest evaluation/tests
```

Os testes usam riscos e casos sintéticos (`evaluation/tests/conftest.py`,
`evaluation/cases/synthetic/`) e dublês locais para o backend de embeddings
(`HashingEmbeddingBackend`) e o cliente LLM (`FakeLLMClient`), então não
exigem rede, modelo baixado nem servidor LLM rodando.
