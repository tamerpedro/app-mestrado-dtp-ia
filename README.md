# MVP Matriz de Riscos TIC

Protótipo local para apoiar a elaboração, padronização, justificativa e revisão humana de matrizes de risco em contratações de TIC.

## Escopo do MVP

Esta primeira versão não tenta localizar automaticamente matrizes de risco em PDFs longos. O foco é apoiar uma nova contratação a partir de:

- cadastro estruturado da contratação;
- biblioteca curada de riscos;
- cálculo de nível de risco por probabilidade x impacto;
- sugestão automatizada simples por tipo de contratação, palavras-chave e criticidade;
- revisão humana;
- exportação em CSV, Excel, LaTeX ou Word no padrão do Mapa de Gerenciamento de Riscos.

## Como Rodar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Requer Streamlit 1.53 ou superior (a troca de idioma depende da identidade dos widgets por chave).

Para desenvolver e rodar os testes, instale `requirements-dev.txt` em vez de `requirements.txt` (inclui o pytest) e rode `pytest`.

## Idiomas / Languages

A interface está disponível em português e inglês, com seletor no topo da barra lateral. Para abrir direto em inglês, use `?lang=en` na URL. A troca de idioma preserva tudo o que foi preenchido.

The interface is available in Portuguese and English (selector at the top of the sidebar, or `?lang=en` in the URL).

Estado atual: interface e biblioteca de riscos bilíngues; a justificativa das sugestões e os arquivos exportados ainda estão em português (fases 4 e 5 do plano de internacionalização).

### Biblioteca bilíngue

`data/riscos_base.csv` tem colunas em inglês ao lado das originais: `titulo_en`, `causa_en`, `consequencia_en`, `acao_preventiva_en`, `acao_contingencia_en`, `responsavel_sugerido_en` e `palavras_chave_en`. Se uma coluna em inglês estiver vazia, o app mostra o texto em português (e vice-versa). Um risco salvo pela revisão humana é gravado nas colunas do idioma em que foi escrito. Na revisão humana, os textos vindos da biblioteca acompanham o idioma até serem editados; depois disso, o texto do usuário é mantido.

## Estrutura

```text
app.py                          Interface Streamlit
data/riscos_base.csv            Biblioteca inicial de riscos
data/casos/                     Casos compartilhados (contexto + cache de riscos de IA por caso/idioma)
locales/pt.json                 Textos da interface em português
locales/en.json                 Textos da interface em inglês
prompts/gerar_risco_v2.txt      Prompt versionado usado para gerar os riscos de IA (v1 mantido para referência)
requirements.txt                Dependências de produção (deploy no Streamlit Cloud)
requirements-dev.txt            + pytest, para desenvolvimento local
.streamlit/secrets.example.toml Modelo dos secrets (AI_PROVIDER, AI_MODEL, chaves, MODO_AVALIACAO)
scripts/consolidar_avaliacao.py Consolidação dos logs de avaliação (resumo por processo e por categoria)
src/domain.py                   Códigos de domínio e rótulos por idioma
src/i18n.py                     Função t() de tradução
src/ai_provider.py              Provedor de IA (Anthropic / compatível OpenAI / fake para testes)
src/eval_mode.py                Flag MODO_AVALIACAO e hash da biblioteca
src/risk_generator.py           Prompt e validação do risco gerado por IA
src/catalog_mix.py              Lista cega catálogo+IA por categoria (sem pontuação nem pareamento)
src/case_store.py               Casos compartilhados e cache de riscos de IA por caso/idioma
src/eval_log.py                 Schema do log de avaliação, montagem (build_evaluation_log) e gravação local opcional
src/                            Regras, modelos e exportadores
tests/                          Testes da lógica central, da troca de idioma e da avaliação
```

Os logs de avaliação (`log_<código>_<sessão>.json`) são baixados pelo participante na aba "Exportação" — nada fica salvo no disco da aplicação (veja "Rodar no Streamlit Cloud" abaixo). Para consolidar, junte os arquivos recebidos numa pasta local qualquer (ex.: `logs_recebidos/`) e aponte `--log-dir` para ela.

Para adicionar ou alterar um texto da interface, edite a mesma chave em `locales/pt.json` e `locales/en.json`; o teste `tests/test_i18n.py` falha se as chaves divergirem.

## Avaliação com catálogo e IA

Segundo ciclo de avaliação do estudo (Design Science Research): em cada categoria, a interface mostra todo o catálogo filtrado pelo tipo de contratação (sem pontuação nem ranqueamento) misturado a `min(3, n)` riscos gerados por IA (`n` = total de riscos do catálogo na categoria antes do filtro), em uma única lista às cegas com IDs neutros (`S01`, `S02`...) — a pessoa que avalia escolhe quantos quiser, sem saber a origem de cada um. A avaliação mede **quanto desse processo é adotado e qual a participação da IA na matriz final**, e não é uma comparação ou competição entre a taxa da IA e a do catálogo.

### Configurar a chave de IA

Não há LLM local no ambiente; defina o provedor e a chave por variável de ambiente ou em `.streamlit/secrets.toml` (nunca comitado; já está no `.gitignore`):

```toml
# .streamlit/secrets.toml
AI_PROVIDER = "anthropic"       # ou "openai"
AI_MODEL = "claude-sonnet-5-5"
ANTHROPIC_API_KEY = "sk-ant-..."
# Para um provedor compativel com OpenAI:
# OPENAI_API_KEY = "sk-..."
# OPENAI_BASE_URL = "https://..."
```

Ou por variável de ambiente (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
streamlit run app.py
```

Sem chave configurada (ou se a chamada falhar), o app mostra um aviso e segue a avaliação só com o catálogo — nenhuma sugestão de IA é exibida nas categorias afetadas, e a falha fica registrada no log exportado. O prompt usado fica versionado em `prompts/gerar_risco_v2.txt`; a versão (`v2`) é gravada em cada log.

### Criar um caso compartilhado

Para que todos os participantes vejam exatamente os mesmos riscos de IA, crie um JSON em `data/casos/` com os campos da contratação e um `"nome"` (veja `data/casos/exemplo.json`). Na primeira geração de sugestões para esse caso, os riscos de IA de cada categoria são cacheados em `data/casos/<arquivo>_ia_<idioma>.json`; execuções seguintes reusam o cache, sem chamar a IA de novo. Sem caso selecionado ("Nenhum, preencher manualmente"), a geração acontece a cada clique, sem cache.

### Rodar a avaliação

```powershell
streamlit run app.py
```

Na barra lateral, informe um **código do processo** (texto livre, sem dados pessoais), escolha um **caso** (ou preencha a contratação manualmente) e clique em **"Gerar sugestões"** — só então a IA é chamada; abrir a tela ou editar um campo não gera nada sozinho. Na aba "Sugestões", marque quantos itens quiser por categoria (lista às cegas, catálogo+IA misturados). Na aba "Revisão humana" aparecem só os itens escolhidos (mais os riscos manuais), para editar livremente; desmarcar um item na aba "Sugestões" o remove da revisão, e marcá-lo de novo preserva a edição anterior. Na aba "Exportação", além da matriz final e dos downloads, a seção **"Resultado da avaliação"** mostra a origem de cada risco (catálogo/IA/manual), a composição da matriz, a adoção das sugestões de IA e os editados por origem; o formulário final pede a **experiência com contratações de TIC** (obrigatória) e as 3 perguntas de escala 1-5 + um comentário opcional. **"Exportar log da avaliação (.json)"** baixa o log do processo (`log_<código>_<sessão>.json`) com o estado atual — nada é gravado no servidor (o disco do Streamlit Cloud é efêmero); ao terminar, cada participante envia esse arquivo ao pesquisador.

Durante a avaliação (`MODO_AVALIACAO`, ligado por padrão nesta branch), o botão de salvar manualmente na biblioteca (revisão humana) fica desativado, para não alterar `data/riscos_base.csv` durante a coleta. Para desligar, defina `MODO_AVALIACAO=false` por variável de ambiente ou em `secrets.toml`.

### Consolidar os resultados

```powershell
python scripts/consolidar_avaliacao.py
```

Junte os arquivos `log_*.json` baixados pelos participantes numa pasta (ex.: `logs_recebidos/`) e aponte `--log-dir` para ela. O script lê todos os JSON de schema v2 ali dentro (logs de schema antigo são pulados com um aviso), gera `resumo_processos.csv` (uma linha por processo: código, caso, experiência, mostrados/escolhidos por origem, tamanho da matriz, participação percentual por origem, adoção da IA, editados por origem, riscos manuais, respostas do formulário) e `resumo_categorias.csv` (composição da matriz final por categoria, somada entre todos os processos) na raiz do projeto. No terminal, imprime um relatório **geral e por faixa de experiência** com o número de processos, o tamanho médio da matriz, a participação média de catálogo/IA/manual, a adoção média da IA, a média de riscos manuais e as respostas médias do formulário — só estatística descritiva, sem comparação direta entre a taxa da IA e a do catálogo.

```powershell
python scripts/consolidar_avaliacao.py --log-dir logs_recebidos --out-dir .
```

## Rodar no Streamlit Cloud

1. No [Streamlit Community Cloud](https://share.streamlit.io), crie um app a partir do repositório `tamerpedro/app-mestrado-dtp-ia`, branch `feature/catalogo-misto-ia` (ou a branch de avaliação em uso), arquivo principal `app.py`.
2. Em "Settings → Secrets" do app, cole o conteúdo de `.streamlit/secrets.example.toml` já preenchido com os valores reais (`AI_PROVIDER`, `AI_MODEL`, `ANTHROPIC_API_KEY` e, se quiser, `MODO_AVALIACAO`) — nunca comite esse arquivo preenchido.
3. O caso de exemplo (`data/casos/exemplo.json`) já sobe com o cache de riscos de IA (`data/casos/exemplo_ia_pt.json`) versionado, então participantes que usarem esse caso não dependem de uma chamada de IA bem-sucedida no deploy.
4. O disco do Streamlit Cloud é efêmero (reinicia a cada deploy/sleep): o log de cada avaliação **não fica salvo no servidor** — cada participante baixa o próprio log pela aba "Exportação" (`log_<código>_<sessão>.json`) e envia o arquivo ao pesquisador (e-mail, upload etc.).
5. Para consolidar: junte os arquivos recebidos numa pasta local e rode `python scripts/consolidar_avaliacao.py --log-dir <pasta>` (veja "Consolidar os resultados" acima).

## Decisao de Projeto

A extracao automatica de conhecimento de processos antigos fica como evolucao futura. O MVP concentra-se na operacionalizacao assistida de conhecimento ja estruturado, com revisao humana obrigatoria e exportacao da matriz final.

## Saida Word

A exportacao Word segue o formato institucional observado no mapa de riscos de referencia: capa, historico, orientacoes, grupos de riscos, tabela individual por risco, escala 1-5 de probabilidade e impacto, nivel calculado, estrategia, consequencias, acoes preventivas, acoes de contingencia, responsavel por acao e anexos de escala.
