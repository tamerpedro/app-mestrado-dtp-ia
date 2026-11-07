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
data/candidatos_ia.csv          Riscos de IA escolhidos (gerado em runtime; fora do versionamento)
data/avaliacao/                 Log por processo da avaliação cega (gerado em runtime; fora do versionamento)
locales/pt.json                 Textos da interface em português
locales/en.json                 Textos da interface em inglês
prompts/gerar_risco_v1.txt      Prompt versionado usado para gerar o risco de IA
scripts/consolidar_avaliacao.py Consolidação dos logs de avaliação (CSVs + teste de McNemar)
src/domain.py                   Códigos de domínio e rótulos por idioma
src/i18n.py                     Função t() de tradução
src/ai_provider.py              Provedor de IA (Anthropic / compatível OpenAI / fake para testes)
src/eval_mode.py                Flag MODO_AVALIACAO e hash da biblioteca
src/risk_generator.py           Prompt e validação do risco gerado por IA
src/pairing.py                  Pareamento 1 biblioteca + 1 IA por categoria (avaliação cega)
src/eval_log.py                 Schema e gravação do log de avaliação por processo
src/                            Regras, modelos e exportadores
tests/                          Testes da lógica central, da troca de idioma e da avaliação cega
```

Para adicionar ou alterar um texto da interface, edite a mesma chave em `locales/pt.json` e `locales/en.json`; o teste `tests/test_i18n.py` falha se as chaves divergirem.

## Avaliação cega com IA

Segundo ciclo de avaliação do estudo (Design Science Research): em cada categoria com risco na biblioteca, a interface mostra esse risco (o de maior pontuação, mesmo abaixo do corte atual de sugestão) e 1 gerado por IA lado a lado, sem indicar qual é qual (IDs neutros `S01`, `S02`...). `sem_par` só ocorre quando a categoria não tem nenhum risco na biblioteca. Ao gerar a matriz final, o app revela a origem de cada risco escolhido e coleta uma avaliação curta.

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

Sem chave configurada (ou se a chamada falhar), o app mostra um aviso e segue a avaliação só com a biblioteca — nenhuma sugestão de IA é exibida nas categorias afetadas, e a falha fica registrada no log. O prompt usado fica versionado em `prompts/gerar_risco_v1.txt`; a versão (`v1`) é gravada em cada log.

### Rodar a avaliação

```powershell
streamlit run app.py
```

Na barra lateral, informe um **código do processo** (texto livre, sem dados pessoais), preencha os dados da contratação e clique em **"Gerar sugestões"** — só então a IA é chamada; abrir a tela ou editar um campo não gera nada sozinho. Clicar de novo refaz o pareamento (nova chamada à IA e novo sorteio de ordem) a partir dos dados atuais. Na aba "Sugestões", cada par aparece com IDs neutros e as duas sugestões começam desmarcadas; ao descartar uma, é possível informar um motivo (opcional). Na aba "Exportação", a origem de cada risco escolhido é revelada, com as contagens por origem e desfecho, e ficam disponíveis as 3 perguntas finais (escala 1-5) e um comentário opcional; "Salvar avaliação" grava o log do processo em `data/avaliacao/<código>_<timestamp>.json`.

Durante a avaliação (`MODO_AVALIACAO`, ligado por padrão nesta branch), nada é gravado em `data/riscos_base.csv`: riscos de IA escolhidos vão para `data/candidatos_ia.csv`, e o botão de salvar manualmente na biblioteca (revisão humana) fica desativado. Para desligar (voltando a gravar direto na biblioteca), defina `MODO_AVALIACAO=false` por variável de ambiente ou em `secrets.toml`.

### Consolidar os resultados

```powershell
python scripts/consolidar_avaliacao.py
```

Lê todos os JSON em `data/avaliacao/`, gera `resumo_processos.csv` (uma linha por processo) e `resumo_pares.csv` (uma linha por par biblioteca+IA, com o grupo de corte e a pontuação da sugestão de biblioteca) na raiz do projeto, e imprime no terminal os totais e as taxas por desfecho (`so_biblioteca`, `so_ia`, `ambos`, `nenhum`, `sem_par`) **separados para pares acima do corte, abaixo do corte e `sem_par`**. O teste de McNemar exato entre `so_biblioteca` e `so_ia` usa só os pares acima do corte — abaixo do corte ou sem risco de biblioteca não são uma comparação justa entre as duas origens. Use `--log-dir` e `--out-dir` para outros caminhos.

## Decisao de Projeto

A extracao automatica de conhecimento de processos antigos fica como evolucao futura. O MVP concentra-se na operacionalizacao assistida de conhecimento ja estruturado, com revisao humana obrigatoria e exportacao da matriz final.

## Saida Word

A exportacao Word segue o formato institucional observado no mapa de riscos de referencia: capa, historico, orientacoes, grupos de riscos, tabela individual por risco, escala 1-5 de probabilidade e impacto, nivel calculado, estrategia, consequencias, acoes preventivas, acoes de contingencia, responsavel por acao e anexos de escala.
