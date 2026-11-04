"""Avaliacao comparativa dos motores de sugestao de riscos.

Este pacote e isolado do app (``src/``): tem seu proprio ``requirements.txt``
e nao altera ``src/suggestions.py`` nem o comportamento do app. Ele apenas
importa ``src.models``, ``src.risk_library`` e ``src.suggestions`` como
consumidor somente leitura, para avaliar o motor de regras existente ao lado
dos motores novos (BM25, embeddings, LLM restrito ao repositorio).
"""
