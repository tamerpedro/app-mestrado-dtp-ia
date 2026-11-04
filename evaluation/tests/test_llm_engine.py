import json

import pytest

from evaluation.config import LLMConfig, LLMConfigError
from evaluation.engines import llm_engine
from evaluation.engines.llm_engine import LLMNotEnabledError
from evaluation.text_utils import risk_document_text


class FakeLLMClient:
    def __init__(self, response_text: str, last_model_id: str | None = None) -> None:
        self.response_text = response_text
        self.calls = 0
        self.last_user_prompt: str | None = None
        self.last_model_id = last_model_id

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.last_user_prompt = user_prompt
        return self.response_text


class FakeLLMClientWithoutModelId:
    """Cliente minimo, sem o atributo opcional ``last_model_id`` -- deve
    funcionar normalmente, so sem preencher ``reported_model_id``."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.response_text


def test_llm_disabled_by_default():
    config = LLMConfig()
    assert config.enabled is False


def test_rank_raises_when_llm_not_enabled(synthetic_risks, security_context):
    config = LLMConfig(enabled=False)
    with pytest.raises(LLMNotEnabledError):
        llm_engine.rank(security_context, synthetic_risks, config, client=FakeLLMClient("[]"))


def test_non_local_endpoint_requires_explicit_allow_external():
    with pytest.raises(LLMConfigError):
        LLMConfig(enabled=True, endpoint="https://api.example.com/v1/chat/completions")


def test_non_local_endpoint_allowed_when_explicitly_opted_in():
    config = LLMConfig(
        enabled=True,
        endpoint="https://api.example.com/v1/chat/completions",
        allow_external=True,
    )
    assert config.endpoint.startswith("https://api.example.com")


def test_local_endpoint_does_not_require_allow_external():
    config = LLMConfig(enabled=True, endpoint="http://localhost:11434/v1/chat/completions")
    assert config.enabled is True


def test_invalid_ids_are_filtered_out_and_counted(synthetic_risks, security_context):
    config = LLMConfig(enabled=True)
    response = json.dumps(["S001", "S999", "S002", "FABRICATED"])
    client = FakeLLMClient(response)

    result = llm_engine.rank(security_context, synthetic_risks, config, client=client)

    assert result.ranked_ids == ("S001", "S002")
    assert result.raw_ids_before_validation == ("S001", "S999", "S002", "FABRICATED")


def test_duplicate_valid_ids_are_deduplicated_keeping_first_occurrence(synthetic_risks, security_context):
    config = LLMConfig(enabled=True)
    client = FakeLLMClient(json.dumps(["S001", "S001", "S002"]))

    result = llm_engine.rank(security_context, synthetic_risks, config, client=client)

    assert result.ranked_ids == ("S001", "S002")


def test_malformed_response_yields_empty_ranking(synthetic_risks, security_context):
    config = LLMConfig(enabled=True)
    client = FakeLLMClient("isto nao e JSON valido")

    result = llm_engine.rank(security_context, synthetic_risks, config, client=client)

    assert result.ranked_ids == ()


def test_response_with_surrounding_text_is_still_parsed(synthetic_risks, security_context):
    config = LLMConfig(enabled=True)
    client = FakeLLMClient('Aqui esta minha resposta: ["S001", "S003"] -- espero que ajude.')

    result = llm_engine.rank(security_context, synthetic_risks, config, client=client)

    assert result.ranked_ids == ("S001", "S003")


def test_prompt_uses_the_same_risk_text_as_bm25_and_embeddings(synthetic_risks, security_context):
    """O prompt deve descrever cada candidato com o mesmo texto usado pelo
    BM25 e pelo motor de embeddings (``text_utils.risk_document_text``), nao
    um resumo proprio -- para que os tres motores vejam a mesma informacao."""
    config = LLMConfig(enabled=True)
    client = FakeLLMClient("[]")

    llm_engine.rank(security_context, synthetic_risks, config, client=client)

    assert client.last_user_prompt is not None
    for risk in synthetic_risks:
        assert risk_document_text(risk) in client.last_user_prompt


def test_reported_model_id_is_captured_from_the_client(synthetic_risks, security_context):
    config = LLMConfig(enabled=True, model="llama3.1:8b-instruct")
    client = FakeLLMClient(json.dumps(["S001"]), last_model_id="llama3.1:8b-instruct-q4_0")

    result = llm_engine.rank(security_context, synthetic_risks, config, client=client)

    assert result.reported_model_id == "llama3.1:8b-instruct-q4_0"


def test_reported_model_id_is_none_when_client_does_not_expose_it(synthetic_risks, security_context):
    config = LLMConfig(enabled=True)
    client = FakeLLMClientWithoutModelId(json.dumps(["S001"]))

    result = llm_engine.rank(security_context, synthetic_risks, config, client=client)

    assert result.reported_model_id is None
    assert result.ranked_ids == ("S001",)


def test_non_llm_rank_results_never_set_reported_model_id(synthetic_risks, security_context):
    from evaluation.engines import bm25_engine, rules_engine

    assert rules_engine.rank(security_context, synthetic_risks).reported_model_id is None
    assert bm25_engine.rank(security_context, synthetic_risks).reported_model_id is None
