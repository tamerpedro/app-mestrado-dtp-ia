import json

import pytest

from evaluation.config import LLMConfig, LLMConfigError
from evaluation.engines import llm_engine
from evaluation.engines.llm_engine import LLMNotEnabledError


class FakeLLMClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls = 0

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
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
