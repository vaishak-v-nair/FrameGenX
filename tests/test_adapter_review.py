import json

import pytest

from scripts.review_adapter import parse_answers, summarize


def test_missing_and_duplicate_requirement_answers_are_rejected():
    reply = {"answers": [{"index": 0, "answer": "yes", "observation": "Two objects"}]}
    with pytest.raises(ValueError, match="omitted"):
        parse_answers(json.dumps(reply), 2)
    reply["answers"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        parse_answers(json.dumps(reply), 2)


def test_speculation_and_nonboolean_indexes_are_rejected():
    for index, answer in ((0, "probably yes"), (False, "yes")):
        with pytest.raises(ValueError):
            parse_answers(json.dumps({"answers": [{"index": index, "answer": answer, "observation": "uncertain"}]}), 1)


def test_unknown_requirements_remain_in_denominator_and_prevent_all_yes_pass():
    result = summarize([{"case_id": "case", "seed": 42, "arm": "candidate", "answers": [
        {"answer": "yes"}, {"answer": "unknown"}]}])
    assert result["arms"]["candidate"]["uncalibrated_supported_requirement_rate"] == .5
    assert result["arms"]["candidate"]["uncalibrated_all_yes_images"] == 0
    assert result["arms"]["candidate"]["unknown"] == 1
    assert result["full_contract_CSR"] is None
    assert result["promotion_status"] == "unqualified"


def test_paired_diagnostics_are_not_promoted_to_validated_csr():
    records = [{"case_id": "case", "seed": 42, "arm": arm, "answers": [{"answer": answer}]}
               for arm, answer in (("baseline", "no"), ("candidate", "yes"))]
    result = summarize(records)
    assert result["uncalibrated_candidate_only_passes"] == 1
    assert result["full_contract_CSR"] is None
    assert result["quality_status"] == "unverified"


def test_empty_requirements_cannot_produce_a_successful_review():
    with pytest.raises(ValueError, match="No requirements"):
        parse_answers('{"answers": []}', 0)
    result = summarize([{"case_id": "empty", "seed": 42, "arm": "candidate", "answers": []}])
    assert result["arms"]["candidate"]["uncalibrated_all_yes_images"] is None


def test_unavailable_judge_does_not_become_a_zero_quality_score():
    records = [{"case_id": "case", "seed": 42, "arm": arm, "answers": [{"answer": "unknown"}]}
               for arm in ("baseline", "candidate")]
    result = summarize(records)
    assert result["paired_cases"] == 1
    assert result["paired_fully_answered_cases"] == 0
    assert result["arms"]["candidate"]["uncalibrated_supported_requirement_rate"] is None
    assert result["uncalibrated_candidate_only_passes"] is None


@pytest.mark.parametrize("method", ["generate", "generate_json", "generate_vision"])
def test_router_error_logs_never_echo_provider_credentials(monkeypatch, capsys, method):
    from types import SimpleNamespace
    from src.core import llm_router as module
    class CredentialBearingError(Exception):
        status_code = 401
    def fail(*args, **kwargs):
        raise CredentialBearingError("Incorrect API key: synthetic-secret-do-not-print")
    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))))
    router = module.LLMRouter()
    provider = module.ProviderConfig(name="Test", base_url="https://example.invalid",
                                     model="test", vision_model="test", priority=1)
    router.providers = router.available_providers = [provider]
    with pytest.raises(RuntimeError):
        getattr(router, method)([{"role": "user", "content": "synthetic fixture"}])
    assert "synthetic-secret" not in capsys.readouterr().out
    assert router.last_failures == [{"provider": "Test", "error_type": "CredentialBearingError", "http_status": 401}]
