from __future__ import annotations

import sys
from pathlib import Path

import pytest

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.evaluation_service as evaluation_service  # type: ignore  # noqa: E402


def test_auto_parallel_limits_openai_single_dict():

    def resolve_provider(model_name: str) -> str:
        assert model_name
        return "openai" if model_name == "gpt" else "ollama"

    tasks, configs, mode = evaluation_service._auto_parallel_limits(
        {"llm_model": "gpt", "evaluation_llm_model": "mistral"},
        default_parallel_tasks=99,
        default_parallel_configs=77,
        resolve_provider=resolve_provider,
    )
    assert (tasks, configs, mode) == (99, 77, "openai")


def test_auto_parallel_limits_openai_mixed_list_is_safe_side():
    def resolve_provider(model_name: str) -> str:
        return "openai" if model_name.startswith("openai") else "ollama"

    tasks, configs, mode = evaluation_service._auto_parallel_limits(
        [
            {"llm_model": "ollama:mistral"},
            {"llm_model": "openai:gpt-4o"},
        ],
        default_parallel_tasks=3,
        default_parallel_configs=2,
        resolve_provider=resolve_provider,
    )
    assert mode == "openai"
    assert tasks == 3
    assert configs == 2


def test_auto_parallel_limits_ollama_only():
    def resolve_provider(model_name: str) -> str:
        return "ollama"

    tasks, configs, mode = evaluation_service._auto_parallel_limits(
        [
            {"llm_model": "mistral"},
            {"evaluation_llm_model": "gpt-oss"},
        ],
        default_parallel_tasks=8,
        default_parallel_configs=4,
        resolve_provider=resolve_provider,
    )
    assert (tasks, configs, mode) == (8, 4, "ollama")


def test_auto_parallel_limits_unknown_when_no_models():
    def resolve_provider(model_name: str) -> str:  # pragma: no cover
        raise AssertionError("resolve_provider should not be called")

    tasks, configs, mode = evaluation_service._auto_parallel_limits(
        ["not-a-dict", {"foo": "bar"}],
        default_parallel_tasks=5,
        default_parallel_configs=6,
        resolve_provider=resolve_provider,
    )
    assert (tasks, configs, mode) == (5, 6, "unknown")


def test_pick_provider_from_config_prefers_llm_model_then_eval_llm_model():
    def resolve_provider(model_name: str) -> str:
        return "openai" if model_name == "openai:gpt" else "ollama"

    provider = evaluation_service._pick_provider_from_config(
        {"llm_model": "openai:gpt", "evaluation_llm_model": "ollama:mistral"},
        resolve_provider=resolve_provider,
    )
    assert provider == "openai"


def test_pick_provider_from_config_uses_eval_llm_when_llm_missing():
    def resolve_provider(model_name: str) -> str:
        return "ollama" if model_name else "unknown"

    provider = evaluation_service._pick_provider_from_config(
        {"evaluation_llm_model": "mistral"},
        resolve_provider=resolve_provider,
    )
    assert provider == "ollama"


def test_pick_provider_from_config_unknown_for_non_dict():
    def resolve_provider(model_name: str) -> str:  # pragma: no cover
        raise AssertionError("resolve_provider should not be called")

    provider = evaluation_service._pick_provider_from_config(
        ["not-a-dict"],
        resolve_provider=resolve_provider,
    )
    assert provider == "unknown"


def test_resolve_parallel_limits_caps_ollama_when_env_not_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_MAX_PARALLEL_TASKS", raising=False)
    monkeypatch.delenv("EVAL_MAX_PARALLEL_CONFIGS", raising=False)
    monkeypatch.delenv("EVAL_MAX_PARALLEL_TASKS_OLLAMA", raising=False)
    monkeypatch.delenv("EVAL_MAX_PARALLEL_CONFIGS_OLLAMA", raising=False)
    monkeypatch.setenv("EVAL_MAX_PARALLEL_TASKS_OPENAI", "3")
    monkeypatch.setenv("EVAL_MAX_PARALLEL_CONFIGS_OPENAI", "2")

    (
        tasks_openai,
        tasks_ollama,
        configs_openai,
        configs_ollama,
        env_tasks_mode,
        env_configs_mode,
    ) = evaluation_service._resolve_parallel_limits_for_bulk_evaluate(cpu_count=8)

    assert env_tasks_mode == "auto"
    assert env_configs_mode == "auto"
    assert tasks_openai == 3
    assert configs_openai == 2
    assert tasks_ollama == 1
    assert configs_ollama == 1


def test_resolve_parallel_limits_respects_explicit_ollama_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_MAX_PARALLEL_TASKS", raising=False)
    monkeypatch.delenv("EVAL_MAX_PARALLEL_CONFIGS", raising=False)
    monkeypatch.setenv("EVAL_MAX_PARALLEL_TASKS_OPENAI", "3")
    monkeypatch.setenv("EVAL_MAX_PARALLEL_CONFIGS_OPENAI", "2")
    monkeypatch.setenv("EVAL_MAX_PARALLEL_TASKS_OLLAMA", "4")
    monkeypatch.setenv("EVAL_MAX_PARALLEL_CONFIGS_OLLAMA", "5")

    (
        tasks_openai,
        tasks_ollama,
        configs_openai,
        configs_ollama,
        env_tasks_mode,
        env_configs_mode,
    ) = evaluation_service._resolve_parallel_limits_for_bulk_evaluate(cpu_count=8)

    assert env_tasks_mode == "auto"
    assert env_configs_mode == "auto"
    assert tasks_openai == 3
    assert configs_openai == 2
    assert tasks_ollama == 4
    assert configs_ollama == 5
