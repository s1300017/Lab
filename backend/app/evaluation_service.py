from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import re
import textwrap
import time
import statistics
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from logging.handlers import RotatingFileHandler
from typing import Any, Iterable, Mapping, Sequence

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector

try:  # FAISS は任意依存
    from langchain_community.vectorstores import FAISS

    _FAISS_AVAILABLE = True
except ImportError:  # noqa: WPS440
    FAISS = None  # type: ignore
    _FAISS_AVAILABLE = False

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from .chunk_utils import (
    fixed_chunk_text,
    sentence_chunk_text,
    paragraph_chunk_text,
    semantic_chunk_text,
)
from .llm_ragas_utils import RAGASLLMAsyncAdapter, _extract_answer_text, build_rag_answer_prompt
from .persistence_utils import persist_experiment_results
from .settings import JsonFormatter


logger = logging.getLogger(__name__)


_RAGAS_EVAL_LOGGER: logging.Logger | None = None


def _get_ragas_eval_logger() -> logging.Logger:
    global _RAGAS_EVAL_LOGGER

    if _RAGAS_EVAL_LOGGER is not None:
        return _RAGAS_EVAL_LOGGER

    log_path = os.getenv("RAGAS_EVAL_LOG_PATH", "").strip()
    if not log_path:
        is_pytest = ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules)
        if not is_pytest:
            try:
                from pathlib import Path

                project_root = Path(__file__).resolve().parents[2]
                log_path = str(project_root / "logs" / "ragas_eval.log")
            except Exception:  # noqa: BLE001
                log_path = ""
    max_bytes_env = os.getenv("RAGAS_EVAL_LOG_MAX_BYTES", "10485760").strip()
    backup_count_env = os.getenv("RAGAS_EVAL_LOG_BACKUP_COUNT", "5").strip()
    level_name = os.getenv("RAGAS_EVAL_LOG_LEVEL", "INFO").upper().strip()

    ragas_logger = logging.getLogger("ragas_eval")
    ragas_logger.setLevel(getattr(logging, level_name, logging.INFO))
    ragas_logger.propagate = False

    if log_path:
        try:
            max_bytes = int(max_bytes_env) if max_bytes_env else 10485760
        except Exception:  # noqa: BLE001
            max_bytes = 10485760
        try:
            backup_count = int(backup_count_env) if backup_count_env else 5
        except Exception:  # noqa: BLE001
            backup_count = 5

        try:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            handler = RotatingFileHandler(
                log_path,
                maxBytes=max(0, max_bytes),
                backupCount=max(0, backup_count),
                encoding="utf-8",
            )
            handler.setFormatter(JsonFormatter())

            if not any(
                getattr(h, "baseFilename", None) == getattr(handler, "baseFilename", None)
                for h in ragas_logger.handlers
            ):
                ragas_logger.addHandler(handler)
        except Exception:  # noqa: BLE001
            pass

    _RAGAS_EVAL_LOGGER = ragas_logger
    return ragas_logger


def _run_ragas_evaluate_to_pandas(
    *,
    dataset: Any,
    metrics: list[Any],
    llm: Any,
    embeddings: Any,
    max_workers: int,
    timeout: int | None,
) -> Any:
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            timeout=timeout,
            max_workers=max_workers,
        ),
    )
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    if hasattr(result, "to_dict") and hasattr(result, "columns"):
        return result
    return None


def _auto_parallel_limits(
    data: Any,
    *,
    default_parallel_tasks: int,
    default_parallel_configs: int,
    resolve_provider,
) -> tuple[int, int, str]:
    def _as_list(d: Any) -> list[dict[str, Any]]:
        if isinstance(d, list):
            return [x for x in d if isinstance(x, dict)]
        if isinstance(d, dict):
            return [d]
        return []

    items = _as_list(data)
    providers: set[str] = set()
    for one in items:
        llm_name = one.get("llm_model")
        eval_llm_name = one.get("evaluation_llm_model")
        for name in (llm_name, eval_llm_name):
            if not name:
                continue
            try:
                providers.add(str(resolve_provider(str(name))).strip())
            except Exception:  # noqa: BLE001
                continue

    has_openai = "openai" in providers
    has_ollama = "ollama" in providers

    if has_openai:
        return default_parallel_tasks, default_parallel_configs, "openai"
    if has_ollama:
        return default_parallel_tasks, default_parallel_configs, "ollama"
    return default_parallel_tasks, default_parallel_configs, "unknown"


def _resolve_parallel_limits_for_bulk_evaluate(
    *,
    cpu_count: int | None = None,
) -> tuple[int, int, int, int, str, str]:
    """bulk_evaluate 用の並列数設定を環境変数から解決する。

    Ollama は 429 等のエラーを踏みやすいため、環境変数で明示されていない(auto)場合のみ
    安全側にキャップする。
    """

    cpu = cpu_count or (os.cpu_count() or 4)
    cpu_based_default_parallel = max(2, min(8, max(1, cpu // 2)))

    env_tasks = os.getenv("EVAL_MAX_PARALLEL_TASKS")
    env_configs = os.getenv("EVAL_MAX_PARALLEL_CONFIGS")

    default_tasks_openai = int(os.getenv("EVAL_MAX_PARALLEL_TASKS_OPENAI", "2"))
    default_configs_openai = int(os.getenv("EVAL_MAX_PARALLEL_CONFIGS_OPENAI", "1"))

    raw_tasks_ollama = os.getenv("EVAL_MAX_PARALLEL_TASKS_OLLAMA")
    raw_configs_ollama = os.getenv("EVAL_MAX_PARALLEL_CONFIGS_OLLAMA")

    if raw_tasks_ollama is None or str(raw_tasks_ollama).strip() == "":
        default_tasks_ollama = cpu_based_default_parallel
        tasks_ollama_explicit = False
    else:
        default_tasks_ollama = int(str(raw_tasks_ollama).strip())
        tasks_ollama_explicit = True

    if raw_configs_ollama is None or str(raw_configs_ollama).strip() == "":
        default_configs_ollama = 2
        configs_ollama_explicit = False
    else:
        default_configs_ollama = int(str(raw_configs_ollama).strip())
        configs_ollama_explicit = True

    if env_tasks is None or str(env_tasks).strip() == "":
        tasks_openai = max(1, int(default_tasks_openai))
        tasks_ollama = max(1, int(default_tasks_ollama))
        env_tasks_mode = "auto"
    else:
        forced = max(1, int(env_tasks))
        tasks_openai = forced
        tasks_ollama = forced
        env_tasks_mode = "set"

    if env_configs is None or str(env_configs).strip() == "":
        configs_openai = max(1, int(default_configs_openai))
        configs_ollama = max(1, int(default_configs_ollama))
        env_configs_mode = "auto"
    else:
        try:
            forced_cfg = max(1, int(env_configs))
        except ValueError:
            forced_cfg = 1
        configs_openai = forced_cfg
        configs_ollama = forced_cfg
        env_configs_mode = "set"

    # Ollama は未指定(auto)の場合のみ安全側にキャップする（明示指定は尊重する）
    if env_tasks_mode == "auto" and not tasks_ollama_explicit:
        before = tasks_ollama
        tasks_ollama = min(tasks_ollama, 1)
        if tasks_ollama != before:
            logger.info(
                "[設定] OllamaのTASKS並列数を安全側にキャップ: %d -> %d",
                before,
                tasks_ollama,
            )

    if env_configs_mode == "auto" and not configs_ollama_explicit:
        before = configs_ollama
        configs_ollama = min(configs_ollama, 1)
        if configs_ollama != before:
            logger.info(
                "[設定] OllamaのCONFIGS並列数を安全側にキャップ: %d -> %d",
                before,
                configs_ollama,
            )

    return (
        tasks_openai,
        tasks_ollama,
        configs_openai,
        configs_ollama,
        env_tasks_mode,
        env_configs_mode,
    )


def _pick_provider_from_config(one: Any, *, resolve_provider) -> str:
    if not isinstance(one, dict):
        return "unknown"
    llm_name = one.get("llm_model")
    eval_llm_name = one.get("evaluation_llm_model")
    for name in (llm_name, eval_llm_name):
        if not name:
            continue
        try:
            provider = str(resolve_provider(str(name))).strip()
            if provider:
                return provider
        except Exception:  # noqa: BLE001
            continue
    return "unknown"


def _unique_preserve_order(seq: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _summarize_bulk_request(request_items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """複数ジョブの設定から実験全体の概要を構築する。"""
    valid_items = [item for item in request_items if isinstance(item, dict)]
    if not valid_items:
        return {}

    first = valid_items[0]
    summary: dict[str, Any] = {}

    for key in (
        "experiment_name",
        "file_id",
        "pdf_file_id",
        "question_mode",
        "use_n",
        "top_k",
        "use_mmr",
        "fetch_k",
        "lambda_mult",
    ):
        if first.get(key) is not None:
            summary[key] = first.get(key)

    # Retrieval条件は「実験条件」として保存（複数ジョブで値が異なる可能性もあるため unique を保存）
    retrieval_keys = ("top_k", "use_mmr", "fetch_k", "lambda_mult")
    for key in retrieval_keys:
        values = _unique_preserve_order(
            item.get(key) for item in valid_items if item.get(key) is not None
        )
        if not values:
            continue
        summary[key] = values[0] if len(values) == 1 else values

    llm_models = _unique_preserve_order(
        item.get("llm_model")
        for item in valid_items
        if item.get("llm_model")
    )
    if llm_models:
        summary["llm_models"] = llm_models
        summary["llm_model"] = llm_models[0]

    embedding_models = _unique_preserve_order(
        item.get("embedding_model")
        for item in valid_items
        if item.get("embedding_model")
    )
    if embedding_models:
        summary["embedding_models"] = embedding_models

    chunk_methods = []
    for item in valid_items:
        methods = item.get("chunk_methods")
        if isinstance(methods, list):
            chunk_methods.extend(methods)
        elif item.get("chunk_method"):
            chunk_methods.append(item.get("chunk_method"))
    chunk_methods = _unique_preserve_order(
        method for method in chunk_methods if method
    )
    if chunk_methods:
        summary["chunk_methods"] = chunk_methods

    chunk_sizes = []
    for item in valid_items:
        sizes = item.get("chunk_sizes")
        if isinstance(sizes, list):
            chunk_sizes.extend(sizes)
        elif item.get("chunk_size") is not None:
            chunk_sizes.append(item.get("chunk_size"))
    chunk_sizes = _unique_preserve_order(
        int(size) for size in chunk_sizes if size is not None
    )
    if chunk_sizes:
        summary["chunk_sizes"] = chunk_sizes

    chunk_overlaps = []
    for item in valid_items:
        overlaps = item.get("chunk_overlaps")
        if isinstance(overlaps, list):
            chunk_overlaps.extend(overlaps)
        elif item.get("chunk_overlap") is not None:
            chunk_overlaps.append(item.get("chunk_overlap"))
    chunk_overlaps = _unique_preserve_order(
        int(ov) for ov in chunk_overlaps if ov is not None
    )
    if chunk_overlaps:
        summary["chunk_overlaps"] = chunk_overlaps

    include_answer_similarity_values = [
        item.get("include_answer_similarity")
        for item in valid_items
        if item.get("include_answer_similarity") is not None
    ]
    if include_answer_similarity_values:
        summary["include_answer_similarity"] = bool(include_answer_similarity_values[-1])

    similarity_thresholds = [
        item.get("semantic_params", {}).get("similarity_threshold")
        for item in valid_items
        if isinstance(item.get("semantic_params"), dict)
    ]
    if similarity_thresholds:
        summary["similarity_threshold"] = similarity_thresholds[-1]

    evaluation_llm_models = _unique_preserve_order(
        item.get("evaluation_llm_model")
        for item in valid_items
        if item.get("evaluation_llm_model")
    )
    if evaluation_llm_models:
        summary["evaluation_llm_models"] = evaluation_llm_models
        summary["evaluation_llm_model"] = evaluation_llm_models[0]

    force_flags = [
        bool(item.get("force_llm_generation"))
        for item in valid_items
        if "force_llm_generation" in item
    ]
    summary["force_llm_generation"] = any(force_flags) if force_flags else False

    summary["total_jobs"] = len(valid_items)
    return summary


async def bulk_evaluate(data: Any, job_id: str | None = None) -> Any:
    """RAGAS 一括評価ロジック本体。

    `main.bulk_evaluate` から切り出したもので、リクエスト JSON 本体 (`dict` または `list`)
    を受け取り、評価結果の `dict` / `list` を返す。
    """

    from . import main as main_module  # 循環依存回避のため遅延インポート

    DEFAULT_LLM_NAME = main_module.DEFAULT_LLM_NAME
    get_embeddings = main_module.get_embeddings
    get_llm_eval = main_module.get_llm_eval
    init_generation_llm = main_module.init_generation_llm
    get_collection_name = main_module.get_collection_name
    SUPPORTED_EMBEDDING_MODELS = main_module.SUPPORTED_EMBEDDING_MODELS
    _parse_timeout_env = main_module._parse_timeout_env
    _bool_env = main_module._bool_env
    _hash_text = main_module._hash_text
    _hash_chunks = main_module._hash_chunks
    _CHUNK_CACHE = main_module._CHUNK_CACHE
    _CHUNK_CACHE_LOCK = main_module._CHUNK_CACHE_LOCK
    _VECTORSTORE_CACHE = main_module._VECTORSTORE_CACHE
    _VECTORSTORE_CACHE_LOCK = main_module._VECTORSTORE_CACHE_LOCK
    jst_now_str = main_module.jst_now_str

    try:
        ragas_logger = _get_ragas_eval_logger()
        ragas_logger.info(
            "[RAGAS] bulk_evaluate start",
            extra={"component": "ragas", "endpoint": "bulk_evaluate", "job_id": job_id},
        )
        # --- 数値のNaN/infガード用ユーティリティ ---
        def safe_val(x: Any) -> float:
            try:
                if isinstance(x, (int, float)) and (math.isnan(x) or math.isinf(x)):
                    return 0.0
                return float(x)
            except Exception:  # noqa: BLE001
                return 0.0

        # --- dataがリスト型なら各要素ごとに個別評価 ---
        def find_first_dict(obj: Any) -> dict:
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list):
                for item in obj:
                    found = find_first_dict(item)
                    if isinstance(found, dict):
                        return found
            return {}

        def _update_job_progress_safe(message: str) -> None:
            """evaluation_job_service.update_job_progress を安全に呼び出すヘルパー。

            job_id が指定されていない場合や、進捗更新時に例外が出た場合は黙って無視する。
            """

            if not job_id:
                return
            try:
                from . import evaluation_job_service as eval_job_service  # 遅延インポートで循環依存を回避

                eval_job_service.update_job_progress(job_id, message)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[進捗] bulk_evaluate からのジョブ進捗更新に失敗しました",
                )

        # 並列処理の最大数を制限するセマフォを作成
        cpu_count = os.cpu_count() or 4

        (
            MAX_PARALLEL_TASKS_OPENAI,
            MAX_PARALLEL_TASKS_OLLAMA,
            MAX_PARALLEL_CONFIGS_OPENAI,
            MAX_PARALLEL_CONFIGS_OLLAMA,
            env_tasks_mode,
            env_configs_mode,
        ) = _resolve_parallel_limits_for_bulk_evaluate(cpu_count=cpu_count)

        def _resolve_provider(model_name: str) -> str:
            _, entry = main_module._resolve_llm_entry(model_name)  # type: ignore[attr-defined]
            return str(entry.get("provider") or "unknown")

        # provider別セマフォ
        semaphore_tasks_openai = asyncio.Semaphore(MAX_PARALLEL_TASKS_OPENAI)
        semaphore_tasks_ollama = asyncio.Semaphore(MAX_PARALLEL_TASKS_OLLAMA)
        outer_semaphore_openai = asyncio.Semaphore(MAX_PARALLEL_CONFIGS_OPENAI)
        outer_semaphore_ollama = asyncio.Semaphore(MAX_PARALLEL_CONFIGS_OLLAMA)

        # retrieval は provider に依存しないため、タスク並列数の大きい方に揃える
        semaphore_retrieval = asyncio.Semaphore(
            max(MAX_PARALLEL_TASKS_OPENAI, MAX_PARALLEL_TASKS_OLLAMA),
        )

        logger.info(
            "[設定] 並列数: TASKS(openai=%d, ollama=%d, mode=%s) / CONFIGS(openai=%d, ollama=%d, mode=%s)",
            MAX_PARALLEL_TASKS_OPENAI,
            MAX_PARALLEL_TASKS_OLLAMA,
            env_tasks_mode,
            MAX_PARALLEL_CONFIGS_OPENAI,
            MAX_PARALLEL_CONFIGS_OLLAMA,
            env_configs_mode,
        )

        # 計測ログの有効化（環境変数でON/OFF）
        TIMING_LOG = os.getenv("EVAL_TIMING_LOG", "1").lower() in {"1", "true", "yes"}

        def _tnow() -> float:
            return time.monotonic()

        def _tlog(label: str, start: float | None = None) -> None:
            """EVAL_TIMING_LOG が有効なときに timing 情報を INFO ログとして出力する。"""

            if not TIMING_LOG:
                return
            try:
                if start is None:
                    logger.info("[timing] %s at %s", label, jst_now_str())
                else:
                    dur = _tnow() - start
                    logger.info("[timing] %s took %.3fs", label, dur)
            except Exception:  # noqa: BLE001
                pass

        def _check_cancel() -> None:
            """evaluation_jobs の cancel_requested フラグを確認し、キャンセル要求があれば例外を送出する。"""

            if not job_id:
                return
            try:
                from . import evaluation_job_service as eval_job_service  # 遅延インポート
            except Exception:  # noqa: BLE001
                return

            state = eval_job_service.get_bulk_job(job_id)
            if state and getattr(state, "cancel_requested", False):
                # 進捗メッセージも更新しておく
                _update_job_progress_safe(
                    "キャンセル要求を検知しました。現在の処理を安全に停止しています…",
                )
                raise eval_job_service.BulkJobCancelled("ユーザーによりキャンセルされました。")

        async def evaluate_one_bulk(one: dict) -> Any:
            try:
                logger.info("[進捗] 評価データを処理中...")
                _check_cancel()
                try:
                    ragas_logger.info(
                        "[RAGAS] evaluate_one_bulk start",
                        extra={
                            "component": "ragas",
                            "endpoint": "bulk_evaluate",
                            "job_id": job_id,
                            "pdf_file_id": one.get("pdf_file_id"),
                            "file_id": one.get("file_id"),
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
                # タイムアウト設定（環境変数で調整可能）
                # 既定値: LLM呼び出し=45秒, 評価全体は質問数に応じて自動調整
                LLM_TIMEOUT = _parse_timeout_env("EVAL_LLM_TIMEOUT_SECONDS", 45)
                EVAL_TIMEOUT: int | None = None  # 後段で質問数を基に算出

                embedding_model = one.get("embedding_model")
                chunk_methods = one.get("chunk_methods", [one.get("chunk_method", "recursive")])
                chunk_sizes = one.get("chunk_sizes", [one.get("chunk_size", 1000)])
                chunk_overlaps = one.get("chunk_overlaps", [one.get("chunk_overlap", 0)])
                request_llm_model = one.get("llm_model", DEFAULT_LLM_NAME)
                evaluation_llm_model = one.get("evaluation_llm_model")
                force_llm_generation = bool(one.get("force_llm_generation"))
                rag_prompt_style = one.get("rag_prompt_style") or "simple_en"
                llm_instance_generation, resolved_llm_model = init_generation_llm(
                    request_llm_model,
                    purpose="/bulk_evaluate generation",
                )

                resolved_eval_llm_model = evaluation_llm_model or DEFAULT_LLM_NAME

                generation_provider = _resolve_provider(resolved_llm_model)
                if generation_provider == "openai":
                    semaphore_tasks = semaphore_tasks_openai
                    max_workers_generation = MAX_PARALLEL_TASKS_OPENAI
                else:
                    semaphore_tasks = semaphore_tasks_ollama
                    max_workers_generation = MAX_PARALLEL_TASKS_OLLAMA

                evaluation_provider = _resolve_provider(resolved_eval_llm_model)
                if evaluation_provider == "openai":
                    max_workers_eval = MAX_PARALLEL_TASKS_OPENAI
                else:
                    max_workers_eval = MAX_PARALLEL_TASKS_OLLAMA
                if evaluation_provider == "ollama" and "cloud" in str(resolved_eval_llm_model).lower():
                    try:
                        max_workers_eval = min(
                            max_workers_eval,
                            max(1, int(os.getenv("EVAL_MAX_PARALLEL_TASKS_OLLAMA_CLOUD", "1"))),
                        )
                    except Exception:  # noqa: BLE001
                        max_workers_eval = 1
                logger.info(
                    "[設定] 生成LLM=%s / 評価LLM=%s / 再生成フラグ=%s",
                    resolved_llm_model,
                    resolved_eval_llm_model,
                    force_llm_generation,
                )

                # セマンティックチャンキングが選択されている場合の情報メッセージ
                if "semantic" in chunk_methods:
                    if len(chunk_methods) == 1:
                        logger.info(
                            "情報: セマンティックチャンキングが選択されました。チャンクサイズとオーバーラップは使用されません。",
                        )
                    else:
                        logger.info(
                            "情報: セマンティックチャンキングとその他のチャンキング方式が同時に選択されています。",
                        )
                        logger.info(
                            "      セマンティックチャンキング: デフォルトパラメータを使用",
                        )
                        logger.info(
                            "      その他の方式: 指定されたチャンクサイズとオーバーラップを使用",
                        )

                # 必須パラメータチェック
                sample_text = one.get("text")
                if not sample_text:
                    raise ValueError("textが指定されていません")

                # サポートされているモデルかチェック
                if embedding_model not in SUPPORTED_EMBEDDING_MODELS:
                    raise ValueError(
                        f"未サポートの埋め込みモデルが指定されました: {embedding_model}",
                    )

                # OpenAI埋め込みの旧指定に対する注意喚起
                if embedding_model == "openai":
                    logger.warning(
                        "警告: 'openai' は包括的な指定です。具体的な 'text-embedding-3-small' または 'text-embedding-3-large' を選択してください。",
                    )

                raw_questions = one.get("questions")
                # ground_truthキーまたはanswersキーのどちらかを使用（互換性のため）
                raw_ground_truth = one.get("ground_truth", one.get("answers"))
                if not isinstance(raw_questions, list) or not raw_questions:
                    raise ValueError(
                        "questionsが指定されていません。PDFアップロード時の自動生成結果をそのまま送信してください。",
                    )
                questions = [str(q) for q in raw_questions]
                ground_truth: list[str] | None = None
                if isinstance(raw_ground_truth, list) and raw_ground_truth:
                    ground_truth = [str(a) for a in raw_ground_truth]

                # 質問数が多すぎる場合は環境変数 EVAL_MAX_QUESTIONS で上限をかける
                try:
                    max_questions_env = os.getenv("EVAL_MAX_QUESTIONS")
                    max_questions: int | None = None
                    if max_questions_env is not None and str(max_questions_env).strip() != "":
                        max_questions = int(str(max_questions_env).strip())
                except Exception:  # noqa: BLE001
                    max_questions = None

                if (
                    isinstance(questions, list)
                    and max_questions is not None
                    and max_questions > 0
                    and len(questions) > max_questions
                ):
                    logger.info(
                        "[進捗] 質問数 %d 件のうち先頭 %d 件のみを評価対象とします (EVAL_MAX_QUESTIONS)",
                        len(questions),
                        max_questions,
                    )
                    questions = questions[:max_questions]

                    if ground_truth is not None:
                        ground_truth = ground_truth[:max_questions]

                if ground_truth is not None and len(ground_truth) != len(questions):
                    use_n = min(len(questions), len(ground_truth))
                    logger.warning(
                        "[警告] questions(%d) と answers/ground_truth(%d) の件数が一致しません。先頭 %d 件に揃えます。",
                        len(questions),
                        len(ground_truth),
                        use_n,
                    )
                    questions = questions[:use_n]
                    ground_truth = ground_truth[:use_n]

                if not (sample_text and questions):
                    raise ValueError(
                        "PDFアップロードとQA自動生成を先に実施してください（text, questions必須）。",
                    )

                include_answer_similarity = _bool_env(one.get("include_answer_similarity"), True)
                has_reference = bool(ground_truth)

                # 質問数に応じて評価タイムアウトのデフォルト値を自動調整
                question_count = len(questions)
                # LLMタイムアウトが無制限の場合は基準を60秒とする
                base_llm_timeout = 60 if LLM_TIMEOUT is None else max(LLM_TIMEOUT, 30)
                # 質問1件あたり約30秒を目安としつつ、全体は最大480秒で抑制
                dynamic_eval_default = max(180, min(480, question_count * 30))
                # LLM呼び出し時間が長い場合を考慮して上限を引き上げ
                dynamic_eval_default = max(
                    dynamic_eval_default,
                    min(600, question_count * base_llm_timeout),
                )
                EVAL_TIMEOUT = _parse_timeout_env(
                    "RAGAS_EVAL_TIMEOUT_SECONDS",
                    dynamic_eval_default,
                )

                def _fmt(t: int | None) -> str:
                    return "no-timeout" if t is None else f"{t}s"

                logger.info(
                    "[設定] TIMEOUT: LLM_TIMEOUT=%s, EVAL_TIMEOUT=%s, MAX_PARALLEL_TASKS(generation=%d, eval=%d)",
                    _fmt(LLM_TIMEOUT),
                    _fmt(EVAL_TIMEOUT),
                    max_workers_generation,
                    max_workers_eval,
                )
                try:
                    ragas_logger.info(
                        "[RAGAS] job config",
                        extra={
                            "component": "ragas",
                            "endpoint": "bulk_evaluate",
                            "job_id": job_id,
                            "file_id": one.get("file_id"),
                            "pdf_file_id": one.get("pdf_file_id"),
                            "llm_model": resolved_llm_model,
                            "evaluation_llm_model": resolved_eval_llm_model,
                            "embedding_model": embedding_model,
                            "chunk_methods": chunk_methods,
                            "rag_prompt_style": rag_prompt_style,
                            "top_k": one.get("top_k"),
                            "use_mmr": one.get("use_mmr"),
                            "fetch_k": one.get("fetch_k"),
                            "lambda_mult": one.get("lambda_mult"),
                            "max_workers_generation": max_workers_generation,
                            "max_workers_eval": max_workers_eval,
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass

                results: list[dict[str, Any]] = []

                # embedding_modelのインスタンスを一度だけロードし再利用
                logger.info("[進捗] 埋め込みモデル '%s' をロード中...", embedding_model)
                _t0_embed = _tnow()
                embedder = get_embeddings(embedding_model)
                _tlog("embedder.load", _t0_embed)

                # chunk_method/chunk_size/chunk_overlapごとに完全に独立して処理
                for i, chunk_method in enumerate(chunk_methods):
                    job_start_time = _tnow()
                    try:
                        _check_cancel()
                        logger.info("[進捗] チャンク方法 '%s' の処理を開始...", chunk_method)

                        # セマンティックチャンキングの場合、チャンクサイズとオーバーラップは無視する
                        if chunk_method == "semantic":
                            if not embedding_model:
                                duration_seconds = _tnow() - job_start_time
                                results.append(
                                    {
                                        "error": "セマンティックチャンキングにはembedding_modelの指定が必須です",
                                        "chunk_method": chunk_method,
                                        "duration_seconds": duration_seconds,
                                    },
                                )
                                continue

                            logger.info(
                                "[進捗] セマンティックチャンキングを開始します（chunk_sizeとchunk_overlapは無視されます）...",
                            )
                            _t0_chunk = _tnow()

                            # セマンティックチャンキングのパラメータを取得
                            semantic_params = one.get("semantic_params", {}) or {}
                            similarity_threshold = float(
                                semantic_params.get("similarity_threshold", 0.7),
                            )

                            logger.info(
                                "[進捗] セマンティックチャンキングを実行: similarity_threshold=%s",
                                similarity_threshold,
                            )
                            text_hash = _hash_text(sample_text)
                            chunk_cache_key = (
                                text_hash,
                                embedding_model or "",
                                similarity_threshold,
                                chunk_method,
                            )
                            with _CHUNK_CACHE_LOCK:
                                cached_chunks = _CHUNK_CACHE.get(chunk_cache_key)
                            if cached_chunks is not None:
                                logger.info(
                                    "[CACHE] semantic chunks hit: key=%s len=%d",
                                    chunk_cache_key,
                                    len(cached_chunks),
                                )
                                chunks = cached_chunks
                            else:
                                chunks = await asyncio.to_thread(
                                    semantic_chunk_text,
                                    text=sample_text,
                                    chunk_size=None,
                                    chunk_overlap=None,
                                    embedding_model=embedder,
                                    similarity_threshold=similarity_threshold,
                                )
                                with _CHUNK_CACHE_LOCK:
                                    _CHUNK_CACHE[chunk_cache_key] = chunks
                                logger.info(
                                    "[CACHE] semantic chunks store: key=%s len=%d",
                                    chunk_cache_key,
                                    len(chunks),
                                )
                            _tlog("chunking.semantic", _t0_chunk)

                            # セマンティックチャンキングの場合はchunk_sizeとchunk_overlapをNoneに設定
                            chunk_size_val = None
                            chunk_overlap_val = None
                            chunk_strategy = "semantic"
                        else:
                            # 通常のチャンキング方法の場合
                            chunk_size = (
                                chunk_sizes[i] if i < len(chunk_sizes) else 1000
                            )
                            chunk_overlap = (
                                chunk_overlaps[i] if i < len(chunk_overlaps) else 200
                            )

                            logger.info(
                                "[進捗] チャンク分割を実行: 方式=%s, サイズ=%d, オーバーラップ=%d",
                                chunk_method,
                                chunk_size,
                                chunk_overlap,
                            )
                            _t0_chunk = _tnow()

                            # 非同期でチャンク分割を実行
                            if chunk_method == "recursive":
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    length_function=len,
                                )
                                chunks = await asyncio.to_thread(
                                    text_splitter.split_text,
                                    sample_text,
                                )
                            elif chunk_method == "fixed":
                                chunks = await asyncio.to_thread(
                                    fixed_chunk_text,
                                    sample_text,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                )
                            elif chunk_method == "sentence":
                                chunks = await asyncio.to_thread(
                                    sentence_chunk_text,
                                    sample_text,
                                )
                            elif chunk_method == "paragraph":
                                chunk_func = paragraph_chunk_text
                                if not callable(chunk_func):
                                    logger.warning(
                                        "[警告] paragraph_chunk_text が未定義のため簡易段落分割ロジックを使用します。",
                                    )

                                    def _paragraph_fallback(text: str) -> list[str]:
                                        if not text:
                                            return []
                                        paragraphs = [
                                            block.strip()
                                            for block in re.split(r"\n\s*\n", text)
                                            if block.strip()
                                        ]
                                        return paragraphs if paragraphs else [text]

                                    chunk_func = _paragraph_fallback
                                chunks = await asyncio.to_thread(chunk_func, sample_text)
                            else:
                                raise ValueError(
                                    f"未対応のchunk_method: {chunk_method}",
                                )
                            _tlog(f"chunking.{chunk_method}", _t0_chunk)

                            # チャンク戦略を設定
                            chunk_size_val = (
                                chunk_sizes[i] if i < len(chunk_sizes) else chunk_sizes[0]
                            )
                            chunk_overlap_val = (
                                chunk_overlaps[i]
                                if i < len(chunk_overlaps)
                                else chunk_overlaps[0]
                            )
                            chunk_strategies = (
                                one.get("chunk_strategies", [])
                                if isinstance(one, dict)
                                else []
                            )
                            if chunk_strategies and i < len(chunk_strategies):
                                chunk_strategy = chunk_strategies[i]
                            else:
                                chunk_strategy = (
                                    f"{chunk_method}-{chunk_size_val}-{chunk_overlap_val}"
                                )

                        logger.info(
                            "[進捗] %d個のチャンクを作成しました。平均長さ: %.1f文字",
                            len(chunks),
                            sum(len(c) for c in chunks) / max(len(chunks), 1),
                        )
                        logger.info(
                            "[進捗] ベクトルストアを構築中 (%s)...",
                            "FAISS" if _FAISS_AVAILABLE else "PGVector",
                        )
                        _t0_vs = _tnow()
                        chunk_hash = _hash_chunks(chunks)
                        vector_cache_key = (
                            embedding_model or "",
                            chunk_hash,
                            chunk_method,
                        )
                        with _VECTORSTORE_CACHE_LOCK:
                            cached_vs = _VECTORSTORE_CACHE.get(vector_cache_key)
                        if cached_vs is not None:
                            logger.info("[CACHE] vectorstore hit: key=%s", vector_cache_key)
                            vectorstore = cached_vs
                        else:
                            if _FAISS_AVAILABLE and FAISS is not None:
                                vectorstore = await asyncio.to_thread(
                                    FAISS.from_texts,
                                    texts=chunks,
                                    embedding=embedder,
                                )
                                _tlog("vectorstore.faiss.build", _t0_vs)
                            else:
                                # PGVectorへのフォールバック
                                logger.warning(
                                    "[警告] FAISSが未インストールのためPGVectorにフォールバックします",
                                )
                                vectorstore = PGVector.from_documents(
                                    documents=[],
                                    embedding=embedder,
                                    collection_name=get_collection_name(embedding_model),
                                )
                                await asyncio.to_thread(
                                    vectorstore.add_texts,
                                    texts=chunks,
                                )
                                _tlog("vectorstore.pgvector.build", _t0_vs)
                            with _VECTORSTORE_CACHE_LOCK:
                                _VECTORSTORE_CACHE[vector_cache_key] = vectorstore

                        # 検索パラメータの受け口（既定は従来と互換）
                        top_k = int(one.get("top_k", 5))
                        use_mmr = bool(one.get("use_mmr", False))
                        fetch_k = int(one.get("fetch_k", max(top_k * 2, 20)))
                        try:
                            lambda_mult = float(one.get("lambda_mult", 0.5))
                        except Exception:  # noqa: BLE001
                            lambda_mult = 0.5
                        if use_mmr:
                            logger.info(
                                "[設定] retriever=MMR k=%d, fetch_k=%d, lambda_mult=%.3f",
                                top_k,
                                fetch_k,
                                lambda_mult,
                            )
                            retriever = vectorstore.as_retriever(
                                search_type="mmr",
                                search_kwargs={
                                    "k": top_k,
                                    "fetch_k": fetch_k,
                                    "lambda_mult": lambda_mult,
                                },
                            )
                        else:
                            logger.info("[設定] retriever=similarity k=%d", top_k)
                            retriever = vectorstore.as_retriever(
                                search_kwargs={"k": top_k},
                            )

                        # RAG回答生成＆コンテキスト取得
                        contexts: list[list[str]] = []
                        pred_answers: list[str] = []

                        # PDFアップロード時の回答が揃っていれば使い回し（高速化）
                        if (
                            ground_truth
                            and len(ground_truth) == len(questions)
                            and not force_llm_generation
                        ):
                            logger.info(
                                "[進捗] PDFアップロード時の回答を使用（%d個の回答）",
                                len(ground_truth),
                            )
                            pred_answers = list(ground_truth)

                            async def get_context_only(q: str) -> list[str]:
                                async with semaphore_retrieval:
                                    _check_cancel()
                                    retrieved_docs = await asyncio.to_thread(
                                        retriever.get_relevant_documents,
                                        q,
                                    )
                                    return [doc.page_content for doc in retrieved_docs]

                            _t0_ctx = _tnow()
                            contexts = await asyncio.gather(
                                *[get_context_only(q) for q in questions],
                            )
                            _tlog("retrieval.contexts_only", _t0_ctx)
                            logger.info("[進捗] コンテキスト取得完了。評価処理を開始...")
                        else:
                            logger.info(
                                "[進捗] 新しいRAG回答を生成（%d個の質問）...",
                                len(questions),
                            )

                            async def get_context_and_answer(q: str) -> tuple[list[str], str]:
                                # 呼ばれるLLM(provider)に応じてセマフォを切り替える
                                async with semaphore_tasks:
                                    _check_cancel()
                                    retrieved_docs = await asyncio.to_thread(
                                        retriever.get_relevant_documents,
                                        q,
                                    )
                                    context_texts = [
                                        doc.page_content for doc in retrieved_docs
                                    ]
                                    llm_instance = llm_instance_generation
                                    if rag_prompt_style == "simple_en":
                                        prompt = ChatPromptTemplate.from_template(
                                            """Answer the question based only on the following context:\n{context}\n\nQuestion: {question}""",
                                        )

                                        def _to_text(x: Any) -> Any:
                                            try:
                                                return x.to_string()
                                            except Exception:  # noqa: BLE001
                                                return x

                                        llm_runnable = RunnableLambda(
                                            lambda x: llm_instance.invoke(_to_text(x)),
                                        )
                                        chain = (
                                            {
                                                "context": lambda _: context_texts,
                                                "question": lambda _: q,
                                            }
                                            | prompt
                                            | llm_runnable
                                            | StrOutputParser()
                                        )
                                        call = lambda: chain.ainvoke({})
                                    else:
                                        prompt_text = build_rag_answer_prompt(
                                            context="\n".join(context_texts),
                                            question=q,
                                        )

                                        async def _invoke_prompt() -> str:
                                            raw = await asyncio.to_thread(
                                                llm_instance.invoke,
                                                prompt_text,
                                            )
                                            return _extract_answer_text(raw).strip()

                                        call = _invoke_prompt

                                    try:
                                        if LLM_TIMEOUT is None:
                                            answer = await call()
                                        else:
                                            answer = await asyncio.wait_for(
                                                call(),
                                                timeout=LLM_TIMEOUT,
                                            )
                                    except asyncio.TimeoutError:
                                        logger.warning(
                                            "[警告] LLM回答生成がタイムアウト: model=%s, timeout=%s, question=%s...",
                                            resolved_llm_model,
                                            _fmt(LLM_TIMEOUT),
                                            q[:30],
                                        )
                                        answer = "[LLMタイムアウト]"
                                    except Exception as e:  # noqa: BLE001
                                        logger.warning("[警告] LLM回答生成失敗: %s", e)
                                        answer = "[LLMエラー]"
                                    return context_texts, answer

                            _t0_rag = _tnow()
                            results_list = await asyncio.gather(
                                *[get_context_and_answer(q) for q in questions],
                            )
                            _tlog("retrieval+llm_answers", _t0_rag)
                            for context_texts, ans in results_list:
                                contexts.append(context_texts)
                                pred_answers.append(ans)
                            logger.info("[進捗] RAG回答生成完了。評価処理を開始...")

                        # RAGAS等で自動評価
                        _check_cancel()
                        logger.info("[進捗] 評価メトリクスの計算を開始...")

                        dataset_dict: dict[str, Any] = {
                            "question": questions,
                            "answer": pred_answers,
                            "contexts": contexts,
                        }
                        if has_reference and ground_truth is not None:
                            dataset_dict["ground_truth"] = ground_truth
                            dataset_dict["reference"] = ground_truth
                        dataset = Dataset.from_dict(dataset_dict)

                        llm_instance_eval = get_llm_eval(evaluation_llm_model)
                        ragas_llm = RAGASLLMAsyncAdapter(llm_instance_eval)

                        metric_defs = [
                            ("faithfulness", faithfulness),
                            ("answer_relevancy", answer_relevancy),
                            ("context_recall", context_recall),
                            ("context_precision", context_precision),
                            ("answer_correctness", answer_correctness),
                            ("answer_similarity", answer_similarity),
                        ]
                        selected_metric_defs = []
                        for name, metric in metric_defs:
                            if name in {"answer_correctness", "answer_similarity"} and not has_reference:
                                continue
                            if name == "answer_similarity" and not include_answer_similarity:
                                continue
                            selected_metric_defs.append((name, metric))
                        metrics_local = [
                            deepcopy(m) for _, m in selected_metric_defs
                        ]
                        for m in metrics_local:
                            if hasattr(m, "llm"):
                                m.llm = ragas_llm
                            if hasattr(m, "embeddings"):
                                m.embeddings = embedder

                        eval_df = None
                        try:
                            _t0_eval = _tnow()
                            if EVAL_TIMEOUT is None:
                                eval_df = await asyncio.to_thread(
                                    _run_ragas_evaluate_to_pandas,
                                    dataset=dataset,
                                    metrics=metrics_local,
                                    llm=ragas_llm,
                                    embeddings=embedder,
                                    max_workers=max_workers_eval,
                                    timeout=EVAL_TIMEOUT,
                                )
                            else:
                                eval_df = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        _run_ragas_evaluate_to_pandas,
                                        dataset=dataset,
                                        metrics=metrics_local,
                                        llm=ragas_llm,
                                        embeddings=embedder,
                                        max_workers=max_workers_eval,
                                        timeout=EVAL_TIMEOUT,
                                    ),
                                    timeout=EVAL_TIMEOUT,
                                )
                            _tlog("ragas.evaluate", _t0_eval)
                        except asyncio.TimeoutError:
                            logger.warning(
                                "[警告] ragas.evaluate 一括評価がタイムアウト: timeout=%s",
                                _fmt(EVAL_TIMEOUT),
                            )
                            eval_df = None
                        except TypeError:
                            try:
                                if EVAL_TIMEOUT is None:
                                    eval_df = await asyncio.to_thread(
                                        _run_ragas_evaluate_to_pandas,
                                        dataset=dataset,
                                        metrics=metrics_local,
                                        llm=ragas_llm,
                                        embeddings=embedder,
                                        max_workers=max_workers_eval,
                                        timeout=EVAL_TIMEOUT,
                                    )
                                else:
                                    eval_df = await asyncio.wait_for(
                                        asyncio.to_thread(
                                            _run_ragas_evaluate_to_pandas,
                                            dataset=dataset,
                                            metrics=metrics_local,
                                            llm=ragas_llm,
                                            embeddings=embedder,
                                            max_workers=max_workers_eval,
                                            timeout=EVAL_TIMEOUT,
                                        ),
                                        timeout=EVAL_TIMEOUT,
                                    )
                            except TypeError:
                                try:
                                    if EVAL_TIMEOUT is None:
                                        eval_df = await asyncio.to_thread(
                                            _run_ragas_evaluate_to_pandas,
                                            dataset=dataset,
                                            metrics=metrics_local,
                                            llm=ragas_llm,
                                            embeddings=embedder,
                                            max_workers=max_workers_eval,
                                            timeout=None,
                                        )
                                    else:
                                        eval_df = await asyncio.wait_for(
                                            asyncio.to_thread(
                                                _run_ragas_evaluate_to_pandas,
                                                dataset=dataset,
                                                metrics=metrics_local,
                                                llm=ragas_llm,
                                                embeddings=embedder,
                                                max_workers=max_workers_eval,
                                                timeout=None,
                                            ),
                                            timeout=EVAL_TIMEOUT,
                                        )
                                except Exception:  # noqa: BLE001
                                    logger.warning("[警告] ragas.evaluate 一括評価フォールバック失敗")
                                    eval_df = None

                        metrics_keys = [name for name, _ in selected_metric_defs]
                        metrics_per_qa: list[dict[str, Any]] = []
                        metrics_avg = {k: 0.0 for k in metrics_keys}
                        metrics_error: str | None = None
                        try:
                            if eval_df is not None:
                                # eval_dfの列情報と answer_similarity のサンプル値をデバッグ出力
                                try:
                                    cols = list(getattr(eval_df, "columns", []))
                                    logger.info("[DEBUG] ragas eval_df columns: %s", cols)
                                    if "answer_similarity" not in cols and "semantic_similarity" in cols:
                                        try:
                                            eval_df["answer_similarity"] = eval_df["semantic_similarity"]  # type: ignore[index]
                                            cols = list(getattr(eval_df, "columns", []))
                                            logger.info("[DEBUG] mapped semantic_similarity column to answer_similarity")
                                        except Exception:  # noqa: BLE001
                                            logger.info("[DEBUG] mapping semantic_similarity to answer_similarity failed")

                                    if "answer_similarity" in cols:
                                        try:
                                            sample_vals = eval_df["answer_similarity"].tolist()  # type: ignore[index]
                                            logger.info(
                                                "[DEBUG] answer_similarity sample (first 5): %s",
                                                sample_vals[:5],
                                            )
                                        except Exception:  # noqa: BLE001
                                            logger.info(
                                                "[DEBUG] answer_similarity column present but sampling failed",
                                            )
                                except Exception:  # noqa: BLE001
                                    logger.info("[DEBUG] eval_df column inspection failed")

                                try:
                                    rows = eval_df.to_dict(orient="records")  # type: ignore[arg-type]
                                except Exception:  # noqa: BLE001
                                    rows = []
                                for idx_row, r in enumerate(rows):
                                    metric_values = {
                                        k: safe_val(r.get(k, 0.0)) for k in metrics_keys
                                    }
                                    metrics_per_qa.append(
                                        {
                                            "question": (
                                                questions[idx_row]
                                                if idx_row < len(questions)
                                                else ""
                                            ),
                                            "pred_answer": (
                                                pred_answers[idx_row]
                                                if idx_row < len(pred_answers)
                                                else ""
                                            ),
                                            "ground_truth": (
                                                ground_truth[idx_row]
                                                if ground_truth is not None and idx_row < len(ground_truth)
                                                else ""
                                            ),
                                            "metrics": metric_values,
                                        },
                                    )
                                for k in metrics_keys:
                                    try:
                                        if hasattr(eval_df, "columns") and k in list(
                                            eval_df.columns,  # type: ignore[attr-defined]
                                        ):
                                            raw_mean = float(eval_df[k].mean())  # type: ignore[index]
                                            metrics_avg[k] = safe_val(raw_mean)
                                        else:
                                            metrics_avg[k] = 0.0
                                    except Exception:  # noqa: BLE001
                                        metrics_avg[k] = 0.0

                                try:
                                    cols_all = list(getattr(eval_df, "columns", []))
                                except Exception:  # noqa: BLE001
                                    cols_all = []
                                metric_cols = [k for k in metrics_keys if k in cols_all]
                                if metric_cols:
                                    any_finite_metric = False
                                    for col in metric_cols:
                                        try:
                                            values = eval_df[col].tolist()  # type: ignore[index]
                                        except Exception:  # noqa: BLE001
                                            continue
                                        for v in values:
                                            try:
                                                fv = float(v)
                                            except Exception:  # noqa: BLE001
                                                continue
                                            if not (math.isnan(fv) or math.isinf(fv)):
                                                any_finite_metric = True
                                                break
                                        if any_finite_metric:
                                            break
                                    if not any_finite_metric:
                                        metrics_error = (
                                            "RAGAS評価メトリクスが取得できませんでした（全てNaN/inf）。"
                                            "評価LLMのレート制限(429)・利用上限、またはモデル応答を確認してください。"
                                            f" evaluation_llm_model={resolved_eval_llm_model}"
                                        )
                            else:
                                metrics_per_qa = [
                                    {
                                        "question": (
                                            questions[idx]
                                            if idx < len(questions)
                                            else ""
                                        ),
                                        "pred_answer": (
                                            pred_answers[idx]
                                            if idx < len(pred_answers)
                                            else ""
                                        ),
                                        "ground_truth": (
                                            ground_truth[idx]
                                            if ground_truth is not None and idx < len(ground_truth)
                                            else ""
                                        ),
                                        "metrics": {k: 0.0 for k in metrics_keys},
                                        "retrieval_conditions": {
                                            "top_k": top_k,
                                            "use_mmr": use_mmr,
                                            "fetch_k": fetch_k,
                                            "lambda_mult": lambda_mult,
                                        },
                                    }
                                    for idx in range(len(questions))
                                ]
                                metrics_avg = {k: 0.0 for k in metrics_keys}
                        except Exception:  # noqa: BLE001
                            metrics_per_qa = [
                                {
                                    "question": (
                                        questions[idx]
                                        if idx < len(questions)
                                        else ""
                                    ),
                                    "pred_answer": (
                                        pred_answers[idx]
                                        if idx < len(pred_answers)
                                        else ""
                                    ),
                                    "ground_truth": (
                                        ground_truth[idx]
                                        if ground_truth is not None and idx < len(ground_truth)
                                        else ""
                                    ),
                                    "metrics": {k: 0.0 for k in metrics_keys},
                                }
                                for idx in range(len(questions))
                            ]
                            metrics_avg = {k: 0.0 for k in metrics_keys}

                        if metrics_error:
                            raise RuntimeError(metrics_error)

                        overall_score = (
                            metrics_avg.get("answer_relevancy", 0.0) * 0.25
                            + metrics_avg.get("faithfulness", 0.0) * 0.25
                            + metrics_avg.get("context_precision", 0.0) * 0.2
                            + metrics_avg.get("context_recall", 0.0) * 0.2
                            + metrics_avg.get("answer_correctness", 0.0) * 0.1
                        )
                        overall_score = safe_val(overall_score)

                        num_chunks = len(chunks)
                        avg_chunk_len = (
                            int(sum(len(c) for c in chunks) / num_chunks)
                            if num_chunks > 0
                            else 0
                        )

                        required_keys = {
                            "overall_score",
                            "avg_chunk_len",
                            "num_chunks",
                        }
                        required_keys.update(metrics_keys)
                        if include_answer_similarity:
                            required_keys.add("answer_similarity")

                        logger.info(
                            "[進捗] 評価メトリクスの計算が完了しました。総合スコア: %.4f",
                            overall_score,
                        )
                        response_dict: dict[str, Any] = {
                            "embedding_model": embedding_model,
                            "llm_model": resolved_llm_model,
                            "rag_prompt_style": rag_prompt_style,
                            "file_id": one.get("file_id"),
                            "pdf_file_id": one.get("pdf_file_id"),
                            "question_mode": one.get("question_mode"),
                            "use_n": one.get("use_n"),
                            "top_k": top_k,
                            "use_mmr": use_mmr,
                            "fetch_k": fetch_k,
                            "lambda_mult": lambda_mult,
                            "chunk_size": (
                                chunk_size_val if chunk_method != "semantic" else None
                            ),
                            "chunk_overlap": (
                                chunk_overlap_val
                                if chunk_method != "semantic"
                                else None
                            ),
                            "chunk_method": chunk_method,
                            "overall_score": overall_score,
                            "chunk_strategy": chunk_strategy,
                            "num_chunks": num_chunks,
                            "avg_chunk_len": avg_chunk_len,
                            "metrics": metrics_per_qa,
                            "force_llm_generation": force_llm_generation,
                            "evaluation_llm_model": resolved_eval_llm_model,
                        }

                        for metric_name, metric_value in metrics_avg.items():
                            response_dict[metric_name] = metric_value

                        if include_answer_similarity and "answer_similarity" not in response_dict:
                            response_dict["answer_similarity"] = None

                        if chunk_method == "semantic":
                            response_dict["similarity_threshold"] = similarity_threshold

                        for k in required_keys:
                            if k not in response_dict:
                                response_dict[k] = 0.0

                        duration_seconds = _tnow() - job_start_time
                        logger.info(
                            "[進捗] チャンク方法 '%s' の処理が完了しました。スコア: %.4f 所要時間: %.2fs",
                            chunk_method,
                            overall_score,
                            duration_seconds,
                        )
                        response_dict["duration_seconds"] = duration_seconds
                        results.append(response_dict)
                        try:
                            ragas_logger.info(
                                "[RAGAS] evaluate_one_bulk done",
                                extra={
                                    "component": "ragas",
                                    "endpoint": "bulk_evaluate",
                                    "job_id": job_id,
                                    "file_id": one.get("file_id"),
                                    "pdf_file_id": one.get("pdf_file_id"),
                                    "llm_model": resolved_llm_model,
                                    "embedding_model": embedding_model,
                                    "chunk_method": chunk_method,
                                    "overall_score": overall_score,
                                    "duration_seconds": duration_seconds,
                                },
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    except Exception as e:  # noqa: BLE001
                        import traceback

                        error_detail = traceback.format_exc()
                        duration_seconds = _tnow() - job_start_time
                        logger.error(
                            "[エラー] チャンク方法 '%s' の処理中にエラーが発生しました: %s (%.2fs)",
                            chunk_method,
                            e,
                            duration_seconds,
                        )
                        try:
                            ragas_logger.error(
                                "[RAGAS] evaluate_one_bulk error",
                                extra={
                                    "component": "ragas",
                                    "endpoint": "bulk_evaluate",
                                    "job_id": job_id,
                                    "file_id": one.get("file_id"),
                                    "pdf_file_id": one.get("pdf_file_id"),
                                    "llm_model": resolved_llm_model,
                                    "embedding_model": embedding_model,
                                    "chunk_method": chunk_method,
                                    "duration_seconds": duration_seconds,
                                    "error": str(e),
                                    "error_detail": error_detail,
                                },
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        logger.debug(error_detail)
                        results.append(
                            {
                                "error": str(e),
                                "error_detail": error_detail,
                                "input_data": one,
                                "duration_seconds": duration_seconds,
                            },
                        )

                logger.info(
                    "[進捗] すべてのチャンク方法の評価が完了しました。結果数: %d",
                    len(results),
                )
                return results
            except Exception as e:  # noqa: BLE001
                # ユーザーキャンセルは上位に伝播させる
                try:
                    from . import evaluation_job_service as eval_job_service  # 遅延インポート

                    if isinstance(e, eval_job_service.BulkJobCancelled):
                        raise
                except Exception:  # noqa: BLE001
                    pass

                import traceback

                error_detail = traceback.format_exc()
                logger.error("[重要エラー] evaluate_one_bulk処理全体で例外が発生: %s", e)
                logger.debug(error_detail)
                try:
                    ragas_logger.error(
                        "[RAGAS] evaluate_one_bulk fatal",
                        extra={
                            "component": "ragas",
                            "endpoint": "bulk_evaluate",
                            "job_id": job_id,
                            "file_id": one.get("file_id"),
                            "pdf_file_id": one.get("pdf_file_id"),
                            "error": str(e),
                            "error_detail": error_detail,
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
                return {
                    "error": str(e),
                    "error_detail": error_detail,
                    "input_data": one,
                }

        # --- 本体分岐 ---
        logger.info(
            "[進捗] bulk_evaluate APIが呼び出されました",
            extra={"component": "evaluation", "endpoint": "bulk_evaluate"},
        )
        ragas_logger.info(
            "[RAGAS] bulk_evaluate request received",
            extra={"component": "ragas", "endpoint": "bulk_evaluate", "job_id": job_id},
        )
        if isinstance(data, list):
            total = len(data)
            logger.info(
                "[進捗] リストデータを処理します。データ数: %d, MAX_PARALLEL_CONFIGS=%d",
                total,
                max(MAX_PARALLEL_CONFIGS_OPENAI, MAX_PARALLEL_CONFIGS_OLLAMA),
                extra={"component": "evaluation", "endpoint": "bulk_evaluate"},
            )

            if total > 0:
                _update_job_progress_safe(
                    f"0/{total} 件完了（RAGAS一括評価を開始しました…）",
                )

            # 外側ループ（設定ごと）の並列実行。順序は元の data の順序を維持する。
            results_all: list[Any] = [None] * total
            completed = 0
            completed_lock = asyncio.Lock()

            async def _process_one(index: int, d: Any) -> None:
                nonlocal completed
                try:
                    provider = _pick_provider_from_config(d, resolve_provider=_resolve_provider)
                    if provider == "openai":
                        outer_semaphore = outer_semaphore_openai
                    else:
                        outer_semaphore = outer_semaphore_ollama

                    async with outer_semaphore:
                        logger.info(
                            "[進捗] データ %d/%d を処理中...",
                            index + 1,
                            total,
                            extra={
                                "component": "evaluation",
                                "endpoint": "bulk_evaluate",
                            },
                        )
                        one = d
                        if not isinstance(one, dict):
                            one = find_first_dict(one)
                        # 設定ごとの処理開始前にもキャンセル要求を確認
                        _check_cancel()
                        res = await evaluate_one_bulk(one)
                        results_all[index] = res
                        # 評価処理が完了した直後にもキャンセル要求を確認
                        _check_cancel()
                        logger.info(
                            "[進捗] データ %d/%d の処理が完了しました",
                            index + 1,
                            total,
                            extra={
                                "component": "evaluation",
                                "endpoint": "bulk_evaluate",
                            },
                        )
                except Exception as e:  # noqa: BLE001
                    # ユーザーキャンセルは上位に伝播させる
                    try:
                        from . import evaluation_job_service as eval_job_service  # 遅延インポート

                        if isinstance(e, eval_job_service.BulkJobCancelled):
                            raise
                    except Exception:  # noqa: BLE001
                        pass

                    import traceback

                    error_detail = traceback.format_exc()
                    logger.error(
                        "[エラー] データ %d/%d の処理中にエラーが発生: %s",
                        index + 1,
                        total,
                        e,
                        extra={
                            "component": "evaluation",
                            "endpoint": "bulk_evaluate",
                        },
                    )
                    logger.debug(error_detail)
                    results_all[index] = {
                        "error": str(e),
                        "error_detail": error_detail,
                        "input_data": d,
                    }
                finally:
                    if total > 0:
                        async with completed_lock:
                            completed += 1
                            done = completed
                        _update_job_progress_safe(
                            f"{done}/{total} 件完了（RAGAS一括評価を実行中…）",
                        )

            # すべてのタスクを起動し、MAX_PARALLEL_CONFIGS で同時実行数を制限
            tasks = [
                asyncio.create_task(_process_one(i, d)) for i, d in enumerate(data)
            ]
            await asyncio.gather(*tasks)

            # None が残っている場合は安全側でエラー扱いにしておく
            for i in range(total):
                if results_all[i] is None:
                    results_all[i] = {
                        "error": "unknown error (no result)",
                        "input_data": data[i],
                    }

            logger.info(
                "[進捗] すべてのデータ処理が完了しました。結果数: %d",
                len(results_all),
                extra={"component": "evaluation", "endpoint": "bulk_evaluate"},
            )
            # 実験全体の設定概要を保存
            experiment_params = _summarize_bulk_request([d for d in data if isinstance(d, dict)])
            persist_experiment_results(
                pdf_file_id=experiment_params.get("file_id") or (data[0].get("file_id") if data and isinstance(data[0], dict) else None),
                request_params=experiment_params or (data[0] if data and isinstance(data[0], dict) else {}),
                results=results_all,
            )
            return results_all

        logger.info(
            "[進捗] 単一データを処理します",
            extra={"component": "evaluation", "endpoint": "bulk_evaluate"},
        )
        one_dict = data if isinstance(data, dict) else find_first_dict(data)
        result = await evaluate_one_bulk(one_dict)
        logger.info(
            "[進捗] 処理が完了しました",
            extra={"component": "evaluation", "endpoint": "bulk_evaluate"},
        )
        _update_job_progress_safe("1/1 件完了（RAGAS一括評価を実行中…）")
        persist_experiment_results(
            pdf_file_id=one_dict.get("file_id"),
            request_params=one_dict,
            results=result if isinstance(result, list) else [result],
        )
        return result

    except Exception as e:  # noqa: BLE001
        # ユーザーキャンセルはワーカー側で専用ステータスとして扱うため、そのまま再送出する
        try:
            from . import evaluation_job_service as eval_job_service  # 遅延インポート

            if isinstance(e, eval_job_service.BulkJobCancelled):
                raise
        except Exception:  # noqa: BLE001
            pass

        # 異常時も辞書を直接返す
        import traceback

        error_detail = traceback.format_exc()
        logger.error("[重要エラー] bulk_evaluate全体例外: %s", e)
        logger.debug(error_detail)
        return {
            "error": str(e),
            "error_detail": error_detail,
        }
