from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import statistics
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
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
from .llm_ragas_utils import RAGASLLMAsyncAdapter
from .persistence_utils import persist_experiment_results


logger = logging.getLogger(__name__)


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

    for key in ("experiment_name", "file_id", "pdf_file_id", "question_mode", "use_n"):
        if first.get(key) is not None:
            summary[key] = first.get(key)

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
        # CPUコア数に応じてデフォルト並列数を自動調整（最小2、最大8）
        default_parallel = max(2, min(8, max(1, cpu_count // 2)))
        MAX_PARALLEL_TASKS = int(os.getenv("EVAL_MAX_PARALLEL_TASKS", str(default_parallel)))
        semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)

        # 一括評価の「設定ごと」の並列数（外側ループ用）
        # デフォルトは 1（従来どおり逐次実行）。環境変数で ON にした場合のみ並列化する。
        try:
            MAX_PARALLEL_CONFIGS = int(os.getenv("EVAL_MAX_PARALLEL_CONFIGS", "1"))
        except ValueError:
            MAX_PARALLEL_CONFIGS = 1
        if MAX_PARALLEL_CONFIGS < 1:
            MAX_PARALLEL_CONFIGS = 1

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
                llm_instance_generation, resolved_llm_model = init_generation_llm(
                    request_llm_model,
                    purpose="/bulk_evaluate generation",
                )
                logger.info(
                    "[設定] 生成LLM=%s / 評価LLM=%s / 再生成フラグ=%s",
                    resolved_llm_model,
                    evaluation_llm_model or DEFAULT_LLM_NAME,
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

                questions = one.get("questions")
                # ground_truthキーまたはanswersキーのどちらかを使用（互換性のため）
                answers = one.get("ground_truth", one.get("answers"))
                if not questions or not answers:
                    raise ValueError(
                        "questions/answersが指定されていません。PDFアップロード時の自動生成結果をそのまま送信してください。",
                    )

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
                    and isinstance(answers, list)
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
                    answers = answers[:max_questions]

                if not (sample_text and questions and answers):
                    raise ValueError(
                        "PDFアップロードとQA自動生成を先に実施してください（text, questions, answers必須）。",
                    )

                include_answer_similarity = _bool_env(one.get("include_answer_similarity"), True)

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
                    "[設定] TIMEOUT: LLM_TIMEOUT=%s, EVAL_TIMEOUT=%s, MAX_PARALLEL_TASKS=%d",
                    _fmt(LLM_TIMEOUT),
                    _fmt(EVAL_TIMEOUT),
                    MAX_PARALLEL_TASKS,
                )

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
                            answers
                            and len(answers) == len(questions)
                            and not force_llm_generation
                        ):
                            logger.info(
                                "[進捗] PDFアップロード時の回答を使用（%d個の回答）",
                                len(answers),
                            )
                            pred_answers = list(answers)

                            async def get_context_only(q: str) -> list[str]:
                                async with semaphore:
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
                                async with semaphore:  # セマフォで並列処理数を制限
                                    _check_cancel()
                                    retrieved_docs = await asyncio.to_thread(
                                        retriever.get_relevant_documents,
                                        q,
                                    )
                                    context_texts = [
                                        doc.page_content for doc in retrieved_docs
                                    ]
                                    llm_instance = llm_instance_generation
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

                                    try:
                                        if LLM_TIMEOUT is None:
                                            answer = await chain.ainvoke({})
                                        else:
                                            answer = await asyncio.wait_for(
                                                chain.ainvoke({}),
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

                        dataset_dict = {
                            "question": questions,
                            "answer": pred_answers,
                            "contexts": contexts,
                            "ground_truth": answers,
                        }
                        dataset_dict_with_ref = dict(dataset_dict)
                        dataset_dict_with_ref["reference"] = answers
                        dataset = Dataset.from_dict(dataset_dict_with_ref)

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
                                eval_res_all = await asyncio.to_thread(
                                    evaluate,
                                    dataset=dataset,
                                    metrics=metrics_local,
                                    llm=ragas_llm,
                                    embeddings=embedder,
                                    run_config=RunConfig(
                                        timeout=EVAL_TIMEOUT,
                                        max_workers=MAX_PARALLEL_TASKS,
                                    ),
                                )
                            else:
                                eval_res_all = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        evaluate,
                                        dataset=dataset,
                                        metrics=metrics_local,
                                        llm=ragas_llm,
                                        embeddings=embedder,
                                        run_config=RunConfig(
                                            timeout=EVAL_TIMEOUT,
                                            max_workers=MAX_PARALLEL_TASKS,
                                        ),
                                    ),
                                    timeout=EVAL_TIMEOUT,
                                )
                            _tlog("ragas.evaluate", _t0_eval)
                            try:
                                if hasattr(eval_res_all, "to_pandas"):
                                    eval_df = eval_res_all.to_pandas()
                                elif hasattr(eval_res_all, "to_dict") and hasattr(
                                    eval_res_all,
                                    "columns",
                                ):
                                    eval_df = eval_res_all
                                else:
                                    eval_df = None
                            except Exception:  # noqa: BLE001
                                eval_df = None
                        except asyncio.TimeoutError:
                            logger.warning(
                                "[警告] ragas.evaluate 一括評価がタイムアウト: timeout=%s",
                                _fmt(EVAL_TIMEOUT),
                            )
                            eval_df = None
                        except TypeError:
                            try:
                                if EVAL_TIMEOUT is None:
                                    eval_res_all = await asyncio.to_thread(
                                        evaluate,
                                        dataset=dataset,
                                        metrics=metrics_local,
                                        llm=ragas_llm,
                                        run_config=RunConfig(
                                            timeout=EVAL_TIMEOUT,
                                            max_workers=MAX_PARALLEL_TASKS,
                                        ),
                                    )
                                else:
                                    eval_res_all = await asyncio.wait_for(
                                        asyncio.to_thread(
                                            evaluate,
                                            dataset=dataset,
                                            metrics=metrics_local,
                                            llm=ragas_llm,
                                            run_config=RunConfig(
                                                timeout=EVAL_TIMEOUT,
                                                max_workers=MAX_PARALLEL_TASKS,
                                            ),
                                        ),
                                        timeout=EVAL_TIMEOUT,
                                    )
                                if hasattr(eval_res_all, "to_pandas"):
                                    eval_df = eval_res_all.to_pandas()
                                elif hasattr(eval_res_all, "to_dict") and hasattr(
                                    eval_res_all,
                                    "columns",
                                ):
                                    eval_df = eval_res_all
                                else:
                                    eval_df = None
                            except TypeError:
                                try:
                                    if EVAL_TIMEOUT is None:
                                        eval_res_all = await asyncio.to_thread(
                                            evaluate,
                                            dataset=dataset,
                                            metrics=metrics_local,
                                            llm=ragas_llm,
                                        )
                                    else:
                                        eval_res_all = await asyncio.wait_for(
                                            asyncio.to_thread(
                                                evaluate,
                                                dataset=dataset,
                                                metrics=metrics_local,
                                                llm=ragas_llm,
                                            ),
                                            timeout=EVAL_TIMEOUT,
                                        )
                                    if hasattr(eval_res_all, "to_pandas"):
                                        eval_df = eval_res_all.to_pandas()
                                    elif hasattr(eval_res_all, "to_dict") and hasattr(
                                        eval_res_all,
                                        "columns",
                                    ):
                                        eval_df = eval_res_all
                                    else:
                                        eval_df = None
                                except Exception:  # noqa: BLE001
                                    logger.warning("[警告] ragas.evaluate 一括評価フォールバック失敗")
                                    eval_df = None

                        metrics_keys = [name for name, _ in selected_metric_defs]
                        metrics_per_qa: list[dict[str, Any]] = []
                        metrics_avg = {k: 0.0 for k in metrics_keys}
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
                                                answers[idx_row]
                                                if idx_row < len(answers)
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
                                            metrics_avg[k] = safe_val(
                                                float(eval_df[k].mean()),  # type: ignore[index]
                                            )
                                        else:
                                            metrics_avg[k] = 0.0
                                    except Exception:  # noqa: BLE001
                                        metrics_avg[k] = 0.0
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
                                            answers[idx]
                                            if idx < len(answers)
                                            else ""
                                        ),
                                        "metrics": {k: 0.0 for k in metrics_keys},
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
                                        answers[idx]
                                        if idx < len(answers)
                                        else ""
                                    ),
                                    "metrics": {k: 0.0 for k in metrics_keys},
                                }
                                for idx in range(len(questions))
                            ]
                            metrics_avg = {k: 0.0 for k in metrics_keys}

                        overall_score = (
                            metrics_avg["answer_relevancy"] * 0.25
                            + metrics_avg["faithfulness"] * 0.25
                            + metrics_avg["context_precision"] * 0.2
                            + metrics_avg["context_recall"] * 0.2
                            + metrics_avg["answer_correctness"] * 0.1
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
                            "evaluation_llm_model": evaluation_llm_model
                            or DEFAULT_LLM_NAME,
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
                        logger.debug(error_detail)
                        results.append(
                            {
                                "error": str(e),
                                "chunk_method": chunk_method,
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
        if isinstance(data, list):
            total = len(data)
            logger.info(
                "[進捗] リストデータを処理します。データ数: %d, MAX_PARALLEL_CONFIGS=%d",
                total,
                MAX_PARALLEL_CONFIGS,
                extra={"component": "evaluation", "endpoint": "bulk_evaluate"},
            )

            if total > 0:
                _update_job_progress_safe(
                    f"0/{total} 件完了（RAGAS一括評価を開始しました…）",
                )

            # 外側ループ（設定ごと）の並列実行。順序は元の data の順序を維持する。
            results_all: list[Any] = [None] * total
            outer_semaphore = asyncio.Semaphore(MAX_PARALLEL_CONFIGS)
            completed = 0
            completed_lock = asyncio.Lock()

            async def _process_one(index: int, d: Any) -> None:
                nonlocal completed
                try:
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
