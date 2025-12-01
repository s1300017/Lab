from __future__ import annotations

from pathlib import Path
from typing import Any

import hashlib
import json
import textwrap
import logging

from fastapi import HTTPException
from sqlalchemy import create_engine, text

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector

from .chunk_utils import (
    fixed_chunk_text,
    sentence_chunk_text,
    paragraph_chunk_text,
    semantic_chunk_text,
)
from .llm_ragas_utils import _extract_answer_text
from .settings import DB_URL, REQUEST_ID_CTX


logger = logging.getLogger(__name__)


# データベース接続設定（main.py と同等の環境変数を利用）
engine = create_engine(DB_URL)

# PDF・抽出データ保存用ディレクトリ
DATA_DIR = Path(__file__).parent.parent / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"


# --- main.py の機能をラップするヘルパ ---


def _get_embeddings(model_name: str):
    """main.get_embeddings を遅延インポートで呼び出すヘルパ。"""
    from . import main as main_module

    return main_module.get_embeddings(model_name)


def _get_collection_name(embedding_model: str) -> str:
    """main.get_collection_name を遅延インポートで呼び出すヘルパ。"""
    from . import main as main_module

    return main_module.get_collection_name(embedding_model)


def _build_collection_name_for_pdf(
    embedding_model: str,
    scope: str,
    pdf_file_id: str | None = None,
) -> str:
    """main.build_collection_name_for_pdf を遅延インポートで呼び出すヘルパ。"""
    from . import main as main_module

    return main_module.build_collection_name_for_pdf(embedding_model, scope, pdf_file_id)


def _init_generation_llm(model_name: str, purpose: str = "/query"):
    """main.init_generation_llm を遅延インポートで呼び出すヘルパ。"""
    from . import main as main_module

    return main_module.init_generation_llm(model_name, purpose=purpose)


# --- チャット履歴永続化ヘルパ（main.py / chat_api.py から移植） ---


def _persist_chat_log_and_contexts(
    scope: str,
    request: Any,
    answer: str,
    contexts: list[str],
    resolved_llm: str,
) -> None:
    """chat_logs および chat_contexts への永続化を行うヘルパー。"""

    try:
        # ContextVar から現在の request_id を取得（存在しない場合は None）
        try:
            request_id = REQUEST_ID_CTX.get()
        except LookupError:  # pragma: no cover - 安全側
            request_id = None

        with engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO chat_logs (
                        pdf_file_id,
                        user_message,
                        assistant_message,
                        llm_model_used,
                        embedding_model,
                        scope,
                        request_id
                    )
                    VALUES (
                        :pdf_file_id,
                        :user_message,
                        :assistant_message,
                        :llm_model_used,
                        :embedding_model,
                        :scope,
                        :request_id
                    )
                    RETURNING id
                    """
                ),
                {
                    "pdf_file_id": request.pdf_file_id if scope == "single" else None,
                    "user_message": request.query,
                    "assistant_message": answer,
                    "llm_model_used": resolved_llm,
                    "embedding_model": request.embedding_model,
                    "scope": scope,
                    "request_id": request_id,
                },
            )
            chat_log_id = result.scalar()

            if chat_log_id and contexts:
                try:
                    context_rows = [
                        {
                            "chat_log_id": chat_log_id,
                            "context_index": idx,
                            "content": ctx,
                        }
                        for idx, ctx in enumerate(contexts)
                    ]
                    conn.execute(
                        text(
                            """
                            INSERT INTO chat_contexts (
                                chat_log_id,
                                context_index,
                                content
                            )
                            VALUES (
                                :chat_log_id,
                                :context_index,
                                :content
                            )
                            """
                        ),
                        context_rows,
                    )
                except Exception as e_ctx:  # noqa: BLE001
                    logger.warning(
                        "[WARN] chat_contexts insert failed: %s",
                        e_ctx,
                        extra={"chat_log_id": chat_log_id, "pdf_file_id": request.pdf_file_id},
                    )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[WARN] chat_logs insert failed: %s",
            e,
            extra={"pdf_file_id": request.pdf_file_id},
        )


# --- サービス関数群 ---


def chunk_text(request: Any) -> dict[str, Any]:
    """chunk_method に応じてテキストをチャンク分割するサービスロジック。"""

    if request.chunk_method == "semantic":
        # embedding_modelが指定されていることを確認
        if not request.embedding_model:
            raise HTTPException(
                status_code=400,
                detail="semanticチャンキングにはembedding_modelの指定が必要です",
            )
        try:
            # モデル名から埋め込みインスタンスを生成
            embedder = _get_embeddings(request.embedding_model)
            chunks = semantic_chunk_text(
                text=request.text,
                chunk_size=None,
                chunk_overlap=None,
                embedding_model=embedder,  # インスタンスを渡す
            )
            return {"chunks": chunks}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"テキストのチャンキング中にエラーが発生しました: {str(e)}",
            ) from e
    if request.chunk_method == "recursive":
        # 再帰的な文字数分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(request.text)
        return {"chunks": chunks}
    if request.chunk_method == "fixed":
        # 固定長で分割
        chunks = fixed_chunk_text(
            request.text,
            request.chunk_size,
            request.chunk_overlap,
        )
        return {"chunks": chunks}
    if request.chunk_method == "sentence":
        # 文単位で分割
        chunks = sentence_chunk_text(request.text)
        return {"chunks": chunks}
    if request.chunk_method == "paragraph":
        # 段落単位で分割
        chunks = paragraph_chunk_text(request.text)
        return {"chunks": chunks}

    raise HTTPException(
        status_code=400,
        detail=(
            "未対応のchunk_method: {method}。'recursive', 'fixed', "
            "'semantic', 'sentence', 'paragraph' のいずれかを指定してください。".format(
                method=request.chunk_method,
            )
        ),
    )


def embed_and_store(request: Any) -> dict[str, Any]:
    """テキストチャンクを埋め込み、PGVector に保存するサービスロジック。"""

    try:
        embeddings_instance = _get_embeddings(request.embedding_model)
        vectorstore = PGVector.from_documents(
            documents=[],  # 空のドキュメントで初期化
            embedding=embeddings_instance,
            collection_name=_get_collection_name(request.embedding_model),
        )
        # chunk_methodを全チャンクのmetadataに付与して保存
        chunk_method = getattr(request, "chunk_method", None)
        metadatas = [{"chunk_method": chunk_method} for _ in request.chunks]
        vectorstore.add_texts(texts=request.chunks, metadatas=metadatas)
        return {
            "message": (
                f"Successfully embedded and stored {len(request.chunks)} "
                f"chunks using {request.embedding_model} (method={chunk_method}) ."
            )
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


def build_vectorstore(request: Any) -> dict[str, Any]:
    """pdf_chunks を元に、PDFベースRAG用のベクトルストアを構築するサービスロジック。"""

    scope = (request.scope or "single").lower()
    if scope not in {"single", "all"}:
        raise HTTPException(
            status_code=400,
            detail=f"未対応のscopeです: {scope}。'single' または 'all' を指定してください。",
        )

    if scope == "single" and not request.pdf_file_id:
        raise HTTPException(
            status_code=400,
            detail="scope='single' の場合は pdf_file_id を指定してください。",
        )

    # 埋め込みモデルをロード
    embedder = _get_embeddings(request.embedding_model)

    # チャンク分割用の内部ヘルパ
    def _chunk_text_for_request(text: str) -> list[str]:
        method = (request.chunk_method or "recursive").lower()
        if not text:
            return []
        if method == "semantic":
            # semantic チャンキングは埋め込みモデル必須
            sim_th = (
                request.similarity_threshold
                if request.similarity_threshold is not None
                else 0.7
            )
            return semantic_chunk_text(
                text=text,
                chunk_size=None,
                chunk_overlap=None,
                embedding_model=embedder,
                similarity_threshold=sim_th,
            )
        if method == "recursive":
            size = request.chunk_size or 1000
            overlap = request.chunk_overlap or 200
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                length_function=len,
            )
            return splitter.split_text(text)
        if method == "fixed":
            size = request.chunk_size or 1000
            overlap = request.chunk_overlap or 0
            return fixed_chunk_text(text, size, overlap)
        if method == "sentence":
            return sentence_chunk_text(text)
        if method == "paragraph":
            return paragraph_chunk_text(text)
        raise HTTPException(status_code=400, detail=f"未対応のchunk_methodです: {method}")

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    # pdf_chunks からテキストを収集し、チャンク化する
    try:
        with engine.begin() as conn:
            if scope == "single":
                rows = conn.execute(
                    text(
                        """
                        SELECT content FROM pdf_chunks
                        WHERE pdf_file_id = :fid
                        ORDER BY chunk_index ASC
                        """
                    ),
                    {"fid": request.pdf_file_id},
                ).fetchall()
                if not rows:
                    fallback_text = ""
                    try:
                        extracted_path = EXTRACTED_DIR / f"{request.pdf_file_id}.json"
                        if extracted_path.exists():
                            with extracted_path.open("r", encoding="utf-8") as f:
                                data = json.load(f)
                            raw_text = data.get("text") or ""
                            if raw_text and isinstance(raw_text, str):
                                fallback_text = raw_text
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[WARN] pdf_chunks fallback from extracted JSON failed: %s",
                            e,
                            extra={"pdf_file_id": request.pdf_file_id},
                        )
                    if not fallback_text.strip():
                        raise HTTPException(
                            status_code=404,
                            detail="指定されたPDFのチャンクが見つかりません。",
                        )
                    sample_text = fallback_text
                    chunks = _chunk_text_for_request(sample_text)
                    if not chunks:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "抽出テキストからのチャンク化結果が空です。PDF内容を確認してください。"
                            ),
                        )
                    try:
                        chunk_rows = [
                            {
                                "pdf_file_id": request.pdf_file_id,
                                "chunk_index": idx,
                                "content": ch,
                                "content_hash": hashlib.sha256(
                                    ch.encode("utf-8"),
                                ).hexdigest(),
                            }
                            for idx, ch in enumerate(chunks)
                        ]
                        conn.execute(
                            text(
                                """
                                INSERT INTO pdf_chunks (
                                    pdf_file_id,
                                    chunk_index,
                                    content,
                                    content_hash
                                )
                                VALUES (
                                    :pdf_file_id,
                                    :chunk_index,
                                    :content,
                                    :content_hash
                                )
                                """
                            ),
                            chunk_rows,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[WARN] failed to repopulate pdf_chunks from extracted JSON: %s",
                            e,
                        )
                else:
                    sample_text = "\n".join(r[0] for r in rows if r[0])
                    chunks = _chunk_text_for_request(sample_text)
                for idx, ch in enumerate(chunks):
                    texts.append(ch)
                    metadatas.append(
                        {
                            "pdf_file_id": request.pdf_file_id,
                            "chunk_index": idx,
                            "chunk_method": request.chunk_method,
                            "scope": scope,
                        },
                    )
            else:  # scope == "all"
                pdf_rows = conn.execute(
                    text(
                        """
                        SELECT DISTINCT pdf_file_id
                        FROM pdf_chunks
                        WHERE pdf_file_id IS NOT NULL
                        ORDER BY pdf_file_id
                        """
                    ),
                ).fetchall()
                pdf_ids = [r[0] for r in pdf_rows if r[0]]
                if not pdf_ids:
                    raise HTTPException(
                        status_code=404,
                        detail="pdf_chunks に有効なデータが存在しません。",
                    )

                for fid in pdf_ids:
                    rows = conn.execute(
                        text(
                            """
                            SELECT content FROM pdf_chunks
                            WHERE pdf_file_id = :fid
                            ORDER BY chunk_index ASC
                            """
                        ),
                        {"fid": fid},
                    ).fetchall()
                    if not rows:
                        continue
                    sample_text = "\n".join(r[0] for r in rows if r[0])
                    if not sample_text.strip():
                        continue
                    chunks = _chunk_text_for_request(sample_text)
                    for idx, ch in enumerate(chunks):
                        texts.append(ch)
                        metadatas.append(
                            {
                                "pdf_file_id": fid,
                                "chunk_index": idx,
                                "chunk_method": request.chunk_method,
                                "scope": scope,
                            },
                        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"pdf_chunks 読み取り中にエラーが発生しました: {str(e)}",
        ) from e

    # 収集したテキストからベクトルストアを構築
    try:
        collection_name = _build_collection_name_for_pdf(
            request.embedding_model,
            scope,
            request.pdf_file_id if scope == "single" else None,
        )
    except ValueError as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        vectorstore = PGVector.from_documents(
            documents=[],
            embedding=embedder,
            collection_name=collection_name,
        )
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"ベクトルストア構築中にエラーが発生しました: {str(e)}",
        ) from e

    return {
        "status": "ok",
        "collection_name": collection_name,
        "num_chunks": len(texts),
        "scope": scope,
    }


def query_rag(request: Any) -> dict[str, Any]:
    """事前に構築されたPDFベースのベクトルストアを用いてRAG応答を生成するサービスロジック。"""

    try:
        llm_instance, resolved_llm = _init_generation_llm(
            request.llm_model,
            purpose="/query",
        )
        logger.info(
            "[INFO] /query 生成LLM=%s",
            resolved_llm,
            extra={"component": "chat", "endpoint": "/query"},
        )

        scope = (request.scope or "single").lower()
        if scope not in {"single", "all"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"未対応のscopeです: {scope}。'single' または 'all' を指定してください。"
                ),
            )
        if scope == "single" and not request.pdf_file_id:
            raise HTTPException(
                status_code=400,
                detail="scope='single' の場合は pdf_file_id を指定してください。",
            )

        embeddings_instance = _get_embeddings(request.embedding_model)

        # 対象コレクション名を決定
        try:
            collection_name = _build_collection_name_for_pdf(
                request.embedding_model,
                scope,
                request.pdf_file_id if scope == "single" else None,
            )
        except ValueError as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(e)) from e

        # 既存のベクトルストアを利用（PGVector内でコレクション名を解決）
        try:
            vectorstore = PGVector(
                embedding_function=embeddings_instance,
                collection_name=collection_name,
                connection_string=DB_URL,
                use_jsonb=True,
            )
        except Exception as e:  # noqa: BLE001
            # コレクション未構築などの場合は404として扱う
            raise HTTPException(
                status_code=404,
                detail=(
                    "指定されたベクトルストアが存在しません。先に /build_vectorstore/ を実行してください。"
                    f" collection_name={collection_name}, error={str(e)}"
                ),
            ) from e

        retriever = vectorstore.as_retriever()

        # 関連するドキュメントを取得し、コンテキスト文字列を構築
        retrieved_docs = retriever.get_relevant_documents(request.query)
        contexts = [doc.page_content for doc in retrieved_docs]
        try:
            logger.debug(
                "[DEBUG] /query retrieved_docs=%d, contexts_len=%d",
                len(retrieved_docs),
                len(contexts),
                extra={"component": "chat", "endpoint": "/query"},
            )
            if contexts:
                snippet = contexts[0][:120].replace("\n", " ")
                logger.debug(
                    "[DEBUG] /query first_context_snippet=%r",
                    snippet,
                    extra={"component": "chat", "endpoint": "/query"},
                )
        except Exception:  # noqa: BLE001
            pass

        if not contexts:
            # コンテキストが全く取得できなかった場合は、PDFに基づく回答ができないことを明示
            raise HTTPException(
                status_code=404,
                detail="関連するコンテキストが見つからなかったため、回答を生成できません。",
            )

        # LLM へ質問とコンテキストを渡して回答を生成
        context_text = "\n".join(contexts)
        prompt = textwrap.dedent(
            f"""
            あなたは日本語のRAGシステムにおける回答エンジンです。以下の制約を厳密に守ってください。
            - 提供されたコンテキストに含まれる事実のみを用いて回答すること。
            - 文書内に記載が見つからない場合は「本文に該当記述がありません。」と明示すること。
            - 回答は自然な日本語で2〜3文以内にまとめること。
            - 重要な根拠がある場合はその文を要約して含めること。

            ### コンテキスト
            {context_text}

            ### 質問
            {request.query}

            ### 回答
            """
        ).strip()

        raw_answer = llm_instance.invoke(prompt)
        answer = _extract_answer_text(raw_answer).strip()

        # 単純なフォールバック戦略（空回答や「該当記述がありません」のみなどを避ける）
        try:
            normalized_answer = answer.replace(" ", "").replace("　", "")
            fallback_needed = False
            if not normalized_answer:
                fallback_needed = True
            elif "本文に該当記述がありません" in normalized_answer and contexts:
                fallback_needed = True

            if fallback_needed and contexts:
                try:
                    fallback_prompt = textwrap.dedent(
                        f"""
                        あなたは日本語のRAGシステムにおける回答エンジンです。
                        以下のコンテキストに基づいて、質問に対してできるだけ近い答えを推測して下さい。
                        完全に一致する記述がなくても、関連しそうな情報から妥当な推測を含めて回答してください。
                        「本文に該当記述がありません」のようなメッセージは禁止です。

                        ### コンテキスト
                        {context_text}

                        ### 質問
                        {request.query}

                        ### 回答
                        """
                    ).strip()
                    raw_fallback = llm_instance.invoke(fallback_prompt)
                    fallback_answer = _extract_answer_text(raw_fallback).strip()
                    if fallback_answer:
                        answer = fallback_answer
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[WARN] /query fallback generation failed: %s",
                        e,
                        extra={"component": "chat", "endpoint": "/query"},
                    )
        except Exception:  # noqa: BLE001
            pass

        if not answer:
            answer = "本文に該当記述がありません。"

        _persist_chat_log_and_contexts(scope, request, answer, contexts, resolved_llm)

        return {
            "answer": answer,
            "contexts": contexts,
            "source_documents": [
                {"page_content": doc.page_content} for doc in retrieved_docs
            ],
            "llm_model_used": resolved_llm,
        }

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback

        error_trace = traceback.format_exc()
        logger.error(
            "Error in query_rag: %s",
            error_trace,
            extra={"component": "chat", "endpoint": "/query"},
        )
        raise HTTPException(
            status_code=500,
            detail=f"エラーが発生しました: {str(e)}\n{error_trace}",
        ) from e
