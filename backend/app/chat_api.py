from __future__ import annotations

from pathlib import Path
from typing import Any

import hashlib
import json
import os
import textwrap

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
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
from . import chat_service
from .settings import DB_URL, DATA_DIR, EXTRACTED_DIR


# データベース接続設定（main.py と同等の環境変数を利用）
engine = create_engine(DB_URL)

# PDF・抽出データ保存用ディレクトリ


router = APIRouter()


# --- Pydantic Models（main.py から移植） ---


class ChunkRequest(BaseModel):
    text: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_method: str = "recursive"  # 'recursive' or 'semantic'
    embedding_model: str | None = None  # semantic 用


class EmbedRequest(BaseModel):
    chunks: list[str]
    embedding_model: str  # 埋め込みモデル名
    chunk_method: str  # チャンク方式（recursive, semantic, fixed, sentence, paragraph など）


class QueryRequest(BaseModel):
    query: str
    llm_model: str = "mistral"  # デフォルト値を設定
    embedding_model: str = "huggingface_bge_small"  # デフォルト値を設定
    scope: str = "single"  # "single" または "all"
    pdf_file_id: str | None = None  # scope == "single" のときに対象PDFを指定


class BuildVectorStoreRequest(BaseModel):
    """PDFベースRAG用にベクトルストアを構築するためのリクエストモデル。"""

    scope: str  # "single" または "all"
    pdf_file_id: str | None = None  # scope == "single" のとき必須
    embedding_model: str
    chunk_method: str = "recursive"  # 'recursive', 'fixed', 'semantic', 'sentence', 'paragraph'
    chunk_size: int | None = 1000
    chunk_overlap: int | None = 200
    similarity_threshold: float | None = 0.7  # semantic 用


# --- API Endpoints（/chunk, /embed_and_store, /build_vectorstore, /query） ---


@router.post("/chunk/")
def chunk_text(request: ChunkRequest) -> dict[str, Any]:
    """chunk_method に応じてテキストをチャンク分割するAPI。"""

    return chat_service.chunk_text(request)


@router.post("/embed_and_store/")
def embed_and_store(request: EmbedRequest) -> dict[str, Any]:
    """テキストチャンクを埋め込み、PGVector に保存するAPI。"""

    return chat_service.embed_and_store(request)


@router.post("/build_vectorstore/")
def build_vectorstore(request: BuildVectorStoreRequest) -> dict[str, Any]:
    """pdf_chunks を元に、PDFベースRAG用のベクトルストアを構築するAPI。"""

    return chat_service.build_vectorstore(request)


@router.post("/query/")
def query_rag(request: QueryRequest) -> dict[str, Any]:
    """事前に構築されたPDFベースのベクトルストアを用いてRAG応答を生成するAPI。"""

    return chat_service.query_rag(request)


@router.post("/bulk_evaluate/")
async def bulk_evaluate(request: Request) -> dict[str, Any] | list[Any]:
    """/bulk_evaluate エンドポイントを evaluation_service に委譲するルーター。"""

    from .evaluation_service import bulk_evaluate as eval_bulk

    data = await request.json()
    return await eval_bulk(data)
