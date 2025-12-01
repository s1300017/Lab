from __future__ import annotations

"""チャンキング関連のユーティリティ関数群。

main.py から切り出した固定長・文単位・段落単位・セマンティックチャンキングなどを提供する。
"""

from typing import List
import logging
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


def fixed_chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> List[str]:
    """固定長でテキストをチャンク分割するユーティリティ。"""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap if chunk_overlap < chunk_size else chunk_size
    return chunks


def sentence_chunk_text(text: str) -> List[str]:
    """spaCyの日本語モデルで文単位に分割する。"""
    try:
        import spacy

        try:
            nlp = spacy.load("ja_core_news_sm")
        except OSError:
            raise RuntimeError(
                "spaCyの日本語モデル 'ja_core_news_sm' がインストールされていません。\n\n"
                "python -m spacy download ja_core_news_sm\n"
            )
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"spaCyによる日本語文分割時にエラー: {str(e)}") from e


def paragraph_chunk_text(text: str) -> List[str]:
    """空行区切りで段落単位に分割する。"""
    if not text:
        return []
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    return paragraphs if paragraphs else [text]


def semantic_chunk_text(
    text: str,
    chunk_size=None,  # 互換性のため引数だけ残す
    chunk_overlap=None,
    embedding_model=None,
    similarity_threshold: float = 0.7,
) -> List[str]:
    """意味的なまとまりでチャンク分割を行う。"""
    try:
        import spacy

        nlp = spacy.load("ja_core_news_sm")
    except OSError:
        raise RuntimeError(
            "spaCyの日本語モデル 'ja_core_news_sm' がインストールされていません。\n\n"
            "python -m spacy download ja_core_news_sm\n"
        )

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return [text]

    logger.info("セマンティックチャンキング: %d文を処理中...", len(sentences))
    if embedding_model is None:
        raise ValueError("embedding_modelが指定されていません")

    batch_size = 32
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)

    chunks: List[str] = []
    current_chunk: List[str] = []
    for i in range(len(sentences)):
        current_sentence = sentences[i]
        current_embedding = np.array(embeddings[i]).reshape(1, -1)
        if not current_chunk:
            current_chunk.append(current_sentence)
            continue
        last_embedding = np.array(embeddings[i - 1]).reshape(1, -1)
        similarity = cosine_similarity(last_embedding, current_embedding)[0][0]
        if similarity < similarity_threshold:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [current_sentence]
        else:
            current_chunk.append(current_sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    logger.info("セマンティックチャンキング完了: %d個のチャンクを生成", len(chunks))
    return chunks


def generate_default_chunks_for_storage(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """永続化用の固定長チャンクを生成するフォールバックユーティリティ。"""
    if not text:
        return []
    try:
        chunks = fixed_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunks if chunks else [text]
    except Exception as e:  # noqa: BLE001
        logger.warning("[警告] generate_default_chunks_for_storageで例外: %s。全文を1チャンクとして保存します。", e)
        return [text]


def calculate_overlap_metrics(contexts: list[list[str]], embedder=None) -> dict:
    """複数のオーバーラップメトリクスを計算するユーティリティ。

    Args:
        contexts: コンテキストのリスト
        embedder: オプションの埋め込みモデル（セマンティックオーバーラップ用）

    Returns:
        dict: 各種オーバーラップメトリクスを含む辞書
    """
    if not contexts or len(contexts) < 2:
        return {
            "overlap_ratio": 0.0,
            "adjacent_overlap": [0.0],
            "semantic_overlap": 0.0,
        }

    # 1. 元のオーバーラップ計算（後方互換性のため保持）
    all_tokens: list[str] = []
    for ctx in contexts:
        if isinstance(ctx, str):
            all_tokens.extend(ctx.split())
        else:
            for text in ctx:
                all_tokens.extend(text.split())

    unique_tokens = set(all_tokens)
    total_tokens = len(all_tokens)
    unique_count = len(unique_tokens)

    overlap_ratio = 1.0 - (unique_count / total_tokens) if total_tokens > 0 else 0.0

    # 2. 隣接チャンク間のオーバーラップ
    adjacent_overlaps: list[float] = []
    for i in range(len(contexts) - 1):
        current_ctx = contexts[i] if isinstance(contexts[i], list) else [contexts[i]]
        next_ctx = contexts[i + 1] if isinstance(contexts[i + 1], list) else [contexts[i + 1]]

        current_tokens = set(" ".join(current_ctx).split())
        next_tokens = set(" ".join(next_ctx).split())

        common_tokens = current_tokens.intersection(next_tokens)
        min_len = min(len(current_tokens), len(next_tokens))

        overlap = len(common_tokens) / min_len if min_len > 0 else 0.0
        adjacent_overlaps.append(overlap)

    # 3. セマンティックオーバーラップ（埋め込みモデルが利用可能な場合）
    semantic_overlap = 0.0
    if embedder and len(contexts) > 1:
        try:
            chunk_texts = [" ".join(ctx) if isinstance(ctx, list) else ctx for ctx in contexts]
            embeddings = embedder.embed_documents(chunk_texts)

            similarities: list[float] = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity(
                    [embeddings[i]],
                    [embeddings[i + 1]],
                )[0][0]
                similarities.append(float(sim))

            semantic_overlap = sum(similarities) / len(similarities) if similarities else 0.0
        except Exception as e:  # noqa: BLE001
            logger.warning("セマンティックオーバーラップの計算中にエラーが発生しました: %s", e)
            semantic_overlap = 0.0

    return {
        "overlap_ratio": overlap_ratio,
        "adjacent_overlap": adjacent_overlaps,
        "avg_adjacent_overlap": sum(adjacent_overlaps) / len(adjacent_overlaps) if adjacent_overlaps else 0.0,
        "semantic_overlap": semantic_overlap,
    }
