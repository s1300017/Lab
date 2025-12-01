from __future__ import annotations

import sys
from pathlib import Path

import asyncio
import pytest

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.evaluation_service as evaluation_service  # type: ignore  # noqa: E402


@pytest.mark.asyncio
async def test_bulk_evaluate_semantic_chunk_uses_chunk_cache(monkeypatch) -> None:
    """semantic チャンキングで _CHUNK_CACHE が利用されることを確認する。"""

    from app import main as main_module  # type: ignore  # noqa: E402

    # DB 書き込みを無効化
    monkeypatch.setattr(evaluation_service, "persist_experiment_results", lambda *a, **k: None)

    # FAISS をダミー化して DB 接続を避ける
    class DummyVS:
        def __init__(self, *a, **k):  # noqa: ANN001
            pass

        def as_retriever(self, *a, **k):  # noqa: ANN001
            class DummyRetriever:
                def get_relevant_documents(self, query):  # noqa: ANN001
                    class Doc:
                        def __init__(self, text: str) -> None:
                            self.page_content = text

                    return [Doc("ctx1")]

            return DummyRetriever()

    class DummyFAISS:
        @classmethod
        def from_texts(cls, texts, embedding):  # noqa: ANN001
            return DummyVS()

    monkeypatch.setattr(evaluation_service, "_FAISS_AVAILABLE", True)
    monkeypatch.setattr(evaluation_service, "FAISS", DummyFAISS)

    # main 側の依存をダミー化
    monkeypatch.setattr(main_module, "SUPPORTED_EMBEDDING_MODELS", {"emb-x"})
    monkeypatch.setattr(main_module, "_CHUNK_CACHE", {})
    monkeypatch.setattr(main_module, "get_embeddings", lambda name: "EMB")
    monkeypatch.setattr(main_module, "get_llm", lambda name: object())

    # ragas.evaluate を軽量化
    def fake_evaluate(*args, **kwargs):  # noqa: ANN001
        return 1

    monkeypatch.setattr(evaluation_service, "evaluate", fake_evaluate)
    monkeypatch.setattr(evaluation_service, "RAGASLLMAsyncAdapter", lambda llm: llm)

    # semantic_chunk_text の呼び出し回数をカウント
    call_counter = {"count": 0}

    def fake_semantic_chunk_text(text, chunk_size, chunk_overlap, embedding_model, similarity_threshold=0.7):  # noqa: ANN001
        call_counter["count"] += 1
        return ["c1", "c2"]

    monkeypatch.setattr(evaluation_service, "semantic_chunk_text", fake_semantic_chunk_text)

    payload = {
        "text": "dummy text",
        "embedding_model": "emb-x",
        "chunk_methods": ["semantic"],
        "questions": ["Q1"],
        "answers": ["A1"],
    }

    res1 = await evaluation_service.bulk_evaluate(payload)
    res2 = await evaluation_service.bulk_evaluate(payload)

    assert isinstance(res1, list)
    assert isinstance(res2, list)
    # 2 回呼び出しても semantic_chunk_text は 1 回のみ実行される（キャッシュヒット）
    assert call_counter["count"] == 1


@pytest.mark.asyncio
async def test_bulk_evaluate_handles_timeout_error(monkeypatch) -> None:
    """ragas.evaluate 相当が TimeoutError を投げても例外を外に伝播しないことを確認する。"""

    from app import main as main_module  # type: ignore  # noqa: E402

    monkeypatch.setattr(evaluation_service, "persist_experiment_results", lambda *a, **k: None)

    class DummyVS:
        def __init__(self, *a, **k):  # noqa: ANN001
            pass

        def as_retriever(self, *a, **k):  # noqa: ANN001
            class DummyRetriever:
                def get_relevant_documents(self, query):  # noqa: ANN001
                    class Doc:
                        def __init__(self, text: str) -> None:
                            self.page_content = text

                    return [Doc("ctx-timeout")]  # 1 件だけ返す

            return DummyRetriever()

    class DummyFAISS:
        @classmethod
        def from_texts(cls, texts, embedding):  # noqa: ANN001
            return DummyVS()

    monkeypatch.setattr(evaluation_service, "_FAISS_AVAILABLE", True)
    monkeypatch.setattr(evaluation_service, "FAISS", DummyFAISS)

    monkeypatch.setattr(main_module, "SUPPORTED_EMBEDDING_MODELS", {"emb-y"})
    monkeypatch.setattr(main_module, "_CHUNK_CACHE", {})
    monkeypatch.setattr(main_module, "get_embeddings", lambda name: "EMB")
    monkeypatch.setattr(main_module, "get_llm", lambda name: object())

    # ragas.evaluate 自体を TimeoutError を送出するダミーにする
    def timeout_evaluate(*args, **kwargs):  # noqa: ANN001
        raise asyncio.TimeoutError()

    monkeypatch.setattr(evaluation_service, "evaluate", timeout_evaluate)
    monkeypatch.setattr(evaluation_service, "RAGASLLMAsyncAdapter", lambda llm: llm)

    payload = {
        "text": "dummy text timeout",
        "embedding_model": "emb-y",
        "chunk_methods": ["recursive"],
        "chunk_size": 100,
        "chunk_overlaps": [0],
        "questions": ["Q"],
        "answers": ["A"],
    }

    res = await evaluation_service.bulk_evaluate(payload)

    # TimeoutError は catch され、結果リストが返ってくることを確認
    assert isinstance(res, list)
    assert res
    assert "overall_score" in res[0]
    assert "metrics" in res[0]
