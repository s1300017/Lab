from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.chat_service as chat_service  # type: ignore  # noqa: E402


def test_chunk_text_semantic_requires_embedding_model() -> None:
    """semantic 指定時に embedding_model が必須であることを確認する。"""

    req = SimpleNamespace(text="hello", chunk_method="semantic", embedding_model=None)
    with pytest.raises(HTTPException) as exc:
        chat_service.chunk_text(req)
    assert exc.value.status_code == 400
    assert "embedding_model" in exc.value.detail


def test_chunk_text_semantic_uses_embeddings_and_semantic(monkeypatch) -> None:
    """semantic パスで _get_embeddings と semantic_chunk_text が呼ばれることを確認する。"""

    called: dict[str, object] = {}

    def fake_get_embeddings(name: str):  # noqa: ANN001
        called["emb_name"] = name
        return "EMB"

    def fake_semantic_chunk_text(text: str, chunk_size, chunk_overlap, embedding_model, similarity_threshold=0.7):  # noqa: ANN001
        called["semantic_args"] = {
            "text": text,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
        }
        return ["C1", "C2"]

    monkeypatch.setattr(chat_service, "_get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(chat_service, "semantic_chunk_text", fake_semantic_chunk_text)

    req = SimpleNamespace(text="hello world", chunk_method="semantic", embedding_model="emb-model")
    result = chat_service.chunk_text(req)

    assert result == {"chunks": ["C1", "C2"]}
    assert called["emb_name"] == "emb-model"
    assert called["semantic_args"]["embedding_model"] == "EMB"


def test_chunk_text_invalid_method_raises() -> None:
    """未対応の chunk_method 指定時に 400 が返ることを確認する。"""

    req = SimpleNamespace(text="x", chunk_method="unknown", embedding_model=None)
    with pytest.raises(HTTPException) as exc:
        chat_service.chunk_text(req)
    assert exc.value.status_code == 400


def test_embed_and_store_uses_pgvector(monkeypatch) -> None:
    """embed_and_store が PGVector.from_documents / add_texts を呼び出すことを確認する。"""

    called: dict[str, object] = {}

    def fake_get_embeddings(name: str):  # noqa: ANN001
        called["emb_name"] = name
        return "EMB"

    class DummyVS:
        def __init__(self) -> None:
            self._texts = []

        def add_texts(self, texts, metadatas=None):  # noqa: ANN001
            called["texts"] = texts
            called["metadatas"] = metadatas

        @classmethod
        def from_documents(cls, documents, embedding, collection_name):  # noqa: ANN001
            called["collection_name"] = collection_name
            called["embedding"] = embedding
            return cls()

    monkeypatch.setattr(chat_service, "_get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(chat_service, "PGVector", DummyVS)
    monkeypatch.setattr(chat_service, "_get_collection_name", lambda m: f"col-{m}")

    req = SimpleNamespace(chunks=["a", "b"], embedding_model="emb-x", chunk_method="fixed")
    result = chat_service.embed_and_store(req)

    assert "Successfully embedded and stored 2 chunks" in result["message"]
    assert called["emb_name"] == "emb-x"
    assert called["collection_name"] == "col-emb-x"
    assert called["texts"] == ["a", "b"]
    assert called["metadatas"] == [{"chunk_method": "fixed"}, {"chunk_method": "fixed"}]


def test_build_vectorstore_single_happy_path(monkeypatch) -> None:
    """build_vectorstore が単一 PDF 用にチャンク化と PGVector 構築を行うことを確認する。"""

    called: dict[str, object] = {}

    monkeypatch.setattr(chat_service, "_get_embeddings", lambda name: "EMB")
    monkeypatch.setattr(
        chat_service,
        "_build_collection_name_for_pdf",
        lambda emb, scope, fid=None: f"col-{scope}-{fid or 'all'}",  # noqa: ANN001
    )
    monkeypatch.setattr(
        chat_service,
        "fixed_chunk_text",
        lambda text, size, overlap: [text + "_chunked"],  # noqa: ANN001
    )

    class DummyConn:
        def __enter__(self):  # noqa: D401
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: D401, ANN001
            return None

        def execute(self, stmt, params=None):  # noqa: ANN001
            class Result:
                def fetchall(self_inner):  # noqa: ANN001
                    return [("TEXT1",), ("TEXT2",)]

            return Result()

    class DummyEngine:
        def begin(self):  # noqa: ANN001
            return DummyConn()

    monkeypatch.setattr(chat_service, "engine", DummyEngine())

    class DummyVS:
        def __init__(self):  # noqa: D401
            pass

        def add_texts(self, texts, metadatas):  # noqa: ANN001
            called["texts"] = texts
            called["metadatas"] = metadatas

        @classmethod
        def from_documents(cls, documents, embedding, collection_name):  # noqa: ANN001
            called["collection_name"] = collection_name
            called["embedding"] = embedding
            return cls()

    monkeypatch.setattr(chat_service, "PGVector", DummyVS)

    req = SimpleNamespace(
        scope="single",
        pdf_file_id="fid1",
        embedding_model="emb-x",
        chunk_method="fixed",
        chunk_size=1000,
        chunk_overlap=0,
        similarity_threshold=None,
    )
    result = chat_service.build_vectorstore(req)

    assert result["status"] == "ok"
    assert result["collection_name"] == "col-single-fid1"
    assert result["num_chunks"] == len(called["texts"])
    assert called["texts"]  # 1 つ以上のチャンクがあること
    assert all(md["pdf_file_id"] == "fid1" for md in called["metadatas"])


def test_query_rag_success(monkeypatch) -> None:
    """query_rag がコンテキストから回答を生成し、ログ永続化ヘルパを呼ぶことを確認する。"""

    called: dict[str, object] = {}

    class DummyLLM:
        def invoke(self, prompt):  # noqa: ANN001
            called["prompt"] = prompt
            return "最終回答です。"

    def fake_init_llm(model_name: str, purpose: str = "/query"):  # noqa: ANN001
        return DummyLLM(), "dummy-llm"

    monkeypatch.setattr(chat_service, "_init_generation_llm", fake_init_llm)
    monkeypatch.setattr(chat_service, "_get_embeddings", lambda name: "EMB")
    monkeypatch.setattr(
        chat_service,
        "_build_collection_name_for_pdf",
        lambda emb, scope, fid=None: "col-x",  # noqa: ANN001
    )
    monkeypatch.setattr(chat_service, "_extract_answer_text", lambda resp: resp)

    class DummyRetriever:
        def get_relevant_documents(self, query):  # noqa: ANN001
            class Doc:
                def __init__(self, text: str) -> None:
                    self.page_content = text

            return [Doc("コンテキスト1"), Doc("コンテキスト2")]

    class DummyVS:
        def __init__(self, **kwargs):  # noqa: ANN001
            called["vs_kwargs"] = kwargs

        def as_retriever(self) -> DummyRetriever:
            return DummyRetriever()

    monkeypatch.setattr(chat_service, "PGVector", DummyVS)
    monkeypatch.setattr(
        chat_service,
        "_persist_chat_log_and_contexts",
        lambda scope, request, answer, contexts, resolved_llm, **kwargs: called.setdefault(  # noqa: ANN001
            "persist",
            {
                "scope": scope,
                "answer": answer,
                "contexts": contexts,
                "context_source_pdfs": kwargs.get("context_source_pdfs"),
            },
        ),
    )

    req = SimpleNamespace(
        query="テストの質問",
        llm_model="llm-x",
        embedding_model="emb-x",
        scope="single",
        pdf_file_id="fid1",
    )

    result = chat_service.query_rag(req)

    assert result["answer"]
    assert result["contexts"]
    assert result["source_documents"]
    assert result["llm_model_used"] == "dummy-llm"
    assert "Answer the question based only on the following context" in called.get("prompt", "")
    assert called["persist"]["scope"] == "single"
    assert called["persist"]["contexts"] == result["contexts"]


def test_query_rag_success_chat_jp_prompt(monkeypatch) -> None:
    """rag_prompt_style=chat_jp 指定時に日本語プロンプトが使われることを確認する。"""

    called: dict[str, object] = {}

    class DummyLLM:
        def invoke(self, prompt):  # noqa: ANN001
            called["prompt"] = prompt
            return "最終回答です。"

    def fake_init_llm(model_name: str, purpose: str = "/query"):  # noqa: ANN001
        return DummyLLM(), "dummy-llm"

    monkeypatch.setattr(chat_service, "_init_generation_llm", fake_init_llm)
    monkeypatch.setattr(chat_service, "_get_embeddings", lambda name: "EMB")
    monkeypatch.setattr(
        chat_service,
        "_build_collection_name_for_pdf",
        lambda emb, scope, fid=None: "col-x",  # noqa: ANN001
    )
    monkeypatch.setattr(chat_service, "_extract_answer_text", lambda resp: resp)

    class DummyRetriever:
        def get_relevant_documents(self, query):  # noqa: ANN001
            class Doc:
                def __init__(self, text: str) -> None:
                    self.page_content = text

            return [Doc("コンテキスト1"), Doc("コンテキスト2")]

    class DummyVS:
        def __init__(self, **kwargs):  # noqa: ANN001
            pass

        def as_retriever(self) -> DummyRetriever:
            return DummyRetriever()

    monkeypatch.setattr(chat_service, "PGVector", DummyVS)
    monkeypatch.setattr(
        chat_service,
        "_persist_chat_log_and_contexts",
        lambda *a, **k: None,  # noqa: ANN001
    )

    req = SimpleNamespace(
        query="テストの質問",
        llm_model="llm-x",
        embedding_model="emb-x",
        scope="single",
        pdf_file_id="fid1",
        rag_prompt_style="chat_jp",
    )

    result = chat_service.query_rag(req)

    assert result["answer"]
    assert "あなたは日本語のRAGシステムにおける回答エンジンです" in str(called.get("prompt", ""))


def test_query_rag_no_contexts_returns_general_chat(monkeypatch) -> None:
    """関連コンテキストが無い場合でも一般チャット応答で返すことを確認する。"""

    class DummyLLM:
        def invoke(self, prompt):  # noqa: ANN001
            return "回答"

    monkeypatch.setattr(
        chat_service,
        "_init_generation_llm",
        lambda *a, **k: (DummyLLM(), "dummy"),  # noqa: ANN001
    )
    monkeypatch.setattr(chat_service, "_get_embeddings", lambda name: "EMB")
    monkeypatch.setattr(
        chat_service,
        "_build_collection_name_for_pdf",
        lambda emb, scope, fid=None: "col-x",  # noqa: ANN001
    )

    class DummyRetriever:
        def get_relevant_documents(self, query):  # noqa: ANN001
            return []

    class DummyVS:
        def __init__(self, **kwargs):  # noqa: ANN001
            pass

        def as_retriever(self) -> DummyRetriever:
            return DummyRetriever()

    monkeypatch.setattr(chat_service, "PGVector", DummyVS)

    req = SimpleNamespace(
        query="Q",
        llm_model="llm-x",
        embedding_model="emb-x",
        scope="single",
        pdf_file_id="fid1",
    )

    result = chat_service.query_rag(req)
    assert result["contexts"] == []
    assert "関連するコンテキストが見つからなかった" in (result.get("context_notice") or "")
    assert isinstance(result["answer"], str) and result["answer"]
    assert result.get("context_source_pdfs") is None


def test_query_rag_fallback_when_answer_empty(monkeypatch) -> None:
    """最初の回答が空の場合にフォールバックプロンプトで再生成されることを確認する。"""

    class DummyLLM:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def invoke(self, prompt):  # noqa: ANN001
            self.calls.append(prompt)
            # 1 回目は空文字列相当、2 回目はフォールバック回答を返す
            if len(self.calls) == 1:
                return "   "  # 空白のみ
            return "フォールバック回答です。"

    llm = DummyLLM()

    monkeypatch.setattr(
        chat_service,
        "_init_generation_llm",
        lambda model_name, purpose="/query": (llm, "dummy-llm"),  # noqa: ANN001
    )
    monkeypatch.setattr(chat_service, "_get_embeddings", lambda name: "EMB")
    monkeypatch.setattr(
        chat_service,
        "_build_collection_name_for_pdf",
        lambda emb, scope, fid=None: "col-x",  # noqa: ANN001
    )
    monkeypatch.setattr(chat_service, "_extract_answer_text", lambda resp: resp)

    class DummyRetriever:
        def get_relevant_documents(self, query):  # noqa: ANN001
            class Doc:
                def __init__(self, text: str) -> None:
                    self.page_content = text

            return [Doc("コンテキスト1"), Doc("コンテキスト2")]

    class DummyVS:
        def __init__(self, **kwargs):  # noqa: ANN001
            pass

        def as_retriever(self) -> DummyRetriever:
            return DummyRetriever()

    monkeypatch.setattr(chat_service, "PGVector", DummyVS)
    monkeypatch.setattr(
        chat_service,
        "_persist_chat_log_and_contexts",
        lambda *a, **k: None,  # noqa: ANN001
    )

    req = SimpleNamespace(
        query="テストの質問",
        llm_model="llm-x",
        embedding_model="emb-x",
        scope="single",
        pdf_file_id="fid1",
    )

    result = chat_service.query_rag(req)

    # フォールバックが実行され、最終回答がフォールバック由来であること
    assert result["answer"] == "フォールバック回答です。"
    assert len(llm.calls) == 2


def test_query_rag_fallback_when_no_match_message(monkeypatch) -> None:
    """「本文に該当記述がありません。」のみの回答からフォールバックすることを確認する。"""

    class DummyLLM:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def invoke(self, prompt):  # noqa: ANN001
            self.calls.append(prompt)
            if len(self.calls) == 1:
                return "本文に該当記述がありません。"
            return "フォールバック2回目の回答です。"

    llm = DummyLLM()

    monkeypatch.setattr(
        chat_service,
        "_init_generation_llm",
        lambda model_name, purpose="/query": (llm, "dummy-llm"),  # noqa: ANN001
    )
    monkeypatch.setattr(chat_service, "_get_embeddings", lambda name: "EMB")
    monkeypatch.setattr(
        chat_service,
        "_build_collection_name_for_pdf",
        lambda emb, scope, fid=None: "col-x",  # noqa: ANN001
    )
    monkeypatch.setattr(chat_service, "_extract_answer_text", lambda resp: resp)

    class DummyRetriever:
        def get_relevant_documents(self, query):  # noqa: ANN001
            class Doc:
                def __init__(self, text: str) -> None:
                    self.page_content = text

            return [Doc("コンテキストA"), Doc("コンテキストB")]

    class DummyVS:
        def __init__(self, **kwargs):  # noqa: ANN001
            pass

        def as_retriever(self) -> DummyRetriever:
            return DummyRetriever()

    monkeypatch.setattr(chat_service, "PGVector", DummyVS)
    monkeypatch.setattr(
        chat_service,
        "_persist_chat_log_and_contexts",
        lambda *a, **k: None,  # noqa: ANN001
    )

    req = SimpleNamespace(
        query="別の質問",
        llm_model="llm-y",
        embedding_model="emb-y",
        scope="single",
        pdf_file_id="fid2",
    )

    result = chat_service.query_rag(req)

    assert result["answer"] == "フォールバック2回目の回答です。"
    assert len(llm.calls) == 2
