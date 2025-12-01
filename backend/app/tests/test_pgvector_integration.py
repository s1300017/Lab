from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.chat_service as chat_service  # type: ignore  # noqa: E402
from langchain_community.vectorstores.pgvector import PGVector  # noqa: E402


@pytest.mark.integration
def test_pgvector_with_real_db_and_pdf_chunks(monkeypatch) -> None:
    """実 DB (pgvector) を用いた簡易結合テスト。

    環境変数 ENABLE_DB_INTEGRATION_TESTS=1 のときのみ実行し、
    DB 未起動などの場合は安全に skip する。
    """

    if os.getenv("ENABLE_DB_INTEGRATION_TESTS", "0").lower() not in {"1", "true", "yes"}:
        pytest.skip("ENABLE_DB_INTEGRATION_TESTS が有効なときのみ実行します")

    engine = chat_service.engine

    # DB 接続確認
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"DB 接続に失敗したため pgvector 結合テストをスキップします: {e}")

    # テスト用 pdf_file_id を決めて、既存データをクリーンアップ
    fid = "it-pgvector-pdf"
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM pdf_chunks WHERE pdf_file_id = :fid"), {"fid": fid})

        # シンプルなチャンクを 2 件登録
        chunks = ["chunk one", "chunk two"]
        for idx, content in enumerate(chunks):
            conn.execute(
                text(
                    """
                    INSERT INTO pdf_chunks (pdf_file_id, chunk_index, content, content_hash)
                    VALUES (:fid, :idx, :content, :hash)
                    """
                ),
                {
                    "fid": fid,
                    "idx": idx,
                    "content": content,
                    "hash": "dummy-hash",
                },
            )

    # 埋め込みモデルはダミー実装に差し替え
    class DummyEmbeddings:
        def embed_documents(self, texts):  # noqa: ANN001
            return [[0.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):  # noqa: ANN001
            return [0.0, 0.0, 0.0]

    monkeypatch.setattr(chat_service, "_get_embeddings", lambda name: DummyEmbeddings())

    # build_vectorstore を実 DB + PGVector で実行
    from types import SimpleNamespace

    req = SimpleNamespace(
        scope="single",
        pdf_file_id=fid,
        embedding_model="dummy-model",
        chunk_method="recursive",
        chunk_size=100,
        chunk_overlap=0,
        similarity_threshold=None,
    )

    result = chat_service.build_vectorstore(req)

    assert result["status"] == "ok"
    assert result["num_chunks"] > 0

    # 実際に PGVector の collection に対してクエリできることを確認
    collection_name = result["collection_name"]
    vs = PGVector(
        embedding_function=DummyEmbeddings(),
        collection_name=collection_name,
        connection_string=chat_service.DB_URL,
        use_jsonb=True,
    )
    docs = vs.similarity_search("chunk", k=1)
    assert docs
