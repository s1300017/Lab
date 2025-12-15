from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # type: ignore  # noqa: E402
import app.history_api as history_api  # type: ignore  # noqa: E402


client = TestClient(app)


class _DummyResult:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self._rows = rows

    def fetchall(self) -> list[SimpleNamespace]:
        return self._rows


class _DummyConn:
    def __enter__(self) -> "_DummyConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        # 何もしないコンテキストマネージャ
        return None

    def execute(self, stmt):  # noqa: ANN001
        # history_pdf_files では dict(r._mapping) を期待している
        row = SimpleNamespace(
            _mapping={
                "id": 1,
                "original_name": "test.pdf",
                "file_name": "test.pdf",
                "file_size": 123,
                "uploaded_at": "2024-01-01 00:00:00",
            }
        )
        return _DummyResult([row])


class _DummyEngine:
    def begin(self) -> _DummyConn:
        return _DummyConn()


def test_history_pdf_files_smoke(monkeypatch) -> None:
    """/history/pdf-files が 200 を返し、items 配列を持つことを確認する。"""

    # DB アクセスをダミーエンジンに置き換え
    monkeypatch.setattr(history_api, "engine", _DummyEngine())

    resp = client.get("/history/pdf-files")

    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert isinstance(data["items"], list)
    assert len(data["items"]) == 1
    assert data["items"][0]["original_name"] == "test.pdf"


class _DummyEngineForImport(_DummyEngine):
    def begin(self) -> _DummyConn:  # type: ignore[override]
        # import_experiment 内の SELECT id FROM experiments ... を呼ぶが、
        # 結果は experiment_id 取得にのみ使われるため、固定値を返す
        class _Conn(_DummyConn):
            def execute(self, stmt):  # noqa: ANN001
                row = (123,)

                class _Result:
                    def fetchone(self_inner):  # noqa: ANN001
                        return row

                return _Result()

        return _Conn()


def test_import_experiment_smoke(monkeypatch) -> None:
    """/history/import-experiment が evaluation 用に正常応答することを確認する。"""

    # DB アクセスと実際の永続化をモック
    monkeypatch.setattr(history_api, "engine", _DummyEngineForImport())

    def _fake_persist(pdf_file_id, request_params, results):  # noqa: ANN001
        return None

    monkeypatch.setattr(history_api, "persist_experiment_results", _fake_persist)

    payload = {
        "pdf_file_id": 1,
        "experiment_name": "test-exp",
        "parameters": {},
        "results": [],
    }

    resp = client.post("/history/import-experiment", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "experiment_id" in data


class _DummyEngineForExperimentResultsQuery(_DummyEngine):
    def begin(self) -> _DummyConn:  # type: ignore[override]
        class _Conn(_DummyConn):
            def execute(self, stmt, params=None):  # noqa: ANN001
                eids = []
                if isinstance(params, dict):
                    eids = params.get("eids") or []
                rows = []
                for eid in eids:
                    rows.append(
                        SimpleNamespace(
                            _mapping={
                                "id": int(eid) * 10,
                                "experiment_id": int(eid),
                                "embedding_model": "dummy-emb",
                                "llm_model": None,
                                "chunk_strategy": "fixed-100-20",
                                "chunk_size": 100,
                                "chunk_overlap": 20,
                                "num_chunks": 5,
                                "avg_chunk_len": 120,
                                "overall_score": 0.5,
                                "faithfulness": 0.6,
                                "answer_relevancy": 0.7,
                                "context_recall": 0.8,
                                "context_precision": 0.9,
                                "answer_correctness": 0.4,
                                "answer_similarity": 0.3,
                                "evaluation_llm_model": None,
                                "duration_seconds": 1.23,
                                "details": None,
                                "created_at": "2024-01-01 00:00:00",
                                "pdf_file_id": 999,
                                "experiment_name": f"exp-{eid}",
                                "experiment_created_at": "2024-01-01 00:00:00",
                                "parameters": '{"llm_models": ["llm-from-params"], "evaluation_llm_model": "eval-from-params", "force_llm_generation": true}',
                            }
                        )
                    )
                return _DummyResult(rows)

        return _Conn()


def test_history_experiment_results_query_requires_ids(monkeypatch) -> None:
    monkeypatch.setattr(history_api, "engine", _DummyEngineForExperimentResultsQuery())

    resp = client.post("/history/experiment-results/query", json={})

    assert resp.status_code == 400


def test_history_experiment_results_query_smoke(monkeypatch) -> None:
    monkeypatch.setattr(history_api, "engine", _DummyEngineForExperimentResultsQuery())

    resp = client.post("/history/experiment-results/query", json={"experiment_ids": [2, 1, 1]})

    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert isinstance(data["items"], list)
    assert len(data["items"]) == 2
    assert [item["experiment_id"] for item in data["items"]] == [1, 2]
    for item in data["items"]:
        assert item["llm_model"] == "llm-from-params"
        assert item["evaluation_llm_model"] == "eval-from-params"
        assert item["force_llm_generation"] is True
        assert "parameters" not in item
