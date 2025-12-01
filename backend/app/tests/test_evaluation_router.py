from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # type: ignore  # noqa: E402
import app.evaluation_service as evaluation_service  # type: ignore  # noqa: E402


client = TestClient(app)


def test_bulk_evaluate_router_uses_service(monkeypatch) -> None:
    """/bulk_evaluate/ が evaluation_service.bulk_evaluate を呼び出すことを確認する。"""

    called: dict[str, object] = {}

    async def _fake_bulk(data):  # noqa: ANN001
        called["data"] = data
        return {"ok": True, "echo": data}

    monkeypatch.setattr(evaluation_service, "bulk_evaluate", _fake_bulk)

    payload = {"text": "hello", "embedding_model": "huggingface_bge_small", "questions": ["Q"], "answers": ["A"]}

    resp = client.post("/bulk_evaluate/", json=payload)

    assert resp.status_code == 200
    assert called["data"] == payload
    assert resp.json() == {"ok": True, "echo": payload}
