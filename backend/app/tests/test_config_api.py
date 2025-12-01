from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # type: ignore  # noqa: E402
import app.config_api as config_api  # type: ignore  # noqa: E402


client = TestClient(app)


def test_get_inference_mode_api(monkeypatch) -> None:
    """/config/inference_mode (GET) が現在のモードを返すことを確認する。"""

    monkeypatch.setattr(config_api, "get_inference_mode", lambda: "local")

    resp = client.get("/config/inference_mode")

    assert resp.status_code == 200
    assert resp.json() == {"mode": "local"}


def test_update_inference_mode_api_success(monkeypatch) -> None:
    """/config/inference_mode (POST) が set_inference_mode を呼び、更新後の値を返すことを確認する。"""

    called: dict[str, object] = {}

    def fake_set_mode(mode: str) -> None:
        called["mode"] = mode

    monkeypatch.setattr(config_api, "set_inference_mode", fake_set_mode)
    monkeypatch.setattr(config_api, "get_inference_mode", lambda: "remote")

    resp = client.post("/config/inference_mode", json={"mode": "remote"})

    assert resp.status_code == 200
    assert called["mode"] == "remote"
    assert resp.json() == {"mode": "remote"}


def test_update_inference_mode_api_bad_request(monkeypatch) -> None:
    """set_inference_mode が ValueError を投げた場合、400 が返ることを確認する。"""

    def fake_set_mode(mode: str) -> None:
        raise ValueError("invalid mode")

    monkeypatch.setattr(config_api, "set_inference_mode", fake_set_mode)

    resp = client.post("/config/inference_mode", json={"mode": "bad"})

    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"] == "invalid mode"


def test_get_model_selection_api(monkeypatch) -> None:
    """/config/model_selection (GET) が DB ヘルパの値を返すことを確認する。"""

    monkeypatch.setattr(
        config_api,
        "_db_get_model_selection",
        lambda: {"llm_model": "llm-x", "embedding_model": "emb-y"},
    )

    resp = client.get("/config/model_selection")

    assert resp.status_code == 200
    assert resp.json() == {"llm_model": "llm-x", "embedding_model": "emb-y"}


def test_update_model_selection_api(monkeypatch) -> None:
    """/config/model_selection (POST) が _db_update_model_selection を呼び出すことを確認する。"""

    captured: dict[str, object] = {}

    def fake_update(selection):  # noqa: ANN001
        captured["selection"] = selection

    monkeypatch.setattr(config_api, "_db_update_model_selection", fake_update)

    payload = {"llm_model": " llm-x ", "embedding_model": " emb-y "}

    resp = client.post("/config/model_selection", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"llm_model": "llm-x", "embedding_model": "emb-y"}

    sel = captured.get("selection")
    assert sel is not None
    assert sel.llm_model == "llm-x"
    assert sel.embedding_model == "emb-y"


def test_get_inference_health_api_ok(monkeypatch) -> None:
    """/config/inference_health が Ollama API 正常時の情報を返すことを確認する。"""

    monkeypatch.setattr(config_api, "get_inference_mode", lambda: "local")
    monkeypatch.setattr(config_api, "get_ollama_base_url", lambda: "http://ollama:11434")

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.ok = True

    def fake_get(url, timeout=15):  # noqa: ANN001
        return DummyResponse()

    monkeypatch.setattr(config_api.requests, "get", fake_get)

    resp = client.get("/config/inference_health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "local"
    assert body["base_url"] == "http://ollama:11434"
    assert body["ok"] is True
    assert body["status_code"] == 200
    assert body["error"] is None


def test_get_inference_health_api_error(monkeypatch) -> None:
    """Ollama API 側で例外が発生した場合、ok=False かつ error メッセージが設定されることを確認する。"""

    monkeypatch.setattr(config_api, "get_inference_mode", lambda: "local")
    monkeypatch.setattr(config_api, "get_ollama_base_url", lambda: "http://ollama:11434")

    def fake_get(url, timeout=15):  # noqa: ANN001
        raise RuntimeError("connection failed")

    monkeypatch.setattr(config_api.requests, "get", fake_get)

    resp = client.get("/config/inference_health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "connection failed" in (body["error"] or "")
