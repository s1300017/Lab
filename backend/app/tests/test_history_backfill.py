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


def test_history_backfill_no_targets_returns_message(tmp_path, monkeypatch) -> None:
    """対象 JSON が存在しない場合、processed=0 のレスポンスが返ることを確認する。"""

    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    monkeypatch.setattr(history_api, "EXTRACTED_DIR", extracted_dir)

    resp = client.post("/history/backfill", params={"dry_run": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["processed"] == 0
    # 対象なしケースでは total_targets キーは返ってこない実装になっている
    assert "total_targets" not in body
    assert body["dry_run"] is True
    assert body["message"] == "対象データが見つかりません"


def test_history_backfill_dry_run_counts_targets(tmp_path, monkeypatch) -> None:
    """dry_run=True では DB 永続化を行わず、processed/total_targets が一致することを確認する。"""

    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    monkeypatch.setattr(history_api, "EXTRACTED_DIR", extracted_dir)

    # ダミー JSON を 2 件作成
    import json as _json

    for fid in ["f1", "f2"]:
        data = {
            "text": "dummy text",
            "questions": ["Q"],
            "answers": ["A"],
            "qa_meta": [],
            "file_name": f"{fid}.pdf",
        }
        (extracted_dir / f"{fid}.json").write_text(
            _json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    # DB への書き込みを抑止
    monkeypatch.setattr(history_api, "persist_pdf_upload_to_db", lambda *a, **k: None)

    resp = client.post("/history/backfill", params={"dry_run": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["processed"] == 2
    assert body["total_targets"] == 2
    assert body["dry_run"] is True
    assert len(body["details"]) == 2


def test_history_backfill_real_mode_calls_persist(tmp_path, monkeypatch) -> None:
    """dry_run=False では persist_pdf_upload_to_db が呼ばれることを確認する。"""

    extracted_dir = tmp_path / "extracted"
    pdf_dir = tmp_path / "pdf"
    extracted_dir.mkdir()
    pdf_dir.mkdir()
    monkeypatch.setattr(history_api, "EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr(history_api, "PDF_DIR", pdf_dir)

    import json as _json

    fid = "fid-real"
    data = {
        "text": "backfill text",
        "questions": ["Q1"],
        "answers": ["A1"],
        "qa_meta": [],
        "file_name": "x.pdf",
    }
    (extracted_dir / f"{fid}.json").write_text(
        _json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    (pdf_dir / f"{fid}.pdf").write_bytes(b"%PDF-1.4 dummy")

    captured: dict[str, object] = {"calls": []}

    def fake_persist_pdf_upload_to_db(**kwargs):  # noqa: ANN001
        captured["calls"].append(kwargs)

    monkeypatch.setattr(history_api, "persist_pdf_upload_to_db", fake_persist_pdf_upload_to_db)

    resp = client.post("/history/backfill", params={"dry_run": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["processed"] == 1
    assert len(captured["calls"]) == 1
    call = captured["calls"][0]
    assert call["file_id"] == fid
    assert call["file_name"] == "x.pdf"
