from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import HTTPException

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.upload_service as upload_service  # type: ignore  # noqa: E402


def test_start_upload_job_starts_background_worker(monkeypatch) -> None:
    """start_upload_job がジョブを登録し、バックグラウンドスレッドを開始することを確認する。"""

    captured: dict[str, object] = {}

    def fake_set_upload_job(job):  # noqa: ANN001
        captured["job"] = job

    def fake_run_pipeline(**kwargs):  # noqa: ANN001
        # すぐに完了する軽量処理に置き換え
        return {"file_id": "file-123"}

    class DummyThread:
        def __init__(self, target, daemon: bool = False):  # noqa: ANN001
            captured["thread_target"] = target
            captured["thread_daemon"] = daemon

        def start(self) -> None:
            captured["thread_started"] = True
            # 本番同様に worker を即座に実行しておく
            target = captured.get("thread_target")
            if callable(target):
                target()

    # 実DB・実OCRを避けるために内部依存をモック
    monkeypatch.setattr(upload_service, "set_upload_job", fake_set_upload_job)
    monkeypatch.setattr(upload_service, "run_pdf_upload_pipeline_sync", fake_run_pipeline)
    monkeypatch.setattr(upload_service.threading, "Thread", DummyThread)

    result = upload_service.start_upload_job(
        contents=b"PDFDATA",
        file_name="test.pdf",
        cleanse=False,
        question_llm_model="mistral",
        answer_llm_model="mistral",
        generate_image_captions=False,
        ocr_engine="pypdf",
    )

    assert "job_id" in result
    assert isinstance(result["job_id"], str)

    job = captured.get("job")
    assert job is not None
    assert isinstance(job, upload_service.UploadJobState)
    assert job.job_id == result["job_id"]
    assert captured.get("thread_started") is True
    assert captured.get("thread_daemon") is True


def test_get_upload_job_status_from_memory(monkeypatch) -> None:
    """メモリ上の UploadJobState から状態が返ることを確認する。"""

    # DB 書き込みを抑止
    monkeypatch.setattr(upload_service, "_db_create_upload_job", lambda *a, **k: None)

    job = upload_service.UploadJobState(
        job_id="job-1",
        status=upload_service.UploadJobStatus.RUNNING,
    )
    upload_service.set_upload_job(job)

    status = upload_service.get_upload_job_status("job-1")

    assert status["job_id"] == "job-1"
    assert status["status"] == upload_service.UploadJobStatus.RUNNING.value
    assert "progress" in status


def test_get_upload_job_status_db_not_found_raises_404(monkeypatch) -> None:
    """メモリにもDBにも存在しないジョブIDでは 404 が返ることを確認する。"""

    # メモリ側にジョブが無いように見せる
    monkeypatch.setattr(upload_service, "get_upload_job", lambda job_id: None)

    class _DummyMappings:
        def first(self):  # noqa: D401, ANN001
            # 行が存在しないケース
            return None

    class _DummyResult:
        def mappings(self) -> "_DummyMappings":
            return _DummyMappings()

    class _DummyConn:
        def __enter__(self) -> "_DummyConn":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401, ANN001
            return None

        def execute(self, stmt, params=None):  # noqa: ANN001
            return _DummyResult()

    class _DummyEngine:
        def connect(self) -> _DummyConn:  # type: ignore[override]
            return _DummyConn()

    monkeypatch.setattr(upload_service, "engine", _DummyEngine())

    try:
        upload_service.get_upload_job_status("missing-job")
        assert False, "HTTPException が送出されるはず"
    except HTTPException as e:
        assert e.status_code == 404


def test_cancel_upload_job_marks_cancel_requested(monkeypatch) -> None:
    """cancel_upload_job が cancel_requested フラグを立てることを確認する。"""

    # DB 更新を抑止
    monkeypatch.setattr(upload_service, "_db_create_upload_job", lambda *a, **k: None)
    monkeypatch.setattr(upload_service, "_db_update_upload_job", lambda *a, **k: None)

    job = upload_service.UploadJobState(job_id="job-2")
    upload_service.set_upload_job(job)

    resp = upload_service.cancel_upload_job("job-2")

    assert resp["job_id"] == "job-2"
    assert resp["cancel_requested"] is True

    # メモリ上の状態も確認
    job2 = upload_service.get_upload_job("job-2")
    assert job2 is not None
    assert job2.cancel_requested is True


def test_get_extracted_data_reads_json_and_pdf(tmp_path, monkeypatch) -> None:
    """get_extracted_data が JSON と PDF を読み込み、base64 付きで返すことを確認する。"""

    extracted_dir = tmp_path / "extracted"
    pdf_dir = tmp_path / "pdf"
    extracted_dir.mkdir()
    pdf_dir.mkdir()

    monkeypatch.setattr(upload_service, "EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr(upload_service, "PDF_DIR", pdf_dir)

    file_id = "fid-1"
    data = {"text": "hello", "questions": [], "answers": [], "qa_meta": [], "file_name": "x.pdf"}
    (extracted_dir / f"{file_id}.json").write_text(
        __import__("json").dumps(data, ensure_ascii=False), encoding="utf-8"
    )

    pdf_bytes = b"%PDF-1.4..."
    (pdf_dir / f"{file_id}.pdf").write_bytes(pdf_bytes)

    result = upload_service.get_extracted_data(file_id)

    assert result["text"] == "hello"
    assert result["file_name"] == "x.pdf"
    assert "pdf_bytes_base64" in result

    decoded = __import__("base64").b64decode(result["pdf_bytes_base64"])  # type: ignore[arg-type]
    assert decoded == pdf_bytes
