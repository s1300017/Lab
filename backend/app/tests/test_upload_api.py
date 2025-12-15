from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# app パッケージを import 可能にするために backend ディレクトリを sys.path に追加
# __file__ は backend/app/tests/test_upload_api.py を指すので、parents[2] が backend 直下になる
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # type: ignore  # noqa: E402
import app.upload_api as upload_api  # type: ignore  # noqa: E402


client = TestClient(app)


def test_upload_job_start_uses_service(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_start_upload_job(**kwargs):
        called["kwargs"] = kwargs
        return {"job_id": "job-123"}

    monkeypatch.setattr(upload_api, "start_upload_job", fake_start_upload_job)

    resp = client.post(
        "/upload_job/start",
        files={"file": ("test.pdf", b"PDFDATA", "application/pdf")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job-123"
    assert called["kwargs"]["file_name"] == "test.pdf"
    assert called["kwargs"]["contents"] == b"PDFDATA"


def test_upload_job_status_uses_service(monkeypatch) -> None:
    def fake_get_status(job_id: str) -> dict:
        return {"job_id": job_id, "status": "completed"}

    monkeypatch.setattr(upload_api, "get_upload_job_status", fake_get_status)

    resp = client.get("/upload_job/status/job-xyz")

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-xyz", "status": "completed"}


def test_upload_job_cancel_uses_service(monkeypatch) -> None:
    def fake_cancel(job_id: str) -> dict:
        return {"job_id": job_id, "status": "cancelled"}

    monkeypatch.setattr(upload_api, "cancel_upload_job", fake_cancel)

    resp = client.post("/upload_job/cancel/job-xyz")

    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-xyz", "status": "cancelled"}


def test_uploadfile_uses_service(monkeypatch) -> None:
    def fake_upload_and_generate_qa(**kwargs) -> dict:
        return {
            "file_id": "file-1",
            "text": "hello",
            "questions": ["Q1"],
            "answers": ["A1"],
        }

    monkeypatch.setattr(upload_api, "upload_and_generate_qa", fake_upload_and_generate_qa)

    resp = client.post(
        "/uploadfile/",
        files={"file": ("test.pdf", b"PDFDATA", "application/pdf")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["file_id"] == "file-1"
    assert body["questions"] == ["Q1"]
    assert body["answers"] == ["A1"]


def test_get_extracted_uses_service(monkeypatch) -> None:
    def fake_get_extracted(file_id: str) -> dict:
        return {"file_id": file_id, "text": "dummy"}

    monkeypatch.setattr(upload_api, "get_extracted_data", fake_get_extracted)

    resp = client.get("/get_extracted/abc")

    assert resp.status_code == 200
    assert resp.json() == {"file_id": "abc", "text": "dummy"}


def test_generate_qa_for_pdf_uses_service(monkeypatch) -> None:
    def fake_generate_qa(
        file_id: str,
        question_llm_model: str,
        answer_llm_model: str,
        question_count: int = 5,
        min_qa_score: float = 0.0,
    ) -> dict:
        return {
            "file_id": file_id,
            "question_count": question_count,
            "min_qa_score": min_qa_score,
            "questions": ["Q"],
            "answers": ["A"],
        }

    monkeypatch.setattr(upload_api, "generate_qa_for_existing_pdf", fake_generate_qa)

    resp = client.post(
        "/pdf/abc/generate_qa",
        data={
            "question_llm_model": "mistral",
            "answer_llm_model": "mistral",
            "question_count": "7",
            "min_qa_score": "0.5",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "file_id": "abc",
        "question_count": 7,
        "min_qa_score": 0.5,
        "questions": ["Q"],
        "answers": ["A"],
    }


def test_regenerate_qa_for_pdf_uses_service(monkeypatch) -> None:
    def fake_regenerate_qa(
        file_id: str,
        question_llm_model: str,
        answer_llm_model: str,
        question_count: int = 5,
        min_qa_score: float = 0.0,
    ) -> dict:
        return {
            "file_id": file_id,
            "question_count": question_count,
            "min_qa_score": min_qa_score,
            "questions": ["Q"],
            "answers": ["A"],
        }

    monkeypatch.setattr(upload_api, "regenerate_qa_for_existing_pdf", fake_regenerate_qa)

    resp = client.post(
        "/pdf/abc/regenerate_qa",
        data={
            "question_llm_model": "mistral",
            "answer_llm_model": "mistral",
            "question_count": "9",
            "min_qa_score": "0.4",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "file_id": "abc",
        "question_count": 9,
        "min_qa_score": 0.4,
        "questions": ["Q"],
        "answers": ["A"],
    }
