from __future__ import annotations

import sys
from pathlib import Path

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.upload_service as upload_service  # type: ignore  # noqa: E402


def test_run_pdf_upload_pipeline_sync_pypdf_path(tmp_path, monkeypatch) -> None:
    """PyPDF パスで OCR を行い、JSON/PDF と DB 永続化が呼ばれることを確認する。"""

    # 一時ディレクトリに保存先を差し替え
    extracted_dir = tmp_path / "extracted"
    pdf_dir = tmp_path / "pdf"
    images_dir = tmp_path / "images"
    extracted_dir.mkdir()
    pdf_dir.mkdir()
    images_dir.mkdir()
    monkeypatch.setattr(upload_service, "EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr(upload_service, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(upload_service, "IMAGES_DIR", images_dir)

    # OCR とクレンジングを簡易な実装に差し替え
    monkeypatch.setattr(upload_service, "extract_pdf_text_layout", lambda contents: "Hello PDF text")
    monkeypatch.setattr(upload_service, "cleanse_pdf_text", lambda text: text + "_CLEANSED")

    captured: dict[str, object] = {}

    def fake_persist_pdf_upload_to_db(**kwargs):  # noqa: ANN001
        captured["persist_kwargs"] = kwargs

    monkeypatch.setattr(upload_service, "persist_pdf_upload_to_db", fake_persist_pdf_upload_to_db)

    contents = b"%PDF-1.4 test"
    result = upload_service.run_pdf_upload_pipeline_sync(
        contents=contents,
        file_name="sample.pdf",
        cleanse=True,
        question_llm_model="q-llm",
        answer_llm_model="a-llm",
        generate_image_captions=False,
        ocr_engine="pypdf",
        ocr_quality="balanced",
        ocr_image_compression="balanced",
        job_id=None,
    )

    assert "file_id" in result
    file_id = result["file_id"]
    assert isinstance(file_id, str)

    # JSON が保存されていること
    import json as _json

    json_path = extracted_dir / f"{file_id}.json"
    assert json_path.exists()
    data = _json.loads(json_path.read_text(encoding="utf-8"))
    assert data["file_name"] == "sample.pdf"
    assert data["settings"]["ocr_engine_used"] == "pypdf"
    assert data["settings"]["ocr_engine_selected"] == "pypdf"
    assert data["settings"]["cleanse_used"] is True

    # PDF 本体が保存されていること
    pdf_path = pdf_dir / f"{file_id}.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes() == contents

    # DB 永続化ヘルパが呼ばれていること
    persist_kwargs = captured.get("persist_kwargs")
    assert persist_kwargs is not None
    assert persist_kwargs["file_id"] == file_id
    assert persist_kwargs["file_name"] == "sample.pdf"
    assert persist_kwargs["file_size"] == len(contents)
    assert persist_kwargs["ocr_engine_used"] == "pypdf"


def test_run_pdf_upload_pipeline_sync_ollama_fallback_to_pypdf(tmp_path, monkeypatch) -> None:
    """Ollama DeepSeek OCR 失敗時に PyPDF へフォールバックすることを確認する。"""

    extracted_dir = tmp_path / "extracted"
    pdf_dir = tmp_path / "pdf"
    images_dir = tmp_path / "images"
    extracted_dir.mkdir()
    pdf_dir.mkdir()
    images_dir.mkdir()
    monkeypatch.setattr(upload_service, "EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr(upload_service, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(upload_service, "IMAGES_DIR", images_dir)

    def fake_ocr_via_ollama(contents, **kwargs):  # noqa: ANN001
        raise RuntimeError("ocr failed")

    monkeypatch.setattr(upload_service, "run_deepseek_ocr_via_ollama_for_pdf", fake_ocr_via_ollama)
    monkeypatch.setattr(upload_service, "extract_pdf_text_layout", lambda contents: "TEXT_FROM_PDF")
    monkeypatch.setattr(upload_service, "persist_pdf_upload_to_db", lambda **kwargs: None)  # noqa: ANN002

    result = upload_service.run_pdf_upload_pipeline_sync(
        contents=b"PDF",
        file_name="x.pdf",
        cleanse=False,
        question_llm_model="q",
        answer_llm_model="a",
        generate_image_captions=False,
        ocr_engine="ollama_deepseek",
        ocr_quality="balanced",
        ocr_image_compression="balanced",
        job_id=None,
    )

    assert result["ocr_engine_selected"] == "ollama_deepseek"
    assert result["ocr_engine_used"] == "pypdf"
    assert "Ollama DeepSeek OCR によるテキスト抽出に失敗" in (result["warning"] or "")


def test_generate_qa_for_existing_pdf_happy_path_with_caption_error(tmp_path, monkeypatch) -> None:
    """generate_qa_for_existing_pdf が LLM / RAGAS を用いて QA を生成し、画像キャプションエラーを無視することを確認する。"""

    extracted_dir = tmp_path / "extracted"
    pdf_dir = tmp_path / "pdf"
    extracted_dir.mkdir()
    pdf_dir.mkdir()
    monkeypatch.setattr(upload_service, "EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr(upload_service, "PDF_DIR", pdf_dir)

    file_id = "fid-qa"
    text = "これはテスト本文です。\n2行目です。"
    data = {
        "text": text,
        "questions": [],
        "answers": [],
        "qa_meta": [],
        "file_name": "doc.pdf",
        "settings": {
            "cleanse_used": False,
            "generate_image_captions": True,
            "ocr_engine_selected": "pypdf",
            "ocr_engine_used": "pypdf",
        },
    }
    import json as _json

    (extracted_dir / f"{file_id}.json").write_text(
        _json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    (pdf_dir / f"{file_id}.pdf").write_bytes(b"%PDF-1.4 dummy")

    # main.init_generation_llm をダミー実装に差し替え
    from app import main as main_module  # type: ignore  # noqa: E402

    class DummyLLM:
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix

        def invoke(self, prompt, max_tokens: int | None = None):  # noqa: ANN001
            if self.prefix == "Q":
                return "質問1\n質問2"
            return "回答1"

    def fake_init_generation_llm(model_name: str, purpose: str = "question_generation", **kwargs):  # noqa: ANN001
        if "question" in purpose:
            return DummyLLM("Q"), "dummy-q-llm"
        return DummyLLM("A"), "dummy-a-llm"

    monkeypatch.setattr(main_module, "init_generation_llm", fake_init_generation_llm)
    monkeypatch.setattr(upload_service, "_extract_answer_text", lambda resp: resp)

    # RAGAS 関連ヘルパを簡略化
    monkeypatch.setattr(
        upload_service,
        "extract_relevant_context",
        lambda q, s, max_sentences=6: "ctx",  # noqa: ANN001
    )
    monkeypatch.setattr(
        upload_service,
        "evaluate_answer_quality",
        lambda ans, ctx_lines: {"score": 1.0, "is_dummy": False, "needs_retry": False},  # noqa: ANN001
    )
    monkeypatch.setattr(
        upload_service,
        "generate_default_chunks_for_storage",
        lambda combined_text: [{"content": combined_text}],  # noqa: ANN001
    )

    # 画像キャプション生成は例外を投げるようにしても処理継続することを確認
    def fake_generate_image_captions_from_pdf(contents):  # noqa: ANN001
        raise RuntimeError("caption error")

    monkeypatch.setattr(
        upload_service,
        "generate_image_captions_from_pdf",
        fake_generate_image_captions_from_pdf,
    )

    captured: dict[str, object] = {}

    def fake_persist_pdf_upload_to_db(**kwargs):  # noqa: ANN001
        captured["persist"] = kwargs

    monkeypatch.setattr(upload_service, "persist_pdf_upload_to_db", fake_persist_pdf_upload_to_db)

    result = upload_service.generate_qa_for_existing_pdf(
        file_id=file_id,
        question_llm_model="q-model",
        answer_llm_model="a-model",
    )

    assert result["file_id"] == file_id
    assert result["questions"]
    assert result["answers"]
    assert len(result["questions"]) == len(result["answers"])

    # JSON が更新されていること
    data2 = _json.loads((extracted_dir / f"{file_id}.json").read_text(encoding="utf-8"))
    assert data2["questions"]
    assert data2["qa_meta"]
    assert data2["settings"]["ocr_engine_used"] == "pypdf"

    persist_kwargs = captured.get("persist")
    assert persist_kwargs is not None
    assert persist_kwargs["file_id"] == file_id
    assert persist_kwargs["file_name"] == "doc.pdf"
