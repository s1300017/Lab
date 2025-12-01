from __future__ import annotations

import sys
from pathlib import Path

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.upload_service as upload_service  # type: ignore  # noqa: E402
from app import main as main_module  # type: ignore  # noqa: E402


def test_upload_and_generate_qa_end_to_end(tmp_path, monkeypatch) -> None:
    """run_pdf_upload_pipeline_sync + generate_qa_for_existing_pdf をまとめて呼ぶシナリオ結合テスト。"""

    # 保存先ディレクトリを一時ディレクトリに切り替え
    extracted_dir = tmp_path / "extracted"
    pdf_dir = tmp_path / "pdf"
    images_dir = tmp_path / "images"
    extracted_dir.mkdir()
    pdf_dir.mkdir()
    images_dir.mkdir()
    monkeypatch.setattr(upload_service, "EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr(upload_service, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(upload_service, "IMAGES_DIR", images_dir)

    # 抽出処理はシンプルなテキスト返却に差し替え
    monkeypatch.setattr(upload_service, "extract_pdf_text_layout", lambda contents: "本文テキスト")
    monkeypatch.setattr(upload_service, "cleanse_pdf_text", lambda t: t)

    captured: dict[str, object] = {"persist_calls": []}

    def fake_persist_pdf_upload_to_db(**kwargs):  # noqa: ANN001
        captured["persist_calls"].append(kwargs)

    monkeypatch.setattr(upload_service, "persist_pdf_upload_to_db", fake_persist_pdf_upload_to_db)

    # LLM および周辺ヘルパを軽量なダミーに差し替え
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
    monkeypatch.setattr(
        upload_service,
        "generate_image_captions_from_pdf",
        lambda contents: [{"page": 1, "caption": "cap"}],  # noqa: ANN001
    )

    result = upload_service.upload_and_generate_qa(
        contents=b"%PDF dummy",
        file_name="scenario.pdf",
        cleanse=False,
        question_llm_model="q-model",
        answer_llm_model="a-model",
        generate_image_captions=True,
        ocr_engine="pypdf",
        ocr_image_compression="balanced",
    )

    assert result["file_id"]
    assert result["questions"]
    assert result["answers"]
    assert result["file_name"] == "scenario.pdf"

    import json as _json

    fid = result["file_id"]
    json_path = extracted_dir / f"{fid}.json"
    assert json_path.exists()
    data = _json.loads(json_path.read_text(encoding="utf-8"))
    assert data["questions"]
    assert data["qa_meta"]
    assert data["file_name"] == "scenario.pdf"

    # パイプライン内で DB 永続化ヘルパが少なくとも一度は呼ばれていること
    assert len(captured["persist_calls"]) >= 1
