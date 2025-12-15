from __future__ import annotations

import sys
from pathlib import Path

import pytest

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.evaluation_service as evaluation_service  # type: ignore  # noqa: E402


@pytest.mark.asyncio
async def test_bulk_evaluate_missing_text_raises_value_error(monkeypatch) -> None:
    """text が無い場合に ValueError が送出されることを確認する。"""

    # DB 永続化を無効化
    monkeypatch.setattr(evaluation_service, "persist_experiment_results", lambda *a, **k: None)

    one = {"embedding_model": "model-x", "questions": ["Q"], "answers": ["A"]}

    # bulk_evaluate 内部では ValueError をキャッチして error 辞書を返す実装
    res = await evaluation_service.bulk_evaluate(one)

    assert isinstance(res, dict)
    assert "error" in res
    assert "textが指定されていません" in res["error"]


@pytest.mark.asyncio
async def test_bulk_evaluate_unsupported_embedding_model(monkeypatch) -> None:
    """未サポートの embedding_model 指定時に ValueError が送出されることを確認する。"""

    from app import main as main_module  # type: ignore  # noqa: E402

    # DB 永続化を無効化
    monkeypatch.setattr(evaluation_service, "persist_experiment_results", lambda *a, **k: None)

    # SUPPORTED_EMBEDDING_MODELS を制御し、get_embeddings が呼ばれないようにする
    monkeypatch.setattr(main_module, "SUPPORTED_EMBEDDING_MODELS", {"supported-model"})

    def fake_get_embeddings(name: str):  # noqa: ANN001
        raise AssertionError("get_embeddings should not be called for unsupported model")

    monkeypatch.setattr(main_module, "get_embeddings", fake_get_embeddings)

    one = {
        "text": "dummy",
        "embedding_model": "unsupported-model",
        "questions": ["Q"],
        "answers": ["A"],
    }

    res = await evaluation_service.bulk_evaluate(one)

    assert isinstance(res, dict)
    assert "error" in res
    assert "未サポートの埋め込みモデル" in res["error"]


@pytest.mark.asyncio
async def test_bulk_evaluate_missing_questions_or_answers(monkeypatch) -> None:
    """questions / answers 欠如時に適切なエラーが発生することを確認する。"""

    from app import main as main_module  # type: ignore  # noqa: E402

    # DB 永続化を無効化
    monkeypatch.setattr(evaluation_service, "persist_experiment_results", lambda *a, **k: None)

    # サポートされるモデルを 1 つだけに絞る
    monkeypatch.setattr(main_module, "SUPPORTED_EMBEDDING_MODELS", {"emb-x"})

    # get_embeddings はこのケースでは呼ばれない
    monkeypatch.setattr(main_module, "get_embeddings", lambda name: "EMB")

    one = {
        "text": "dummy text",
        "embedding_model": "emb-x",
        # questions / answers をあえて省略
    }

    res = await evaluation_service.bulk_evaluate(one)

    assert isinstance(res, dict)
    assert "error" in res
    msg = res["error"]
    assert (
        "questionsが指定されていません" in msg
        or "questions/answersが指定されていません" in msg
        or "PDFアップロードとQA自動生成を先に実施" in msg
    )
