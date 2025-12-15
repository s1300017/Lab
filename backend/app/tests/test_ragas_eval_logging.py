from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.evaluation_service as evaluation_service  # type: ignore  # noqa: E402


@pytest.mark.asyncio
async def test_ragas_eval_logger_writes_only_when_path_set(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "ragas_eval.log"

    monkeypatch.setenv("RAGAS_EVAL_LOG_PATH", str(log_path))
    monkeypatch.setenv("RAGAS_EVAL_LOG_LEVEL", "INFO")
    monkeypatch.setenv("RAGAS_EVAL_LOG_MAX_BYTES", "100000")
    monkeypatch.setenv("RAGAS_EVAL_LOG_BACKUP_COUNT", "1")

    # グローバルキャッシュをリセット（環境変数が反映されるように）
    monkeypatch.setattr(evaluation_service, "_RAGAS_EVAL_LOGGER", None)

    ragas_logger = evaluation_service._get_ragas_eval_logger()
    ragas_logger.info(
        "[RAGAS] unit test write",
        extra={"component": "ragas", "endpoint": "bulk_evaluate"},
    )

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "unit test write" in content
    assert '"logger": "ragas_eval"' in content


def test_ragas_eval_logger_no_file_when_path_unset(monkeypatch, tmp_path) -> None:
    # RAGAS_EVAL_LOG_PATH が空ならファイルは作られない（handler 追加されない）
    monkeypatch.delenv("RAGAS_EVAL_LOG_PATH", raising=False)
    monkeypatch.setattr(evaluation_service, "_RAGAS_EVAL_LOGGER", None)

    ragas_logger = evaluation_service._get_ragas_eval_logger()
    ragas_logger.info("[RAGAS] no file")

    # tmp_path配下に何も作られないことをざっくり確認
    assert list(tmp_path.iterdir()) == []
