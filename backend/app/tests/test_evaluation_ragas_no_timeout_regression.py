from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# backend ディレクトリを sys.path に追加して app パッケージを import 可能にする
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.evaluation_service as evaluation_service  # type: ignore  # noqa: E402


def test_run_ragas_evaluate_to_pandas_no_timeout(monkeypatch) -> None:
    # ragas.evaluate をモックして、to_pandas を持つ結果を返す
    df = pd.DataFrame(
        [
            {
                "faithfulness": 0.5,
                "answer_relevancy": 0.6,
                "context_precision": 0.7,
                "context_recall": 0.8,
                "answer_correctness": 0.9,
            }
        ]
    )

    class _FakeRes:
        def to_pandas(self):
            return df

    def _fake_evaluate(*, dataset, metrics, llm, embeddings, run_config):  # noqa: ARG001
        # timeout=None（no-timeout）でも run_config 経由で渡せること
        assert run_config is not None
        return _FakeRes()

    monkeypatch.setattr(evaluation_service, "evaluate", _fake_evaluate)

    got = evaluation_service._run_ragas_evaluate_to_pandas(
        dataset=object(),
        metrics=[],
        llm=object(),
        embeddings=object(),
        max_workers=8,
        timeout=None,
    )

    assert got is not None
    assert hasattr(got, "columns")
    assert float(got["faithfulness"].mean()) == 0.5
