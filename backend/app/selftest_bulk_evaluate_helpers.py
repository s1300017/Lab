import os

from app.main import _parse_timeout_env, _bool_env, SUPPORTED_EMBEDDING_MODELS


def _test_parse_timeout_env():
    key = "TEST_TIMEOUT_ENV"
    old = os.environ.get(key)
    try:
        os.environ[key] = "30"
        assert _parse_timeout_env(key, 10) == 30
        os.environ[key] = "none"
        assert _parse_timeout_env(key, 10) is None
        if key in os.environ:
            del os.environ[key]
        assert _parse_timeout_env(key, 15) == 15
    finally:
        if old is not None:
            os.environ[key] = old
        elif key in os.environ:
            del os.environ[key]


def _test_bool_env():
    assert _bool_env(True, False) is True
    assert _bool_env(False, True) is False
    assert _bool_env(1, False) is True
    assert _bool_env(0, True) is False
    assert _bool_env("true", False) is True
    assert _bool_env("false", True) is False
    assert _bool_env(None, True) is True


def _test_supported_embedding_models():
    for m in ("huggingface_bge_small", "text-embedding-3-small", "nomic-embed-text"):
        assert m in SUPPORTED_EMBEDDING_MODELS


def run_all_tests():
    _test_parse_timeout_env()
    _test_bool_env()
    _test_supported_embedding_models()


if __name__ == "__main__":
    run_all_tests()
    print("selftest_bulk_evaluate_helpers: OK")
