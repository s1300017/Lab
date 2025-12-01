from __future__ import annotations

"""Ollama 接続先と推論モードを管理するユーティリティ。

main.py から切り出した get_inference_mode / set_inference_mode / get_ollama_base_url
を提供し、他モジュール（ocr_utils など）からも共通利用できるようにする。
"""

from typing import Final
import os
import threading


_ALLOWED_INFERENCE_MODES: Final[set[str]] = {"mac_local", "windows_gpu"}
_INFERENCE_MODE_LOCK = threading.Lock()
_INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "mac_local")
if _INFERENCE_MODE not in _ALLOWED_INFERENCE_MODES:
    _INFERENCE_MODE = "mac_local"


def get_inference_mode() -> str:
    """現在の推論モードを取得する（スレッドセーフ）。"""
    with _INFERENCE_MODE_LOCK:
        return _INFERENCE_MODE


def set_inference_mode(mode: str) -> None:
    """推論モードを更新する（スレッドセーフ）。"""
    if mode not in _ALLOWED_INFERENCE_MODES:
        raise ValueError(f"未対応の推論モードです: {mode}")
    global _INFERENCE_MODE
    with _INFERENCE_MODE_LOCK:
        _INFERENCE_MODE = mode


def get_ollama_base_url() -> str:
    """推論モードに応じてOllamaのベースURLを返すヘルパー。"""
    mode = get_inference_mode()
    mac_url = (
        os.environ.get("OLLAMA_BASE_URL_MAC")
        or os.environ.get("OLLAMA_BASE_URL")
        or "http://ollama:11434"
    )
    if mode == "windows_gpu":
        windows_url = os.environ.get("OLLAMA_BASE_URL_WINDOWS")
        if windows_url:
            return windows_url
    return mac_url
