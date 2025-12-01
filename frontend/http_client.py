# 共通HTTPクライアント（接続プール + Keep-Alive + リトライ）
# フロントエンド全体で使い回すセッションを提供
# 注意: フロントエンド側では timeout を指定しない（ユーザー方針）

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests

_http_session = None


def get_http_session():
    """接続プール・リトライ設定済みの requests.Session を返す（シングルトン）。
    - Keep-Aliveでコネクション再利用
    - 429/5xx/接続系エラーに対する自動リトライ
    - POST/GET/DELETEなど主要メソッドを対象
    """
    global _http_session
    if _http_session is None:
        s = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1.0,  # 指数バックオフ（1, 2, 4, ...秒）
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"},
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({"Connection": "keep-alive"})
        _http_session = s
    return _http_session


def http_get(url: str, **kwargs):
    """共通GET。timeoutは指定しない。"""
    return get_http_session().get(url, **kwargs)


def http_post(url: str, **kwargs):
    """共通POST。timeoutは指定しない。"""
    return get_http_session().post(url, **kwargs)


def http_delete(url: str, **kwargs):
    """共通DELETE。timeoutは指定しない。"""
    return get_http_session().delete(url, **kwargs)


def format_http_error(resp: requests.Response) -> str:
    """HTTPレスポンスからステータスコードとエラーメッセージを整形して返す。

    - Content-Type が JSON の場合は detail / error フィールドを優先して利用
    - それ以外の場合は resp.text をそのまま利用
    - 代表的なステータスに対して簡易的な対処ヒントを付与
    """
    status = getattr(resp, "status_code", None)
    text = ""
    try:
        if resp.headers.get("Content-Type", "").startswith("application/json"):
            data = resp.json() or {}
            text = data.get("detail") or data.get("error") or resp.text
        else:
            text = resp.text
    except Exception:
        text = getattr(resp, "text", "")

    if status is None:
        return text or "不明なエラーが発生しました。"

    base = f"{status} {text}" if text else str(status)

    # 代表的なHTTPステータスやエラーメッセージに応じて、ユーザー向けの対処ヒントを付与
    hints: list[str] = []
    if status in {502, 503, 504}:
        hints.append("バックエンドサーバーやモデルプロセスが起動中か、ネットワーク状態を確認してください。")
    if status == 404:
        hints.append("エンドポイントURLまたは対象リソース（PDF ID / 実験ID など）が正しいか確認してください。")
    lowered = (text or "").lower()
    if "connection refused" in lowered or "connectionerror" in lowered:
        hints.append("BACKEND_URL の設定とバックエンドの起動状態（ポート番号含む）を確認してください。")
    if not hints:
        hints.append("詳細な原因はバックエンド側のログを確認してください。")

    hint_str = " / ".join(hints)
    return f"{base}（対処の目安: {hint_str}）"
