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
