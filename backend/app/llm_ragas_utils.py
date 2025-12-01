from __future__ import annotations

"""LLM・埋め込み・RAGAS 関連のユーティリティ関数／ラッパークラス群。

main.py から切り出したもので、RAG 用の共通プロンプト生成や
Ollama / HuggingFace / OpenAI 埋め込みの RAGAS 互換ラッパーを提供する。
"""

from dataclasses import dataclass
from typing import Any
import textwrap
import re
from difflib import SequenceMatcher
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

try:  # langchain 新旧バージョン互換
    from langchain_core.messages import BaseMessage
except ImportError:  # noqa: WPS440
    from langchain.schema import BaseMessage  # type: ignore


logger = logging.getLogger(__name__)


def _extract_answer_text(llm_response: Any) -> str:
    """LLM応答オブジェクトからテキスト本体のみを抽出するヘルパー。"""
    try:
        if isinstance(llm_response, BaseMessage):
            return llm_response.content
        if isinstance(llm_response, dict) and "content" in llm_response:
            return llm_response["content"]
        if hasattr(llm_response, "content"):
            return getattr(llm_response, "content")
    except Exception:  # noqa: BLE001
        pass
    return str(llm_response)


def build_rag_answer_prompt(context: str, question: str) -> str:
    """RAG回答用の共通プロンプト文字列を構築するユーティリティ。"""
    answer_system_prompt = textwrap.dedent(
        """
        あなたは日本語のRAGシステムにおける回答エンジンです。以下の制約を厳密に守ってください。
        - 提供されたコンテキストに含まれる事実のみを用いて回答すること。
        - 文書内に記載が見つからない場合は「本文に該当記述がありません。」と明示すること。
        - 回答は自然な日本語で2〜3文以内にまとめること。
        - 重要な根拠がある場合はその文を要約して含めること。
        """
    ).strip()

    return textwrap.dedent(
        f"""
        {answer_system_prompt}

        ### コンテキスト
        {context}

        ### 質問
        {question}

        ### 回答
        """
    ).strip()


ANSWER_DUMMY_PATTERNS = [
    "本文を要約してください",
    "該当内容を本文から要約",
    "情報が見つかりません",
    "記載がありません",
    "わかりません",
]


def split_sentences(text: str) -> list[str]:
    """文書全体を文単位に分割し、スコア計算で利用する。"""
    if not text:
        return []
    segments = re.split(r"[。！？\n]+", text)
    return [seg.strip() for seg in segments if seg.strip()]


def extract_relevant_context(question: str, sentences: list[str], max_sentences: int = 5) -> str:
    """質問と最も関連しそうな文をスコアリングして抽出する。"""
    if not sentences:
        return ""
    keywords = [kw for kw in re.split(r"[\s、,。]", question) if kw]
    scored: list[tuple[float, str]] = []
    for sent in sentences:
        match_hits = sum(1 for kw in keywords if kw and kw in sent)
        similarity = SequenceMatcher(None, question, sent).ratio()
        score = match_hits + similarity
        if score > 0:
            scored.append((score, sent))
    if not scored:
        return ""
    top_sentences = [sent for _, sent in sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]]
    return "\n".join(top_sentences)


def evaluate_answer_quality(answer: str, context_sentences: list[str]) -> dict:
    """回答の品質を判定し、スコアや低信頼フラグを返す。"""
    if not answer:
        return {
            "score": 0.0,
            "is_dummy": True,
            "needs_retry": True,
            "similarity": 0.0,
            "length_score": 0.0,
        }

    lower_answer = answer.lower()
    is_dummy = any(pat.lower() in lower_answer for pat in ANSWER_DUMMY_PATTERNS)
    length_score = min(1.0, len(answer) / 120)
    best_similarity = 0.0
    for sent in context_sentences or [answer]:
        similarity = SequenceMatcher(None, answer, sent).ratio()
        if similarity > best_similarity:
            best_similarity = similarity

    total_score = round((length_score * 0.4 + best_similarity * 0.6), 3)
    needs_retry = is_dummy or len(answer) < 20 or best_similarity < 0.25
    return {
        "score": float(total_score),
        "is_dummy": bool(is_dummy),
        "needs_retry": bool(needs_retry),
        "similarity": float(round(best_similarity, 3)),
        "length_score": float(round(length_score, 3)),
    }


def regenerate_answer_with_context(question: str, context: str, llm: Any, *, max_tokens: int | None = None) -> str:
    """低品質と判定された回答を、より厳密な制約で再生成する。"""
    if not context:
        context = question
    refined_prompt = textwrap.dedent(
        f"""
        あなたは日本語の学術文書に基づいて質問へ回答する専門アシスタントです。
        以下の制約を必ず守って回答してください。
        - 回答は提供されたコンテキストの内容にのみ基づくこと。
        - コンテキストに存在しない情報は「本文に該当記述がありません。」と明示すること。
        - 重要な根拠は1文で引用し、そのまま要約した文を含めること。
        - 回答は3文以内で簡潔にまとめること。

        ### コンテキスト
        {context}

        ### 質問
        {question}

        ### 回答
        """
    ).strip()
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    regenerated = llm.invoke(refined_prompt, **kwargs)
    return _extract_answer_text(regenerated).strip()


class RAGASCompatibleOllamaLLM:
    """RAGAS用に OllamaLLM をラップするクラス。

    - set_run_config の存在を保証
    - プロンプト値の正規化（to_string() 対応など）
    - generate / agenerate / invoke / ainvoke を安全に提供
    """

    def __init__(self, model: str, base_url: str, **kwargs: Any) -> None:
        self._llm = OllamaLLM(model=model, base_url=base_url, **kwargs)

    def set_run_config(self, config: Any) -> None:  # noqa: ARG002
        """RAGAS互換のための no-op 実装。"""
        pass

    def _normalize_prompt_value(self, p: Any) -> str:
        """PromptValue / list などを文字列に正規化する内部ヘルパー。"""
        if p is None:
            return ""
        try:
            to_str = getattr(p, "to_string", None)
            if callable(to_str):
                return to_str()
        except Exception:  # noqa: BLE001
            pass
        if isinstance(p, (list, tuple)):
            return " ".join(self._normalize_prompt_value(pi) for pi in p)
        return p if isinstance(p, str) else str(p)

    def _normalize_prompts(self, prompts: Any) -> list[str]:
        """RAGAS/LLM の generate 系が期待する list[str] を必ず返す。"""
        try:
            to_str = getattr(prompts, "to_string", None)
            if callable(to_str):
                return [to_str()]
        except Exception:  # noqa: BLE001
            pass
        if isinstance(prompts, str):
            return [prompts]
        if isinstance(prompts, (list, tuple)):
            return [self._normalize_prompt_value(p) for p in prompts]
        return [self._normalize_prompt_value(prompts)]

    def _sanitize_kwargs(self, kwargs: dict) -> dict:
        """OllamaLLM が受け付けない kwarg を除去する。"""
        if not kwargs:
            return {}
        drop_keys = {
            "n",
            "best_of",
            "logprobs",
            "top_logprobs",
            "echo",
            "presence_penalty",
            "frequency_penalty",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "num_predict",
            "stop",
            "seed",
        }
        return {k: v for k, v in kwargs.items() if k not in drop_keys}

    def generate(self, prompts: Any, **kwargs: Any) -> Any:
        norm = self._normalize_prompts(prompts)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        return self._llm.generate(norm, **safe_kwargs)

    async def agenerate(self, prompts: Any, **kwargs: Any) -> Any:
        import asyncio
        import inspect

        norm = self._normalize_prompts(prompts)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "agenerate"):
            try:
                res = self._llm.agenerate(norm, **safe_kwargs)
                if inspect.isawaitable(res):
                    return await res
                return res
            except TypeError:  # noqa: BLE001
                pass
        return await asyncio.to_thread(self._llm.generate, norm, **safe_kwargs)

    def invoke(self, prompt: Any, **kwargs: Any) -> Any:
        text = self._normalize_prompt_value(prompt)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "invoke"):
            return self._llm.invoke(text, **safe_kwargs)
        res = self._llm.generate([text], **safe_kwargs)
        try:
            return res.generations[0][0].text
        except Exception:  # noqa: BLE001
            return res

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> Any:
        import asyncio
        import inspect

        text = self._normalize_prompt_value(prompt)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "ainvoke"):
            try:
                res = self._llm.ainvoke(text, **safe_kwargs)
                if inspect.isawaitable(res):
                    return await res
                return res
            except TypeError:  # noqa: BLE001
                pass
        if hasattr(self._llm, "agenerate"):
            try:
                res0 = self._llm.agenerate([text], **safe_kwargs)
                if inspect.isawaitable(res0):
                    res = await res0
                else:
                    res = res0
            except TypeError:  # noqa: BLE001
                res = await asyncio.to_thread(self._llm.generate, [text], **safe_kwargs)
        else:
            res = await asyncio.to_thread(self._llm.generate, [text], **safe_kwargs)
        try:
            return res.generations[0][0].text
        except Exception:  # noqa: BLE001
            return res

    @property
    def client(self) -> Any:
        """Ollama の低レベル client をラップしたプロキシを返す。"""
        try:
            base_client = getattr(self._llm, "client")
        except Exception:  # noqa: BLE001
            return None
        return _RAGASSafeClientProxy(base_client, self._sanitize_kwargs)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - 単純委譲
        return getattr(self._llm, name)


class _RAGASSafeClientProxy:
    """Ollama の低レベル client への呼び出しをラップするプロキシ。"""

    def __init__(self, client: Any, sanitize_fn) -> None:
        self._client = client
        self._sanitize_fn = sanitize_fn

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        """非同期で client.generate を実行する。"""
        import asyncio

        safe_kwargs = self._sanitize_fn(kwargs)
        return await asyncio.to_thread(self._client.generate, *args, **safe_kwargs)

    async def agenerate(self, *args: Any, **kwargs: Any) -> Any:
        return await self.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - 単純委譲
        return getattr(self._client, name)


class RAGASLLMAsyncAdapter:
    """RAGAS が llm.generate(...) を await するケースへのアダプタ。"""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def _sanitize_kwargs(self, kwargs: dict) -> dict:
        try:
            return self._llm._sanitize_kwargs(kwargs)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            drop_keys = {
                "n",
                "best_of",
                "logprobs",
                "top_logprobs",
                "echo",
                "presence_penalty",
                "frequency_penalty",
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
                "num_predict",
                "stop",
                "seed",
            }
            return {k: v for k, v in (kwargs or {}).items() if k not in drop_keys}

    async def generate(self, prompts: Any, **kwargs: Any) -> Any:
        import asyncio
        import inspect

        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "agenerate"):
            try:
                res = self._llm.agenerate(prompts, **safe_kwargs)
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception:  # noqa: BLE001
                pass
        return await asyncio.to_thread(self._llm.generate, prompts, **safe_kwargs)

    async def agenerate(self, prompts: Any, **kwargs: Any) -> Any:
        return await self.generate(prompts, **kwargs)

    async def invoke(self, prompt: Any, **kwargs: Any) -> Any:
        import asyncio
        import inspect

        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "ainvoke"):
            try:
                res = self._llm.ainvoke(prompt, **safe_kwargs)
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception:  # noqa: BLE001
                pass
        return await asyncio.to_thread(self._llm.invoke, prompt, **safe_kwargs)

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> Any:
        return await self.invoke(prompt, **kwargs)

    @property
    def client(self) -> Any:
        try:
            base_client = getattr(self._llm, "client")
        except Exception:  # noqa: BLE001
            return None
        return _RAGASSafeClientProxy(base_client, getattr(self._llm, "_sanitize_kwargs", lambda x: x))

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - 単純委譲
        return getattr(self._llm, name)


class RAGASCompatibleOllamaEmbeddings:
    """RAGAS 用 OllamaEmbeddings ラッパー。"""

    def __init__(self, model: str, base_url: str) -> None:
        self._emb = OllamaEmbeddings(model=model, base_url=base_url)

    def set_run_config(self, config: Any) -> None:  # noqa: ARG002
        pass

    async def embed_text(self, text: str):
        import asyncio

        if hasattr(self._emb, "embed_query"):
            return await asyncio.to_thread(self._emb.embed_query, text)
        vecs = await asyncio.to_thread(self._emb.embed_documents, [text])
        return vecs[0] if vecs else []

    async def aembed_text(self, text: str):
        return await self.embed_text(text)

    def embed_documents(self, texts):
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str):
        try:
            if isinstance(getattr(self, "_model_name", None), str) and "jina-embeddings-v4" in getattr(self, "_model_name"):
                client = getattr(self._emb, "client", None)
                if client is not None and hasattr(client, "encode"):
                    vec = client.encode(
                        sentences=[text],
                        task="retrieval",
                        prompt_name="query",
                        normalize_embeddings=True,
                    )[0]
                    return vec.tolist() if hasattr(vec, "tolist") else vec
        except Exception:  # noqa: BLE001
            pass
        return self._emb.embed_query(text)

    async def aembed_documents(self, texts):
        import asyncio

        return await asyncio.to_thread(self.embed_documents, texts)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - 単純委譲
        return getattr(self._emb, name)

    def __call__(self, text: str):
        return self.embed_query(text)


class RAGASCompatibleHuggingFaceEmbeddings:
    """RAGAS 用 HuggingFaceEmbeddings ラッパー。"""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._model_name = model_name
        self._emb = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        self._log_once: dict[str, bool] = {"doc": False, "qry": False, "doc_fallback": False, "qry_fallback": False}

    def set_run_config(self, config: Any) -> None:  # noqa: ARG002
        pass

    async def embed_text(self, text: str):
        import asyncio

        if hasattr(self._emb, "embed_query"):
            return await asyncio.to_thread(self._emb.embed_query, text)
        vecs = await asyncio.to_thread(self._emb.embed_documents, [text])
        return vecs[0] if vecs else []

    async def aembed_text(self, text: str):
        return await self.embed_text(text)

    def embed_documents(self, texts):
        try:
            if isinstance(self._model_name, str) and "jina-embeddings-v4" in self._model_name:
                client = getattr(self._emb, "_client", None) or getattr(self._emb, "client", None)
                if client is not None and hasattr(client, "encode"):
                    vecs = client.encode(
                        sentences=texts,
                        task="retrieval",
                        prompt_name="passage",
                        normalize_embeddings=True,
                    )
                    if not self._log_once["doc"]:
                        logger.info(
                            "[emb] Jina v4 encode(passages) path used: task=retrieval, prompt_name=passage, normalize_embeddings=True",
                        )
                        self._log_once["doc"] = True
                    return vecs.tolist() if hasattr(vecs, "tolist") else vecs
        except Exception:  # noqa: BLE001
            pass
        if not self._log_once["doc_fallback"]:
            logger.info("[emb] embed_documents fallback path used (HuggingFaceEmbeddings)")
            self._log_once["doc_fallback"] = True
        return self._emb.embed_documents(texts)

    async def aembed_documents(self, texts):
        import asyncio

        return await asyncio.to_thread(self._emb.embed_documents, texts)

    def embed_query(self, text: str):
        try:
            if isinstance(self._model_name, str) and "jina-embeddings-v4" in self._model_name:
                client = getattr(self._emb, "_client", None) or getattr(self._emb, "client", None)
                if client is not None and hasattr(client, "encode"):
                    vec = client.encode(
                        sentences=[text],
                        task="retrieval",
                        prompt_name="query",
                        normalize_embeddings=True,
                    )[0]
                    if not self._log_once["qry"]:
                        logger.info(
                            "[emb] Jina v4 encode(query) path used: task=retrieval, prompt_name=query, normalize_embeddings=True",
                        )
                        self._log_once["qry"] = True
                    return vec.tolist() if hasattr(vec, "tolist") else vec
        except Exception:  # noqa: BLE001
            pass
        if not self._log_once["qry_fallback"]:
            logger.info("[emb] embed_query fallback path used (HuggingFaceEmbeddings)")
            self._log_once["qry_fallback"] = True
        return self._emb.embed_query(text)

    async def aembed_query(self, text: str):
        import asyncio

        return await asyncio.to_thread(self.embed_query, text)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - 単純委譲
        return getattr(self._emb, name)

    def __call__(self, text: str):
        return self.embed_query(text)


class RAGASCompatibleOpenAIEmbeddings:
    """RAGAS 用 OpenAIEmbeddings ラッパー。"""

    def __init__(self, **kwargs: Any) -> None:
        self._emb = OpenAIEmbeddings(**kwargs)

    def set_run_config(self, config: Any) -> None:  # noqa: ARG002
        pass

    async def embed_text(self, text: str):
        import asyncio

        if hasattr(self._emb, "embed_query"):
            return await asyncio.to_thread(self._emb.embed_query, text)
        vecs = await asyncio.to_thread(self._emb.embed_documents, [text])
        return vecs[0] if vecs else []

    async def aembed_text(self, text: str):
        return await self.embed_text(text)

    def embed_documents(self, texts):
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str):
        return self._emb.embed_query(text)

    async def aembed_documents(self, texts):
        import asyncio

        return await asyncio.to_thread(self._emb.embed_documents, texts)

    async def aembed_query(self, text: str):
        import asyncio

        return await asyncio.to_thread(self._emb.embed_query, text)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - 単純委譲
        return getattr(self._emb, name)

    def __call__(self, text: str):
        return self.embed_query(text)
