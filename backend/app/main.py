from datetime import datetime
from pytz import timezone
import os

def jst_now_str():
    return datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import threading

# --- PDF・抽出データ保存用ディレクトリのグローバル定義 ---
import uuid
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PDF_DIR = DATA_DIR / "pdf"
EXTRACTED_DIR = DATA_DIR / "extracted"
PDF_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

print(f"[{jst_now_str()}] === FastAPI main.py 起動開始 [テスト用] ===")

# データベース接続設定
POSTGRES_DB = os.environ.get("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "rag_password")
DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@db:5432/{POSTGRES_DB}"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPIアプリケーションの初期化
app = FastAPI()

# --- Dockerヘルスチェック用エンドポイント ---
@app.get("/health")
def health_check():
    """Docker用のシンプルなヘルスチェックAPI"""
    return {"status": "ok"}

# サーバ起動時にデータベースを初期化
@app.on_event("startup")
async def startup_event():
    print(f"[{jst_now_str()}] [DEBUG] startup_event呼び出し")
    print(f"[{jst_now_str()}] [DEBUG] DB_URL = {os.getenv('DATABASE_URL')}")
    
    # データベース接続をテスト
    max_retries = 5
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
        try:
            print(f"[{jst_now_str()}] [DEBUG] データベース接続を試行中... (試行 {attempt + 1}/{max_retries})")
            init_db()
            print(f"[{jst_now_str()}] [DEBUG] データベース初期化に成功しました")
            break
        except Exception as e:
            print(f"[{jst_now_str()}] [ERROR] データベース初期化エラー (試行 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"[{jst_now_str()}] [CRITICAL] データベース初期化に失敗しました。最大試行回数に達しました。")
                raise
            import time
            time.sleep(retry_delay)

    # --- 起動時にOllamaモデルをバックグラウンドでウォームアップ（任意）---
    try:
        def _warmup():
            try:
                import urllib.request, urllib.error
                import json as _json
                base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
                targets = [
                    {"model": "mistral:latest", "prompt": "ping", "stream": False},
                    {"model": "llama3:latest", "prompt": "ping", "stream": False},
                    {"model": "gpt-oss:20b", "prompt": "ping", "stream": False},
                ]
                for body in targets:
                    try:
                        req = urllib.request.Request(
                            url=f"{base_url.rstrip('/')}/api/generate",
                            data=_json.dumps(body).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            _ = resp.read()
                        print(f"[ウォームアップ] {body['model']} 成功")
                    except Exception as we:
                        print(f"[ウォームアップ警告] {body.get('model')} 失敗: {we}")
            except Exception as e:
                print(f"[ウォームアップ初期化失敗] {e}")

        threading.Thread(target=_warmup, daemon=True).start()
    except Exception as e:
        print(f"[ウォームアップ起動失敗] {e}")

def init_db():
    print(f"[{jst_now_str()}] [DEBUG] init_db呼び出し")
    try:
        print(f"[{jst_now_str()}] [DEBUG] データベース接続テスト開始")
        with engine.connect() as conn:
            print(f"[{jst_now_str()}] [DEBUG] データベース接続成功")
            
            # トランザクションを開始
            with conn.begin():
                # テーブルが存在するか確認
                result = conn.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'embeddings');"
                ))
                table_exists = result.scalar()
                
                if not table_exists:
                    print(f"[{jst_now_str()}] [INFO] embeddingsテーブルを作成します")
                    conn.execute(text("""
                        CREATE TABLE embeddings (
                            id SERIAL PRIMARY KEY,
                            text TEXT NOT NULL,
                            embedding_model TEXT NOT NULL,
                            chunk_strategy TEXT NOT NULL,
                            chunk_size INTEGER,
                            chunk_overlap INTEGER,
                            avg_chunk_len FLOAT,
                            num_chunks INTEGER,
                            overall_score FLOAT,
                            faithfulness FLOAT,
                            answer_relevancy FLOAT,
                            context_recall FLOAT,
                            context_precision FLOAT,
                            answer_correctness FLOAT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                    print(f"[{jst_now_str()}] [INFO] embeddingsテーブルを作成しました")
                else:
                    print(f"[{jst_now_str()}] [INFO] embeddingsテーブルは既に存在します")
                    
                # コミットは自動的に行われる
                
    except Exception as e:
        print(f"[{jst_now_str()}] [ERROR] データベース初期化エラー: {str(e)}")
        # エラーの詳細をログに出力
        import traceback
        print(f"[{jst_now_str()}] [ERROR] スタックトレース:\n{traceback.format_exc()}")
        raise
    for route in app.routes:
        print(f"[{jst_now_str()}] [ROUTE]", route.path, route.methods)

# サーバ起動時にルート一覧を出力
import threading

def print_routes():
    import time
    time.sleep(2)  # サーバ起動待ち
    print(f"[{jst_now_str()}] === FastAPI登録ルート一覧 ===")
    for route in app.routes:
        print(f"[{jst_now_str()}] [ROUTE]", route.path, route.methods)
threading.Thread(target=print_routes, daemon=True).start()

# --- PDFアップロード＆QA自動生成API ---
from fastapi import UploadFile, File
from PyPDF2 import PdfReader

@app.post("/uploadfile/")
async def uploadfile(file: UploadFile = File(...), cleanse: bool = Form(False), question_llm_model: str = Form("mistral"), answer_llm_model: str = Form("mistral")):
    """
    PDFアップロード時にテキスト抽出→LLMで質問自動生成→LLMで回答自動生成まで行い、
    質問・回答セットを返すAPI。
    """
    print(f"[{jst_now_str()}][重要] uploadfile関数実行開始: ファイル名={file.filename}, サイズ={getattr(file, 'size', '不明')}")
    print(f"[{jst_now_str()}][重要] ファイル情報: {file=}, タイプ={type(file)}")
    import io
    try:
        file_id = str(uuid.uuid4())  # ← ここで必ずfile_idを発行
        # 1. PDFからテキスト抽出
        contents = await file.read()
        print(f"[{jst_now_str()}][重要] ファイル読み込み完了: {len(contents)}バイト")
        pdf_stream = io.BytesIO(contents)
        print(f"[重要] BytesIOストリーム作成完了: {pdf_stream.getbuffer().nbytes}バイト")
        try:
            reader = PdfReader(pdf_stream)
            print(f"[重要] PdfReader初期化成功: {len(reader.pages)}ページ")
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
                print(f"[重要] ページ抽出: {len(page_text)}文字")
            # クレンジング処理（オプション）
            if cleanse:
                print("[重要] クレンジング処理を実施します")
                text = cleanse_pdf_text(text)
            sample_text = text[:3000] if len(text) > 3000 else text
            print(f"[重要] PDF抽出完了: 合計{len(text)}文字, サンプル={sample_text[:100]}...")
        except Exception as pdf_error:
            print(f"[重要] PDF処理エラー: {pdf_error}")
            return {"error": f"PDF処理エラー: {str(pdf_error)}"}
        print("[重要] LLM質問生成開始 (GPT-OSS固定)")
        # モデル指定は無視し、内部で常にGPT-OSSを使用
        llm_q_instance = get_llm("gpt-oss")
        prompt_q = f"""以下の内容に関する代表的な質問を日本語で5つ作成してください。\n---\n{text[:1500]}\n---\n質問："""
        try:
            questions_resp = llm_q_instance.invoke(prompt_q)
            print(f"[重要] LLM質問生成レスポンス取得: {len(questions_resp.content)}文字")
            questions = [q.strip() for q in questions_resp.content.split('\n') if q.strip()]
            print(f"[重要] 質問リスト生成完了: {len(questions)}件")
        except Exception as e:
            print(f"[重要] LLM質問生成例外: {e}")
            questions = []
        if not questions:
            import re
            print("[重要] 正規表現によるQA/箇条書き抽出開始")
            bullets = re.findall(r'^[\*\-\d\.]+\s*(.+)', text, re.MULTILINE)
            qas = re.findall(r'Q[\d：: ]*(.+?)\nA[\d：: ]*(.+?)(?=\nQ|\n\Z)', text, re.DOTALL)
            if qas:
                questions = [q.strip() for q, a in qas]
                answers = [a.strip() for q, a in qas]
            elif bullets:
                questions = bullets[:5]
                answers = ["該当内容を本文から要約してください。"] * len(questions)
            else:
                paras = [p.strip() for p in text.split('\n') if p.strip()]
                questions = [f"{p[:20]}について説明してください。" for p in paras[:5]]
                answers = ["該当内容を本文から要約してください。"] * len(questions)
        else:
            answers = []
            # モデル指定は無視し、内部で常にGPT-OSSを使用
            llm_a_instance = get_llm("gpt-oss")
            for i, q in enumerate(questions):
                try:
                    prompt_a = f"""
以下の内容に基づいて、次の質問に日本語で簡潔に答えてください。\n---\n{sample_text}\n---\n質問: {q}\n回答：
"""
                    answer_resp = llm_a_instance.invoke(prompt_a)
                    print(f"[DEBUG] answer_resp={{answer_resp}}, type={{type(answer_resp)}}")
                    # 型ガード: content属性・str型対応
                    if hasattr(answer_resp, "content"):
                        answer = answer_resp.content.strip().split('\n')[0]
                    elif isinstance(answer_resp, str):
                        answer = answer_resp.strip().split('\n')[0]
                    else:
                        answer = str(answer_resp)
                    print(f"[重要] LLM回答{{i+1}}生成完了: {{len(answer)}}文字")
                    answers.append(answer)
                except Exception as e:
                    import traceback
                    print(f"[重要] LLM回答{{i+1}}生成例外: {{e}}")
                    traceback.print_exc()
                    answers.append("該当内容を本文から要約してください。")
        if not questions or not answers:
            print("[重要] ダミーQAセットを返却（questions/answersが空）")
            questions = ["この文書の主題は何ですか？"]
            answers = ["本文を要約してください。"]
        # --- qa_meta を生成（回答長の正規化スコア + ダミー回答フラグ）---
        try:
            max_len = max((len(a) for a in answers), default=1)
            dummy_patterns = ["該当内容を本文から要約", "本文を要約して"]
            qa_meta = []
            for a in answers:
                norm_len = (len(a) / max_len) if max_len else 0.0
                is_dummy = any(pat in a for pat in dummy_patterns)
                qa_meta.append({
                    "score": float(round(norm_len, 3)),
                    "is_auto_fixed": False,
                    "is_dummy_answer": bool(is_dummy),
                    "candidates": [a],
                    "candidate_scores": [float(round(norm_len, 3))]
                })
        except Exception as e:
            print(f"[警告] qa_meta生成時に例外: {e}。全件デフォルト値を設定します")
            qa_meta = [{
                "score": 1.0,
                "is_auto_fixed": False,
                "is_dummy_answer": False,
                "candidates": [a],
                "candidate_scores": [1.0]
            } for a in answers]

        print(f"[重要] API返却直前: questions={questions}, answers={answers}")
        # 4. 抽出データ保存
        extracted_path = EXTRACTED_DIR / f"{file_id}.json"
        with open(extracted_path, "w", encoding="utf-8") as f_json:
            json.dump({
                "text": sample_text,
                "questions": questions,
                "answers": answers,
                "qa_meta": qa_meta,
                "file_name": file.filename,  # ←file_nameで統一
            }, f_json, ensure_ascii=False)
        # PDFファイル保存
        pdf_path = PDF_DIR / f"{file_id}.pdf"
        with open(pdf_path, "wb") as f_pdf:
            f_pdf.write(contents)
        # 5. file_id付きで返却
        return {
            "file_id": file_id,
            "text": sample_text,
            "questions": questions,
            "answers": answers,
            "qa_meta": qa_meta,
            "file_name": file.filename,  # ←file_nameで統一
        }
    except Exception as e:
        print(f"[重要] uploadfile全体例外: {e}")
        return {"error": str(e)}

# --- PDFクレンジング関数 ---
def cleanse_pdf_text(text: str) -> str:
    import re
    lines = text.split('\n')
    # 表形式やノイズ行の除去例
    cleansed = [
        line for line in lines
        if not re.match(r'^\s*[\|\-]{2,}', line) and len(re.findall(r'\|', line)) < 3
    ]
    # 連続空白行の削除
    result = []
    prev_blank = False
    for line in cleansed:
        if line.strip() == "":
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    return '\n'.join(result)

# --- 新規: file_idで抽出済みデータ取得API ---
from fastapi import HTTPException
@app.get("/get_extracted/{file_id}")
def get_extracted(file_id: str):
    """
    指定file_idの抽出テキスト・QA・ファイル名を返すAPI。
    """
    extracted_path = EXTRACTED_DIR / f"{file_id}.json"
    if not extracted_path.exists():
        raise HTTPException(status_code=404, detail=f"file_id={file_id}の抽出データが見つかりません")
    with open(extracted_path, "r", encoding="utf-8") as f_json:
        data = json.load(f_json)
    # PDF本体もbase64で必ず返す
    pdf_path = PDF_DIR / f"{file_id}.pdf"
    if pdf_path.exists():
        import base64
        with open(pdf_path, "rb") as f_pdf:
            data["pdf_bytes_base64"] = base64.b64encode(f_pdf.read()).decode('utf-8')
    # file_nameがなければfile_id.pdfをセット（後方互換）
    if "file_name" not in data:
        data["file_name"] = f"{file_id}.pdf"
    return data


from pydantic import BaseModel
import PyPDF2
import io
import os
import sys
print("[CRITICAL] main.pyロード開始")
from pathlib import Path

# --- models.yaml, strategies.yaml 読み込み用 ---
try:
    import yaml
except ImportError:
    yaml = None  # PyYAMLが未導入の場合

# 設定ファイルのパス（Dockerコンテナ内の絶対パスを指定）
MODELS_YAML_PATH = Path("/app/models.yaml")
STRATEGIES_YAML_PATH = Path("/app/strategies.yaml")

# モデルリスト取得関数
def load_models_yaml():
    if yaml is None:
        raise RuntimeError("PyYAMLがインストールされていません。requirements.txtに 'pyyaml' を追加してください。");
    if not MODELS_YAML_PATH.exists():
        raise FileNotFoundError(f"models.yamlが見つかりません: {MODELS_YAML_PATH}")
    with open(MODELS_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 戦略リスト取得関数
def load_strategies_yaml():
    if yaml is None:
        raise RuntimeError("PyYAMLがインストールされていません。requirements.txtに 'pyyaml' を追加してください。");
    if not STRATEGIES_YAML_PATH.exists():
        raise FileNotFoundError(f"strategies.yamlが見つかりません: {STRATEGIES_YAML_PATH}")
    with open(STRATEGIES_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import os
# NLTK参照パスを明示的に複数指定
nltk.data.path = ['/usr/local/share/nltk_data', '/usr/local/lib/nltk_data'] + nltk.data.path
print('[NLTK] data search path:', nltk.data.path)
# punktを明示的にダウンロード
nltk.download('punkt', download_dir='/usr/local/share/nltk_data')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def fixed_chunk_text(text, chunk_size=1000, chunk_overlap=0):
    """
    固定長でテキストをチャンク分割
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap if chunk_overlap < chunk_size else chunk_size
    return chunks

def sentence_chunk_text(text):
    """
    spaCy日本語モデルで文単位に分割
    """
    try:
        import spacy
        try:
            nlp = spacy.load("ja_core_news_sm")
        except OSError:
            raise RuntimeError("spaCyの日本語モデル 'ja_core_news_sm' がインストールされていません。\n\n下記コマンドでインストールしてください:\n\npython -m spacy download ja_core_news_sm\n")
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        raise RuntimeError(f"spaCyによる日本語文分割時にエラー: {str(e)}")

def paragraph_chunk_text(text):
    """
    段落単位で分割（1つ以上の改行（\n+）で区切る）
    例：章・条文ごとに1つの改行でも分割されます。
    """
    import re
    paras = re.split(r'\n+', text)
    return [p.strip() for p in paras if p.strip()]

def semantic_chunk_text(text, chunk_size=None, chunk_overlap=None, embedding_model=None, similarity_threshold=0.7):
    """
    セマンティックチャンク分割：
    1. 文単位で分割
    2. 各文のembeddingを取得
    3. コサイン類似度で分割点を決定し、意味的に自然なチャンクを作成
    
    Note:
        chunk_size と chunk_overlap パラメータは互換性のために残されていますが、
        セマンティックチャンキングでは使用されません。
    
    Args:
        text: 分割するテキスト
        chunk_size: 互換性のためのパラメータ（使用されません）
        chunk_overlap: 互換性のためのパラメータ（使用されません）
        embedding_model: 埋め込みモデル（必須）
        similarity_threshold: センテンス間の類似度閾値（0〜1）
    
    Returns:
        list: チャンク化されたテキストのリスト
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # spaCy日本語モデルで文単位に分割
    try:
        import spacy
        nlp = spacy.load("ja_core_news_sm")
    except OSError:
        raise RuntimeError("spaCyの日本語モデル 'ja_core_news_sm' がインストールされていません。\n\npython -m spacy download ja_core_news_sm\n")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return [text]
    print(f"セマンティックチャンキング: {len(sentences)}文を処理中...")
    if embedding_model is None:
        raise ValueError("embedding_modelが指定されていません")
    batch_size = 32
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    chunks = []
    current_chunk = []
    for i in range(len(sentences)):
        current_sentence = sentences[i]
        current_embedding = np.array(embeddings[i]).reshape(1, -1)
        if not current_chunk:
            current_chunk.append(current_sentence)
            continue
        last_embedding = np.array(embeddings[i-1]).reshape(1, -1)
        similarity = cosine_similarity(last_embedding, current_embedding)[0][0]
        if similarity < similarity_threshold:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [current_sentence]
        else:
            current_chunk.append(current_sentence)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    print(f"セマンティックチャンキング完了: {len(chunks)}個のチャンクを生成")
    return chunks

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.pgvector import PGVector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,  # 追加: 回答の正確性指標
    answer_similarity,
)
from datasets import Dataset

# モデルパスの設定
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LOCAL_MODEL_PATH = Path("/app/models/BAAI_bge-small-en-v1.5")

# セキュリティ注意: 本番環境ではAPIキーの表示は避けてください
import logging
logging.basicConfig(level=logging.INFO)
logging.info(f"[起動時] OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")

# --- RAGAS互換ラッパー（set_run_config を要求する環境向け）---
class RAGASCompatibleOllamaLLM:
    """
    RAGASが set_run_config を呼んでもエラーにしないための薄いラッパー。
    それ以外の属性・メソッドは元のOllamaLLMに委譲する。
    """
    def __init__(self, model: str, base_url: str, **kwargs):
        self._llm = OllamaLLM(model=model, base_url=base_url, **kwargs)

    # RAGASが存在チェックすることがある
    def set_run_config(self, config):
        pass

    # --- ここからRAGAS互換のためのプロンプト正規化ロジックを追加 ---
    def _normalize_prompt_value(self, p):
        """
        LangChainのStringPromptValue/ChatPromptValue等が来ても
        .to_string() で文字列化する。リスト/タプルは要素を再帰的に処理。
        それ以外はstr()にフォールバック。
        """
        if p is None:
            return ""
        # StringPromptValue / ChatPromptValue を想定（duck-typing）
        try:
            to_str = getattr(p, "to_string", None)
            if callable(to_str):
                return to_str()
        except Exception:
            pass
        # リスト/タプルは各要素を文字列化して結合（単一プロンプト扱い時）
        if isinstance(p, (list, tuple)):
            return " ".join(self._normalize_prompt_value(pi) for pi in p)
        # すでに文字列ならそのまま、その他はstrにフォールバック
        return p if isinstance(p, str) else str(p)

    def _normalize_prompts(self, prompts):
        """
        RAGAS/LLMのgenerate系が期待する list[str] を必ず返す。
        - StringPromptValue/ChatPromptValue → [str]
        - str → [str]
        - list/tuple → 各要素を文字列化
        - それ以外 → [str(obj)]
        """
        # 文字列化可能なPromptValue（to_string持ち）を単一として扱う
        try:
            to_str = getattr(prompts, "to_string", None)
            if callable(to_str):
                return [to_str()]
        except Exception:
            pass
        if isinstance(prompts, str):
            return [prompts]
        if isinstance(prompts, (list, tuple)):
            return [self._normalize_prompt_value(p) for p in prompts]
        return [self._normalize_prompt_value(prompts)]

    def _sanitize_kwargs(self, kwargs: dict) -> dict:
        """OllamaLLMが受け付けないkwargを除去する。
        代表的にはOpenAI系の `n`, `best_of`, `logprobs`, `echo` に加え、
        Ollamaの低レベルclient.generateがトップレベルでは受け付けない
        生成制御系（temperature, max_tokens, top_p, top_k, num_predict, stop, seed）を除去。
        """
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
            # 低レベルclient.generateにトップレベルで渡すとTypeErrorになる代表例
            "temperature",
            "max_tokens",   # OpenAI系 → Ollamaはoptions.num_predict
            "top_p",
            "top_k",
            "num_predict",
            "stop",
            "seed",
        }
        return {k: v for k, v in kwargs.items() if k not in drop_keys}

    # RAGASが直接呼び出すことがあるAPIをラップ
    def generate(self, prompts, **kwargs):
        """promptsをlist[str]へ正規化してから委譲"""
        norm = self._normalize_prompts(prompts)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        return self._llm.generate(norm, **safe_kwargs)

    async def agenerate(self, prompts, **kwargs):
        """promptsをlist[str]へ正規化してから委譲（async）"""
        norm = self._normalize_prompts(prompts)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        # 一部の実装でagenerateが存在しても同期返却の場合があるため awaitable 判定を行う
        if hasattr(self._llm, "agenerate"):
            try:
                res = self._llm.agenerate(norm, **safe_kwargs)
                import inspect
                if inspect.isawaitable(res):
                    return await res
                # 同期返却（list/LLMResultなど）の場合はそのまま返す
                return res
            except TypeError:
                # 同期実装に対してawaitしてしまった等の互換性問題に備えフォールバック
                pass
        # フォールバック：スレッドでgenerateを呼ぶ
        import asyncio
        return await asyncio.to_thread(self._llm.generate, norm, **safe_kwargs)

    def invoke(self, prompt, **kwargs):
        """単一プロンプト入力のラッパー。Runnable互換向け"""
        text = self._normalize_prompt_value(prompt)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "invoke"):
            return self._llm.invoke(text, **safe_kwargs)
        # フォールバック：generateの最初のテキストを返す
        res = self._llm.generate([text], **safe_kwargs)
        try:
            # LangChain LLMResult 互換の取り出し
            return res.generations[0][0].text
        except Exception:
            return res

    async def ainvoke(self, prompt, **kwargs):
        """単一プロンプト入力の非同期ラッパー。Runnable互換向け"""
        text = self._normalize_prompt_value(prompt)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "ainvoke"):
            try:
                res = self._llm.ainvoke(text, **safe_kwargs)
                import inspect
                if inspect.isawaitable(res):
                    return await res
                return res
            except TypeError:
                pass
        # フォールバック：agenerate→最初のテキストを返す
        if hasattr(self._llm, "agenerate"):
            try:
                res0 = self._llm.agenerate([text], **safe_kwargs)
                import inspect
                if inspect.isawaitable(res0):
                    res = await res0
                else:
                    res = res0
            except TypeError:
                # フォールバック：スレッドでgenerate
                import asyncio
                res = await asyncio.to_thread(self._llm.generate, [text], **safe_kwargs)
        else:
            import asyncio
            res = await asyncio.to_thread(self._llm.generate, [text], **safe_kwargs)
        try:
            return res.generations[0][0].text
        except Exception:
            return res
    # --- ここまで追加 ---

    @property
    def client(self):
        """RAGASや内部実装が直接 .client.generate(**kwargs) を呼んでも
        受け付けないkwarg（例: n）を除去できるようにプロキシを返す。
        """
        try:
            base_client = getattr(self._llm, "client")
        except Exception:
            return None
        return _RAGASSafeClientProxy(base_client, self._sanitize_kwargs)

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def set_run_config(self, config):
        """RAGAS互換のためのno-op。"""
        pass


class _RAGASSafeClientProxy:
    """Ollamaの低レベルclientへの呼び出しをラップし、
    OpenAI互換kwarg（n等）を除去してから委譲する簡易プロキシ。
    """
    def __init__(self, client, sanitize_fn):
        self._client = client
        self._sanitize_fn = sanitize_fn

    async def generate(self, *args, **kwargs):
        """非同期で低レベルclient.generateを実行。
        ragas側がawaitしてもTypeErrorにならないようにするため、to_threadで包む。
        """
        safe_kwargs = self._sanitize_fn(kwargs)
        import asyncio
        return await asyncio.to_thread(self._client.generate, *args, **safe_kwargs)

    async def agenerate(self, *args, **kwargs):
        """generateのエイリアス（async）。"""
        return await self.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


class RAGASLLMAsyncAdapter:
    """
    RAGASが llm.generate(...) を await する経路に入っても例外にならないよう、
    generate/invoke を非同期対応で提供するアダプタ。
    既存の `RAGASCompatibleOllamaLLM` を内包して使うことを想定。
    """
    def __init__(self, llm):
        self._llm = llm

    def _sanitize_kwargs(self, kwargs: dict) -> dict:
        try:
            # 既存ラッパのサニタイズを流用
            return self._llm._sanitize_kwargs(kwargs)
        except Exception:
            # 最低限の防御（OpenAI系の代表キーを除去）
            drop_keys = {
                "n", "best_of", "logprobs", "top_logprobs", "echo",
                "presence_penalty", "frequency_penalty",
                "temperature", "max_tokens", "top_p", "top_k",
                "num_predict", "stop", "seed",
            }
            return {k: v for k, v in (kwargs or {}).items() if k not in drop_keys}

    async def generate(self, prompts, **kwargs):
        """await 可能な generate を提供。内部で agenerate か to_thread を使用"""
        safe_kwargs = self._sanitize_kwargs(kwargs)
        # agenerate が存在しても同期返却の場合があるため awaitable 判定を行う
        if hasattr(self._llm, "agenerate"):
            try:
                res = self._llm.agenerate(prompts, **safe_kwargs)
                import inspect
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception:
                pass
        # フォールバック：スレッドでgenerate
        import asyncio
        return await asyncio.to_thread(self._llm.generate, prompts, **safe_kwargs)

    async def agenerate(self, prompts, **kwargs):
        return await self.generate(prompts, **kwargs)

    async def invoke(self, prompt, **kwargs):
        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "ainvoke"):
            try:
                res = self._llm.ainvoke(prompt, **safe_kwargs)
                import inspect
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception:
                pass
        # フォールバック：スレッドでinvoke
        import asyncio
        return await asyncio.to_thread(self._llm.invoke, prompt, **safe_kwargs)

    async def ainvoke(self, prompt, **kwargs):
        return await self.invoke(prompt, **kwargs)

    @property
    def client(self):
        try:
            base_client = getattr(self._llm, "client")
        except Exception:
            return None
        return _RAGASSafeClientProxy(base_client, getattr(self._llm, "_sanitize_kwargs", lambda x: x))

    def __getattr__(self, name):
        return getattr(self._llm, name)


class RAGASCompatibleOllamaEmbeddings:
    """
    RAGASが set_run_config を呼んでもエラーにしないための薄いラッパー。
    それ以外の属性・メソッドは元のOllamaEmbeddingsに委譲する。
    """
    def __init__(self, model: str, base_url: str):
        self._emb = OllamaEmbeddings(model=model, base_url=base_url)

    def set_run_config(self, config):
        pass

    async def embed_text(self, text: str):
        """RAGASがawaitしても安全な単一テキスト埋め込みAPI（async）。"""
        import asyncio
        if hasattr(self._emb, "embed_query"):
            return await asyncio.to_thread(self._emb.embed_query, text)
        # フォールバック: embed_documents に単一要素リストで委譲
        vecs = await asyncio.to_thread(self._emb.embed_documents, [text])
        return vecs[0] if vecs else []

    async def aembed_text(self, text: str):
        """非同期版。ragas 側が await しても安全。"""
        # embed_text 自体が async なのでそのまま await する
        return await self.embed_text(text)

    def embed_documents(self, texts):
        # ベクトルストア（PGVector）が同期メソッドを期待するため同期で提供
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str):
        # ベクトルストア（PGVector）が同期メソッドを期待するため同期で提供
        try:
            if isinstance(self._model_name, str) and "jina-embeddings-v4" in self._model_name:
                client = getattr(self._emb, "client", None)
                if client is not None and hasattr(client, "encode"):
                    vec = client.encode(
                        sentences=[text],
                        task="retrieval",
                        prompt_name="query",
                        normalize_embeddings=True,
                    )[0]
                    # SentenceTransformerはnumpyを返すことがあるためlist化
                    return vec.tolist() if hasattr(vec, "tolist") else vec
        except Exception:
            # 失敗時はフォールバック
            pass
        return self._emb.embed_query(text)

    async def aembed_documents(self, texts):
        import asyncio
        # 同期版 embed_documents に委譲（上のv4専用分岐を再利用）
        return await asyncio.to_thread(self.embed_documents, texts)

    def __getattr__(self, name):
        return getattr(self._emb, name)


class RAGASCompatibleHuggingFaceEmbeddings:
    """
    RAGASが embeddings.set_run_config を呼んでもエラーにしないための薄いラッパー。
    それ以外の属性・メソッドは元のHuggingFaceEmbeddingsに委譲する。
    """
    def __init__(self, model_name: str, **kwargs):
        # 元のHuggingFaceEmbeddingsを内部に保持
        # kwargsにはdeviceやencode_kwargsなどが含まれる
        self._model_name = model_name
        self._emb = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        # 一度だけログを出すためのフラグ（スパム防止）
        self._log_once = {"doc": False, "qry": False, "doc_fallback": False, "qry_fallback": False}

    def set_run_config(self, config):
        # RAGAS互換のためのno-op
        pass

    async def embed_text(self, text: str):
        """RAGASがawaitしても安全な単一テキスト埋め込みAPI（async）。"""
        import asyncio
        if hasattr(self._emb, "embed_query"):
            return await asyncio.to_thread(self._emb.embed_query, text)
        vecs = await asyncio.to_thread(self._emb.embed_documents, [text])
        return vecs[0] if vecs else []

    async def aembed_text(self, text: str):
        # embed_text 自体が async なのでそのまま await する
        return await self.embed_text(text)

    def embed_documents(self, texts):
        # jina-embeddings-v4 は Retrieval タスクで passage/query のプロンプトを切替える必要がある
        try:
            if isinstance(self._model_name, str) and "jina-embeddings-v4" in self._model_name:
                client = getattr(self._emb, "_client", None) or getattr(self._emb, "client", None)
                if client is not None and hasattr(client, "encode"):
                    # Passage 用のプロンプトを指定（正規化も実施）
                    vecs = client.encode(
                        sentences=texts,
                        task="retrieval",
                        prompt_name="passage",
                        normalize_embeddings=True,
                    )
                    if not self._log_once["doc"]:
                        print("[emb] Jina v4 encode(passages) path used: task=retrieval, prompt_name=passage, normalize_embeddings=True")
                        self._log_once["doc"] = True
                    return vecs.tolist() if hasattr(vecs, "tolist") else vecs
        except Exception:
            # 失敗時はフォールバック
            pass
        if not self._log_once["doc_fallback"]:
            print("[emb] embed_documents fallback path used (HuggingFaceEmbeddings)")
            self._log_once["doc_fallback"] = True
        return self._emb.embed_documents(texts)

    async def aembed_documents(self, texts):
        import asyncio
        return await asyncio.to_thread(self._emb.embed_documents, texts)

    def embed_query(self, text: str):
        # ベクトルストア（PGVector）が同期メソッドを期待するため同期で提供
        # Jina v4 ではクエリ用プロンプトとタスクを明示する
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
                        print("[emb] Jina v4 encode(query) path used: task=retrieval, prompt_name=query, normalize_embeddings=True")
                        self._log_once["qry"] = True
                    return vec.tolist() if hasattr(vec, "tolist") else vec
        except Exception:
            # 失敗時はフォールバック
            pass
        if not self._log_once["qry_fallback"]:
            print("[emb] embed_query fallback path used (HuggingFaceEmbeddings)")
            self._log_once["qry_fallback"] = True
        return self._emb.embed_query(text)

    async def aembed_query(self, text: str):
        import asyncio
        # 同期版 embed_query に委譲（上のv4専用分岐を再利用）
        return await asyncio.to_thread(self.embed_query, text)

    def __getattr__(self, name):
        # それ以外の属性アクセスは内部の実体に委譲
        return getattr(self._emb, name)


class RAGASCompatibleOpenAIEmbeddings:
    """
    RAGASが set_run_config を呼んでもエラーにしないための薄いラッパー。
    OpenAIEmbeddings のインスタンスを内包し、必要メソッドを委譲する。
    """
    def __init__(self, **kwargs):
        # OpenAIEmbeddings は model / api_key などを kwargs で受け取る
        self._emb = OpenAIEmbeddings(**kwargs)

    def set_run_config(self, config):
        # RAGAS互換のためのno-op
        pass

    async def embed_text(self, text: str):
        # 非同期で単一テキストの埋め込み（RAGAS側がawaitしても安全）
        import asyncio
        if hasattr(self._emb, "embed_query"):
            return await asyncio.to_thread(self._emb.embed_query, text)
        vecs = await asyncio.to_thread(self._emb.embed_documents, [text])
        return vecs[0] if vecs else []

    async def aembed_text(self, text: str):
        return await self.embed_text(text)

    def embed_documents(self, texts):
        # PGVector 等が同期メソッドを期待するため同期で提供
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str):
        # PGVector 等が同期メソッドを期待するため同期で提供
        return self._emb.embed_query(text)

    async def aembed_documents(self, texts):
        import asyncio
        return await asyncio.to_thread(self._emb.embed_documents, texts)

    async def aembed_query(self, text: str):
        import asyncio
        return await asyncio.to_thread(self._emb.embed_query, text)

    def __getattr__(self, name):
        return getattr(self._emb, name)


# データベース接続設定
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/rag_db")
os.environ["PGVECTOR_CONNECTION_STRING"] = DB_URL

# --- コレクション名をモデルごとに動的生成する関数 ---
def get_collection_name(model_name: str) -> str:
    """
    モデル名ごとにコレクション名を動的に決定する関数。
    例: model_name='openai' → 'rag_collection_openai'
    """
    return f"rag_collection_{model_name}"

# --- Model Selection ---
def get_llm(model_name: str):
    """
    内部LLMは常に GPT-OSS に固定します（gpt-oss:20b on Ollama）。
    引数 model_name は互換性維持のため受け取りますが、無視されます。
    """
    # Ollamaの接続先は環境変数から取得（未指定時はDocker内サービス名を既定値とする）
    # 例: ホスト上のOllamaを使う場合は OLLAMA_BASE_URL=http://host.docker.internal:11434
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    try:
        if model_name not in (None, "gpt-oss", "gpt-oss:20b"):
            print(f"[INFO] get_llm: 要求モデル '{model_name}' を無視し、GPT-OSS(gpt-oss:20b)を使用します")
    except Exception:
        # 念のため例外は握りつぶす（ログ用途のみのため）
        pass
    return RAGASCompatibleOllamaLLM(model="gpt-oss:20b", base_url=ollama_base_url)


# 利用可能なデバイスを自動判定（Apple Siliconならmps, NVIDIAならcuda, どちらもなければcpu）
def get_torch_device():
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"  # Apple Silicon(M1/M2/M3/M4)のMetalアクセラレーション
        elif torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"
    print(f"[INFO] get_torch_device: 利用デバイス = {device}")  # ログ出力
    return device


def get_embeddings(model_name: str):
    device = get_torch_device()  # デバイス自動判定
    common_kwargs = {
        'model_kwargs': {
            'device': device,
            'trust_remote_code': True
        },
        'encode_kwargs': {
            'normalize_embeddings': True
        }
    }
    
    # Ollama埋め込みモデル（優先使用）
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    ollama_embedding_models = {
        "nomic-embed-text": "nomic-embed-text",
        "mxbai-embed-large": "mxbai-embed-large",
        "all-minilm": "all-minilm",
        # 日本語/多言語対応のモデル（Ollama）
        # 事前に `ollama pull bge-m3` / `ollama pull jina-embeddings-v3` を実行しておくこと
        "bge-m3": "bge-m3",
        "jina-embeddings-v3": "jina-embeddings-v3",
    }
    if model_name in ollama_embedding_models:
        # RAGAS互換の薄いラッパーで包む（set_run_config 要求に対応）
        return RAGASCompatibleOllamaEmbeddings(
            model=ollama_embedding_models[model_name],
            base_url=ollama_base_url,
        )

    # OpenAIモデルのマッピング
    openai_models = {
        "gpt-4o": "text-embedding-ada-002",  # 旧モデル名との互換性のため
        "text-embedding-3-small": "text-embedding-3-small",
        "text-embedding-3-large": "text-embedding-3-large",
        "text-embedding-ada-002": "text-embedding-ada-002"
    }
    
    if model_name in openai_models:
        return RAGASCompatibleOpenAIEmbeddings(
            model=openai_models[model_name],
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # HuggingFaceモデルのマッピング
    hf_models = {
        "huggingface_bge_small": "BAAI/bge-small-en-v1.5",
        "huggingface_bge_large": "BAAI/bge-large-en-v1.5",
        "huggingface_miniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "huggingface_mpnet_base": "sentence-transformers/all-mpnet-base-v2",
        "huggingface_multi_qa_minilm": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "huggingface_multi_qa_mpnet": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "huggingface_paraphrase_multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "huggingface_distiluse_multilingual": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "huggingface_xlm_r": "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
        # --- Jina Embeddings v4 (HuggingFace) ---
        # 多言語・マルチモーダル埋め込み。text-only用途ではRetrieval(task)を使用。
        "jina-embeddings-v4": "jinaai/jina-embeddings-v4",
    }
    
    if model_name in hf_models:
        # Jina Embeddings v4 はエンコード前にタスク指定が必須
        # 参考: エラーメッセージ "Task must be specified before encoding data..."
        if model_name == "jina-embeddings-v4":
            # 初期化時に default_task は渡さず、エンコード時に task を指定する
            return RAGASCompatibleHuggingFaceEmbeddings(
                model_name=hf_models[model_name],
                model_kwargs={
                    'device': device,
                    'trust_remote_code': True,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'task': 'retrieval',
                }
            )
        # それ以外は共通設定を適用
        return RAGASCompatibleHuggingFaceEmbeddings(
            model_name=hf_models[model_name],
            **common_kwargs
        )
    
    raise ValueError(f"Unsupported embedding model: {model_name}")


# Default models（モデルが未ダウンロードでもサーバーが起動できるように修正）
current_llm = None
current_embeddings = None

# 内部LLMはGPT-OSS固定
try:
    current_llm = get_llm("gpt-oss")
except Exception as e:
    import logging
    logging.warning(f"LLM初期化失敗 (gpt-oss): {e}")
    current_llm = None

try:
    # デフォルトでHuggingFaceの軽量モデルを使用
    current_embeddings = get_embeddings("huggingface_bge_small")
    if current_embeddings is None:
        raise ValueError("Failed to initialize huggingface_bge_small")
    logging.info("Successfully initialized HuggingFace BGE Small model")
except Exception as e:
    import logging
    logging.error(f"Embedding初期化失敗: {e}")
    try:
        # 代替モデルを試す
        current_embeddings = get_embeddings("huggingface_miniLM")
        logging.info("Falling back to HuggingFace MiniLM model")
    except Exception as e2:
        logging.error(f"代替Embeddingモデルの初期化にも失敗: {e2}")
        current_embeddings = None

# --- Pydantic Models ---
class ChunkRequest(BaseModel):
    text: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_method: str = 'recursive'  # 'recursive' or 'semantic'
    embedding_model: str = None  # Required for semantic chunking

class EmbedRequest(BaseModel):
    chunks: list[str]
    embedding_model: str # 埋め込みモデル名
    chunk_method: str    # チャンク方式（recursive, semantic, fixed, sentence, paragraph など）

class QueryRequest(BaseModel):
    query: str
    llm_model: str = "mistral"  # デフォルト値を設定
    embedding_model: str = "huggingface_bge_small"  # デフォルト値を設定

# 単一評価リクエストは一括評価に統合されました

class ModelSelection(BaseModel):
    llm_model: str
    embedding_model: str

# --- API Endpoints ---


@app.post("/chunk/")
def chunk_text(request: ChunkRequest):
    """
    chunk_methodに応じて適切な方法でテキストをチャンク分割
    - recursive: 再帰的にテキストを分割（デフォルト）
    - fixed: 固定長で分割
    - semantic: 意味的なまとまりで分割（embeddingモデルが必要）
    - sentence: 文単位で分割
    - paragraph: 段落単位で分割
    """
    if request.chunk_method == 'semantic':
        # embedding_modelが指定されていることを確認
        if not request.embedding_model:
            raise HTTPException(
                status_code=400,
                detail="semanticチャンキングにはembedding_modelの指定が必要です"
            )
        try:
            # モデル名から埋め込みインスタンスを生成
            embedder = get_embeddings(request.embedding_model)
            chunks = semantic_chunk_text(
                text=request.text,
                chunk_size=None,
                chunk_overlap=None,
                embedding_model=embedder  # インスタンスを渡す
            )
            return {"chunks": chunks}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"テキストのチャンキング中にエラーが発生しました: {str(e)}"
            )
    elif request.chunk_method == 'recursive':
        # 再帰的な文字数分割
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(request.text)
        return {"chunks": chunks}
    elif request.chunk_method == 'fixed':
        # 固定長で分割
        chunks = fixed_chunk_text(request.text, request.chunk_size, request.chunk_overlap)
        return {"chunks": chunks}
    elif request.chunk_method == 'sentence':
        # 文単位で分割
        chunks = sentence_chunk_text(request.text)
        return {"chunks": chunks}
    elif request.chunk_method == 'paragraph':
        # 段落単位で分割
        chunks = paragraph_chunk_text(request.text)
        return {"chunks": chunks}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"未対応のchunk_method: {request.chunk_method}。'recursive', 'fixed', 'semantic', 'sentence', 'paragraph' のいずれかを指定してください。"
        )


@app.post("/embed_and_store/")
def embed_and_store(request: EmbedRequest):
    try:
        embeddings_instance = get_embeddings(request.embedding_model)
        vectorstore = PGVector.from_documents(
            documents=[],  # 空のドキュメントで初期化
            embedding=embeddings_instance,
            collection_name=get_collection_name(request.embedding_model)  # embeddingモデルごとにコレクションを切り替え
        )
        # chunk_methodを全チャンクのmetadataに付与して保存
        chunk_method = getattr(request, 'chunk_method', None)
        # chunk_methodがEmbedRequestにない場合は、各チャンクのメタ情報としてNoneになる
        metadatas = [{"chunk_method": chunk_method} for _ in request.chunks]
        vectorstore.add_texts(texts=request.chunks, metadatas=metadatas)
        return {"message": f"Successfully embedded and stored {len(request.chunks)} chunks using {request.embedding_model} (method={chunk_method}) ."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
def query_rag(request: QueryRequest):
    try:
        # 利用可能なモデルリストを定義
        # 利用可能なモデルリストを拡張（OpenAI系も含める）
        available_llm_models = [
            "ollama_llama2", "gpt-4o", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
        ]
        # llm_modelが未サポートの場合は自動で置き換え
        llm_model = request.llm_model
        if llm_model not in available_llm_models:
            llm_model = available_llm_models[0]  # ollama_llama2優先
        llm_instance = get_llm(llm_model)
        embeddings_instance = get_embeddings(request.embedding_model)

        # データベースからテキストを取得
        with SessionLocal() as session:
            result = session.execute(
                text("SELECT text FROM embeddings WHERE chunk_strategy = 'test'")
            )
            texts = [row[0] for row in result.fetchall()]
            
            if not texts:
                # 文書が存在しない場合は通常のLLM応答のみを返す
                # OpenAI系モデルの応答が辞書型の場合はcontent部分だけ抽出
                ai_response = llm_instance.invoke(request.query)
                if isinstance(ai_response, dict) and "content" in ai_response:
                    answer = ai_response["content"]
                else:
                    answer = str(ai_response)
                return {
                    "answer": answer,
                    "contexts": [],
                    "source_documents": []
                }

            # ベクトルストアを初期化
            collection_name = get_collection_name(request.embedding_model)
            connection_string = "postgresql://rag_user:rag_password@db:5432/rag_db"
            
            # 既存のコレクションを削除
            try:
                session.execute(text(f"DROP TABLE IF EXISTS {collection_name} CASCADE"))
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"Warning: Failed to drop collection {collection_name}: {str(e)}")
            
            # 新しいコレクションを作成
            vectorstore = PGVector(
                embedding_function=embeddings_instance,
                collection_name=collection_name,
                connection_string=connection_string,
                use_jsonb=True
            )
            
            # チャンクを初期化（データベースから取得したテキストをそのまま使用）
            chunks = texts
            
            # テキストを追加
            vectorstore.add_texts(texts=texts)
            
            # リトリーバーを作成
            retriever = vectorstore.as_retriever()

            # プロンプトテンプレート
            template = """以下の文脈に基づいて質問に答えてください。

文脈:
{context}

質問: {question}"""
            prompt = ChatPromptTemplate.from_template(template)

            # チェーンを作成
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm_instance
                | StrOutputParser()
            )

            # 質問に回答
            answer = chain.invoke(request.query)
            
            # 関連するドキュメントを取得
            retrieved_docs = retriever.get_relevant_documents(request.query)
            contexts = [doc.page_content for doc in retrieved_docs]
            
            # 結果を返却
            return {
                "answer": answer, 
                "contexts": contexts,
                "source_documents": [{"page_content": doc.page_content} for doc in retrieved_docs]
            }
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in query_rag: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"エラーが発生しました: {str(e)}\n{error_trace}"
        )

def calculate_overlap_metrics(contexts: list[list[str]], embedder=None) -> dict:
    """複数のオーバーラップメトリクスを計算する
    
    Args:
        contexts: コンテキストのリスト
        embedder: オプションの埋め込みモデル（セマンティックオーバーラップ用）
    
    Returns:
        dict: 各種オーバーラップメトリクスを含む辞書
    """
    if not contexts or len(contexts) < 2:
        return {
            "overlap_ratio": 0.0,
            "adjacent_overlap": [0.0],
            "semantic_overlap": 0.0
        }
    
    # 1. 元のオーバーラップ計算（後方互換性のため保持）
    all_tokens = []
    for ctx in contexts:
        if isinstance(ctx, str):
            all_tokens.extend(ctx.split())
        else:
            for text in ctx:
                all_tokens.extend(text.split())
    
    unique_tokens = set(all_tokens)
    total_tokens = len(all_tokens)
    unique_count = len(unique_tokens)
    
    overlap_ratio = 1.0 - (unique_count / total_tokens) if total_tokens > 0 else 0.0
    
    # 2. 隣接チャンク間のオーバーラップ
    adjacent_overlaps = []
    for i in range(len(contexts) - 1):
        # 現在のチャンクと次のチャンクのトークンを取得
        current_ctx = contexts[i] if isinstance(contexts[i], list) else [contexts[i]]
        next_ctx = contexts[i+1] if isinstance(contexts[i+1], list) else [contexts[i+1]]
        
        current_tokens = set(' '.join(current_ctx).split())
        next_tokens = set(' '.join(next_ctx).split())
        
        # 共通トークン数を計算
        common_tokens = current_tokens.intersection(next_tokens)
        min_len = min(len(current_tokens), len(next_tokens))
        
        # オーバーラップ率を計算
        overlap = len(common_tokens) / min_len if min_len > 0 else 0.0
        adjacent_overlaps.append(overlap)
    
    # 3. セマンティックオーバーラップ（埋め込みモデルが利用可能な場合）
    semantic_overlap = 0.0
    if embedder and len(contexts) > 1:
        try:
            # 各チャンクを1つの文字列に結合
            chunk_texts = [' '.join(ctx) if isinstance(ctx, list) else ctx for ctx in contexts]
            
            # 埋め込みを取得
            embeddings = embedder.embed_documents(chunk_texts)
            
            # 隣接チャンク間の類似度を計算
            similarities = []
            for i in range(len(embeddings) - 1):
                # コサイン類似度を計算
                sim = cosine_similarity(
                    [embeddings[i]], 
                    [embeddings[i+1]]
                )[0][0]
                similarities.append(sim)
            
            semantic_overlap = sum(similarities) / len(similarities) if similarities else 0.0
        except Exception as e:
            print(f"セマンティックオーバーラップの計算中にエラーが発生しました: {str(e)}")
            semantic_overlap = 0.0
    
    return {
        "overlap_ratio": overlap_ratio,
        "adjacent_overlap": adjacent_overlaps,
        "avg_adjacent_overlap": sum(adjacent_overlaps) / len(adjacent_overlaps) if adjacent_overlaps else 0.0,
        "semantic_overlap": semantic_overlap
    }

# /evaluate/エンドポイントは一括評価に統合されました
# 代わりに/bulk_evaluate/エンドポイントを使用してください

@app.post("/clear_db/")
def clear_db():
    """
    すべてのembeddingモデルのコレクション（DBデータ）を完全削除するAPI。
    主要embeddingモデル（huggingface_bge_small, openai等）すべてをループで削除。
    """
    try:
        if not LOCAL_MODEL_PATH.exists():
            return {
                "status": "error",
                "message": f"モデルが見つかりません: {LOCAL_MODEL_PATH}。DBリセット不可。",
                "model_exists": False
            }
        # 削除対象embeddingモデルリスト
        embedding_models = ["huggingface_bge_small", "gpt-4o"]
        results = []
        for emb_model in embedding_models:
            try:
                if emb_model == "huggingface_bge_small":
                    dummy_embeddings = HuggingFaceEmbeddings(
                        model_name=str(LOCAL_MODEL_PATH),
                        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                elif emb_model == "gpt-4o":
                    from langchain_openai import OpenAIEmbeddings
                    dummy_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
                else:
                    continue
                vectorstore = PGVector.from_documents(
                    documents=[],  # 空のドキュメントで初期化
                    embedding=dummy_embeddings,
                    collection_name=get_collection_name(emb_model)
                )
                vectorstore.delete_collection()
                results.append(f"{emb_model}: 削除成功")
            except Exception as e:
                results.append(f"{emb_model}: 削除失敗 ({str(e)})")
        return {
            "status": "success",
            "message": "全embeddingモデルのコレクションを削除しました。",
            "details": results,
            "model_exists": True
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"DB全体削除時エラー: {str(e)}",
            "model_exists": LOCAL_MODEL_PATH.exists()
        }

@app.get("/models/")
def get_available_models():
    """
    利用可能なモデルと現在のモデル状態を返します。
    """
    model_exists = LOCAL_MODEL_PATH.exists()
    model_info = {
        "model_name": str(MODEL_NAME),
        "local_path": str(LOCAL_MODEL_PATH),
        "exists": model_exists,
        "size_mb": (
            sum(f.stat().st_size for f in LOCAL_MODEL_PATH.glob('**/*') if f.is_file()) / (1024 * 1024)
        ) if model_exists else 0
    }
    
    return {
        "llm_models": ["ollama_llama2", "gpt-4o"],
        "embedding_models": ["huggingface_bge_small", "gpt-4o"],
        "current_embedding_model": {
            "name": "huggingface_bge_small",
            "type": "local" if model_exists else "remote",
            "info": model_info
        },
        "environment": {
            "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", "Not set"),
            "hf_home": os.environ.get("HF_HOME", "Not set")
        }
    }

# --- 一括評価API（ダミー実装） ---
from fastapi.responses import JSONResponse
from fastapi import Request
import asyncio

@app.post("/bulk_evaluate/")
async def bulk_evaluate(request: Request):
    """
    embeddingモデル・チャンク分割パラメータを受けてRAG自動評価を行うAPI。
    Eval.mdの方針に従い、faithfulness等の指標でスコア返却。
    """
    try:
        # --- 数値のNaN/infガード用ユーティリティ ---
        import math
        def safe_val(x):
            try:
                if math.isnan(x) or math.isinf(x):
                    return 0.0
                return float(x)
            except Exception:
                return 0.0

        data = await request.json()
        # --- dataがリスト型なら各要素ごとに個別評価 ---
        def find_first_dict(obj):
            if isinstance(obj, dict):
                return obj
            elif isinstance(obj, list):
                for item in obj:
                    found = find_first_dict(item)
                    if isinstance(found, dict):
                        return found
                return {}

        # 並列処理の最大数を制限するセマフォを作成
        MAX_PARALLEL_TASKS = int(os.getenv("EVAL_MAX_PARALLEL_TASKS", "2"))  # 既定を2に調整（安定性優先）
        semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
        # 計測ログの有効化（環境変数でON/OFF）
        TIMING_LOG = os.getenv("EVAL_TIMING_LOG", "0").lower() in {"1", "true", "yes"}
        import time
        def _tnow():
            return time.monotonic()
        def _tlog(label: str, start: float = None):
            # EVAL_TIMING_LOG が有効なときのみ出力
            if not TIMING_LOG:
                return
            try:
                if start is None:
                    print(f"[timing] {label} at {jst_now_str()}")
                else:
                    dur = _tnow() - start
                    print(f"[timing] {label} took {dur:.3f}s")
            except Exception:
                pass

        async def evaluate_one_bulk(data):
            try:
                print("[進捗] 評価データを処理中...")
                # タイムアウト設定（環境変数で調整可能）
                # 既定値を延長: LLM呼び出し=60秒, 評価全体=600秒
                def _parse_timeout_env(key: str, default_seconds: int):
                    val = os.getenv(key, str(default_seconds))
                    if val is None:
                        return default_seconds
                    v = str(val).strip().lower()
                    if v in ("none", "no", "off", "false", "0", "-1"):
                        return None  # 無制限
                    try:
                        return int(v)
                    except Exception:
                        return default_seconds
                LLM_TIMEOUT = _parse_timeout_env("EVAL_LLM_TIMEOUT_SECONDS", 60)
                EVAL_TIMEOUT = _parse_timeout_env("RAGAS_EVAL_TIMEOUT_SECONDS", 600)
                def _fmt(t):
                    return "no-timeout" if t is None else f"{t}s"
                print(f"[設定] TIMEOUT: LLM_TIMEOUT={_fmt(LLM_TIMEOUT)}, EVAL_TIMEOUT={_fmt(EVAL_TIMEOUT)}, MAX_PARALLEL_TASKS={MAX_PARALLEL_TASKS}")
                embedding_model = data.get("embedding_model")
                chunk_methods = data.get("chunk_methods", [data.get("chunk_method", "recursive")])
                chunk_sizes = data.get("chunk_sizes", [data.get("chunk_size", 1000)])
                chunk_overlaps = data.get("chunk_overlaps", [data.get("chunk_overlap", 0)])
                # ユーザー指定のLLMモデルは無視し、内部では常にGPT-OSSを使用
                _requested_llm_model = data.get("llm_model", None)
                llm_model = "gpt-oss"
                print(f"[設定] LLMモデルはGPT-OSS固定です（リクエスト指定: {_requested_llm_model} → 使用: {llm_model}）")
                
                # セマンティックチャンキングが選択されている場合の情報メッセージ
                if "semantic" in chunk_methods:
                    if len(chunk_methods) == 1:
                        print("情報: セマンティックチャンキングが選択されました。チャンクサイズとオーバーラップは使用されません。")
                    else:
                        print(f"情報: セマンティックチャンキングとその他のチャンキング方式が同時に選択されています。")
                        print(f"      セマンティックチャンキング: デフォルトパラメータを使用")
                        print(f"      その他の方式: 指定されたチャンクサイズとオーバーラップを使用")

                # 必須パラメータチェック
                sample_text = data.get("text")
                if not sample_text:
                    raise ValueError("textが指定されていません")
                    
                # サポートされているモデルかチェック
                supported_models = {
                    # OpenAIモデル
                    'openai', 'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002',
                    # HuggingFaceモデル
                    'huggingface_bge_small', 'huggingface_bge_large', 'huggingface_miniLM', 'huggingface_mpnet_base',
                    'huggingface_multi_qa_minilm', 'huggingface_multi_qa_mpnet',
                    'huggingface_paraphrase_multilingual', 'huggingface_distiluse_multilingual',
                    'huggingface_xlm_r', 'jina-embeddings-v4',
                    # Ollama埋め込みモデル
                    'nomic-embed-text', 'mxbai-embed-large', 'all-minilm', 'bge-m3', 'jina-embeddings-v3',
                }
                
                if embedding_model not in supported_models:
                    raise ValueError(f"未サポートの埋め込みモデルが指定されました: {embedding_model}")
                
                # OpenAI埋め込みの旧指定に対する注意喚起
                if embedding_model == "openai":
                    print("警告: 'openai' は包括的な指定です。具体的な 'text-embedding-3-small' または 'text-embedding-3-large' を選択してください。")

                questions = data.get("questions")
                # ground_truthキーまたはanswersキーのどちらかを使用（互換性のため）
                answers = data.get("ground_truth", data.get("answers"))
                if not questions or not answers:
                    raise ValueError("questions/answersが指定されていません。PDFアップロード時の自動生成結果をそのまま送信してください。")
                if not (sample_text and questions and answers):
                    raise ValueError("PDFアップロードとQA自動生成を先に実施してください（text, questions, answers必須）。")

                results = []
                # embedding_modelのインスタンスを一度だけロードし再利用
                print(f"[進捗] 埋め込みモデル '{embedding_model}' をロード中...")
                _t0_embed = _tnow()
                embedder = get_embeddings(embedding_model)
                _tlog("embedder.load", _t0_embed)
                
                # chunk_method/chunk_size/chunk_overlapごとに完全に独立してチャンク分割→ベクトルストア→retriever→RAG回答生成→評価→スコア集計を実行
                for i in range(len(chunk_methods)):
                    try:
                        chunk_method = chunk_methods[i]
                        print(f"[進捗] チャンク方法 '{chunk_method}' の処理を開始...")
                        
                        # セマンティックチャンキングの場合、チャンクサイズとオーバーラップは無視する
                        if chunk_method == "semantic":
                            if not embedding_model:
                                results.append({
                                    "error": "セマンティックチャンキングにはembedding_modelの指定が必須です", 
                                    "chunk_method": chunk_method
                                })
                                continue
                                
                            print(f"[進捗] セマンティックチャンキングを開始します（chunk_sizeとchunk_overlapは無視されます）...")
                            _t0_chunk = _tnow()
                            
                            # セマンティックチャンキングのパラメータを取得
                            semantic_params = data.get("semantic_params", {})
                            similarity_threshold = float(semantic_params.get("similarity_threshold", 0.7))
                            
                            print(f"[進捗] セマンティックチャンキングを実行: similarity_threshold={similarity_threshold}")
                            # セマンティックチャンキングを非同期実行に変更
                            chunks = await asyncio.to_thread(
                                semantic_chunk_text,
                                text=sample_text,
                                chunk_size=None,  # 無視される
                                chunk_overlap=None,  # 無視される
                                embedding_model=embedder,
                                similarity_threshold=similarity_threshold
                            )
                            _tlog(f"chunking.semantic", _t0_chunk)
                            
                            # セマンティックチャンキングの場合はchunk_sizeとchunk_overlapをNoneに設定
                            chunk_size_val = None
                            chunk_overlap_val = None
                            chunk_strategy = "semantic"
                        else:
                            # 通常のチャンキング方法の場合
                            chunk_size = chunk_sizes[i] if i < len(chunk_sizes) else 1000
                            chunk_overlap = chunk_overlaps[i] if i < len(chunk_overlaps) else 200
                            # チャンク分割
                            print(f"[進捗] チャンク分割を実行: 方式={chunk_method}, サイズ={chunk_size}, オーバーラップ={chunk_overlap}")
                            _t0_chunk = _tnow()
                            
                            # 非同期でチャンク分割を実行
                            if chunk_method == "recursive":
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    length_function=len,
                                )
                                chunks = await asyncio.to_thread(text_splitter.split_text, sample_text)
                            elif chunk_method == "fixed":
                                chunks = await asyncio.to_thread(
                                    fixed_chunk_text,
                                    sample_text, 
                                    chunk_size=chunk_size, 
                                    chunk_overlap=chunk_overlap
                                )
                            elif chunk_method == "sentence":
                                chunks = await asyncio.to_thread(sentence_chunk_text, sample_text)
                            elif chunk_method == "paragraph":
                                chunks = await asyncio.to_thread(paragraph_chunk_text, sample_text)
                            # semanticチャンキングは上記のif文で既に処理済み
                            else:
                                raise ValueError(f"未対応のchunk_method: {chunk_method}")
                            _tlog(f"chunking.{chunk_method}", _t0_chunk)
                            
                            # チャンク戦略を設定
                            chunk_size_val = chunk_sizes[i] if i < len(chunk_sizes) else chunk_sizes[0]
                            chunk_overlap_val = chunk_overlaps[i] if i < len(chunk_overlaps) else chunk_overlaps[0]
                            chunk_strategies = data.get("chunk_strategies", []) if isinstance(data, dict) else []
                            if chunk_strategies and i < len(chunk_strategies):
                                chunk_strategy = chunk_strategies[i]
                            else:
                                chunk_strategy = f"{chunk_method}-{chunk_size_val}-{chunk_overlap_val}"

                        print(f"[進捗] {len(chunks)}個のチャンクを作成しました。平均長さ: {sum(len(c) for c in chunks) / max(len(chunks), 1):.1f}文字")
                        print(f"[進捗] ベクトルストアを構築中...")
                        _t0_vs = _tnow()
                        
                        # ベクトルストア構築
                        vectorstore = PGVector.from_documents(
                            documents=[],  # 空で初期化
                            embedding=embedder,
                            collection_name=get_collection_name(embedding_model)
                        )
                        # チャンクをベクトルストアに追加（大量の場合はバッチ処理）
                        await asyncio.to_thread(vectorstore.add_texts, texts=chunks)
                        _tlog("vectorstore.build+add", _t0_vs)
                        # 検索パラメータの受け口（既定は従来と互換）
                        top_k = int(data.get("top_k", 5))
                        use_mmr = bool(data.get("use_mmr", False))
                        fetch_k = int(data.get("fetch_k", max(top_k * 2, 20)))
                        try:
                            lambda_mult = float(data.get("lambda_mult", 0.5))
                        except Exception:
                            lambda_mult = 0.5
                        if use_mmr:
                            print(f"[設定] retriever=MMR k={top_k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
                            retriever = vectorstore.as_retriever(
                                search_type="mmr",
                                search_kwargs={"k": top_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
                            )
                        else:
                            print(f"[設定] retriever=similarity k={top_k}")
                            retriever = vectorstore.as_retriever(
                                search_kwargs={"k": top_k},
                            )

                        # RAG回答生成＆コンテキスト取得
                        contexts = []
                        pred_answers = []
                        
                        # PDFアップロード時の回答が揃っていれば使い回し（高速化）
                        if answers and len(answers) == len(questions):
                            print(f"[進捗] PDFアップロード時の回答を使用（{len(answers)}個の回答）")
                            pred_answers = answers  # 回答は使い回し
                            
                            async def get_context_only(q):
                                async with semaphore:
                                    retrieved_docs = await asyncio.to_thread(retriever.get_relevant_documents, q)
                                    return [doc.page_content for doc in retrieved_docs]
                            # 全質問のコンテキストのみ取得
                            _t0_ctx = _tnow()
                            contexts = await asyncio.gather(*[get_context_only(q) for q in questions])
                            _tlog("retrieval.contexts_only", _t0_ctx)
                            print(f"[進捗] コンテキスト取得完了。評価処理を開始...")
                        else:
                            print(f"[進捗] 新しいRAG回答を生成（{len(questions)}個の質問）...")
                            
                            # 各質問に対して非同期でコンテキスト取得と回答生成を行う
                            async def get_context_and_answer(q):
                                async with semaphore:  # セマフォで並列処理数を制限
                                    # 各質問ごとにリトリーバーで文脈取得（非同期化）
                                    retrieved_docs = await asyncio.to_thread(retriever.get_relevant_documents, q)
                                    context_texts = [doc.page_content for doc in retrieved_docs]
                                    # LLMインスタンス・プロンプト生成（GPT-OSS固定）
                                    llm_instance = get_llm("gpt-oss")
                                    prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:\n{context}\n\nQuestion: {question}""")
                                    # Ollama ラッパーはLCELのRunnableではないため、RunnableLambdaで委譲して対応
                                    def _to_text(x):
                                        try:
                                            return x.to_string()
                                        except Exception:
                                            return x
                                    # ラッパーのinvokeを確実に通す（型正規化のため）
                                    llm_runnable = RunnableLambda(lambda x: llm_instance.invoke(_to_text(x)))
                                    chain = (
                                        {"context": lambda _: context_texts, "question": lambda _: q}
                                        | prompt
                                        | llm_runnable
                                        | StrOutputParser()
                                    )
                                    # 非同期で回答生成（タイムアウト付与）
                                    try:
                                        # 最初のマッピングでcontext/questionを供給するため空dictで十分
                                        if LLM_TIMEOUT is None:
                                            answer = await chain.ainvoke({})
                                        else:
                                            answer = await asyncio.wait_for(chain.ainvoke({}), timeout=LLM_TIMEOUT)
                                    except asyncio.TimeoutError:
                                        print(f"[警告] LLM回答生成がタイムアウト: model={llm_model}, timeout={LLM_TIMEOUT}s, question={q[:30]}...")
                                        answer = "[LLMタイムアウト]"
                                    except Exception as e:
                                        print(f"[警告] LLM回答生成失敗: {e}")
                                        answer = "[LLMエラー]"
                                    return context_texts, answer
                            
                            # 非同期で全質問の回答を生成
                            _t0_rag = _tnow()
                            results_list = await asyncio.gather(*[get_context_and_answer(q) for q in questions])
                            _tlog("retrieval+llm_answers", _t0_rag)
                            for context_texts, answer in results_list:
                                contexts.append(context_texts)
                                pred_answers.append(answer)
                            print(f"[進捗] RAG回答生成完了。評価処理を開始...")
                        # --- ここまで並列化 ---

                        # RAGAS等で自動評価
                        print(f"[進捗] 評価メトリクスの計算を開始...")
                        
                        dataset_dict = {
                            "question": questions,
                            "answer": pred_answers,
                            "contexts": contexts,
                            "ground_truth": answers
                        }
                        # 必須カラム 'reference' を追加（answer_correctness 用）
                        dataset_dict_with_ref = dict(dataset_dict)
                        dataset_dict_with_ref["reference"] = answers
                        dataset = Dataset.from_dict(dataset_dict_with_ref)
                        # 評価用LLMもGPT-OSS固定
                        llm_instance_eval = get_llm("gpt-oss")
                        # RAGAS が await するケースに対応する非同期アダプタ
                        ragas_llm = RAGASLLMAsyncAdapter(llm_instance_eval)
                        
                        # --- 全質問を1つのDatasetにまとめて一括評価（ragas側でmax_workers並列化） ---
                        import copy as _copy
                        base_metrics = [faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness, answer_similarity]
                        metrics_local = [_copy.deepcopy(m) for m in base_metrics]
                        for m in metrics_local:
                            if hasattr(m, "llm"):
                                m.llm = ragas_llm
                            if hasattr(m, "embeddings"):
                                m.embeddings = embedder
                        # ragas.evaluate は同期関数のため、スレッド実行＋必要に応じてタイムアウトを適用
                        eval_df = None
                        try:
                            _t0_eval = _tnow()
                            if EVAL_TIMEOUT is None:
                                eval_res_all = await asyncio.to_thread(
                                    evaluate,
                                    dataset=dataset,
                                    metrics=metrics_local,
                                    llm=ragas_llm,
                                    embeddings=embedder,
                                    run_config=RunConfig(timeout=EVAL_TIMEOUT, max_workers=MAX_PARALLEL_TASKS),
                                )
                            else:
                                eval_res_all = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        evaluate,
                                        dataset=dataset,
                                        metrics=metrics_local,
                                        llm=ragas_llm,
                                        embeddings=embedder,
                                        run_config=RunConfig(timeout=EVAL_TIMEOUT, max_workers=MAX_PARALLEL_TASKS),
                                    ),
                                    timeout=EVAL_TIMEOUT,
                                )
                            _tlog("ragas.evaluate", _t0_eval)
                            # 代表的な戻り: Resultオブジェクト。to_pandas() があればDataFrame化
                            try:
                                if hasattr(eval_res_all, "to_pandas"):
                                    eval_df = eval_res_all.to_pandas()
                                elif hasattr(eval_res_all, "to_dict") and hasattr(eval_res_all, "columns"):
                                    eval_df = eval_res_all  # 既にDataFrame互換
                                else:
                                    eval_df = None
                            except Exception:
                                eval_df = None
                        except asyncio.TimeoutError:
                            print(f"[警告] ragas.evaluate 一括評価がタイムアウト: timeout={EVAL_TIMEOUT}s")
                            eval_df = None
                        except TypeError:
                            # 互換性問題フォールバック: embeddings を外して実行
                            try:
                                if EVAL_TIMEOUT is None:
                                    eval_res_all = await asyncio.to_thread(
                                        evaluate,
                                        dataset=dataset,
                                        metrics=metrics_local,
                                        llm=ragas_llm,
                                        run_config=RunConfig(timeout=EVAL_TIMEOUT, max_workers=MAX_PARALLEL_TASKS),
                                    )
                                else:
                                    eval_res_all = await asyncio.wait_for(
                                        asyncio.to_thread(
                                            evaluate,
                                            dataset=dataset,
                                            metrics=metrics_local,
                                            llm=ragas_llm,
                                            run_config=RunConfig(timeout=EVAL_TIMEOUT, max_workers=MAX_PARALLEL_TASKS),
                                        ),
                                        timeout=EVAL_TIMEOUT,
                                    )
                                if hasattr(eval_res_all, "to_pandas"):
                                    eval_df = eval_res_all.to_pandas()
                                elif hasattr(eval_res_all, "to_dict") and hasattr(eval_res_all, "columns"):
                                    eval_df = eval_res_all
                                else:
                                    eval_df = None
                            except TypeError:
                                # 最終フォールバック: run_config も外す
                                try:
                                    if EVAL_TIMEOUT is None:
                                        eval_res_all = await asyncio.to_thread(
                                            evaluate,
                                            dataset=dataset,
                                            metrics=metrics_local,
                                            llm=ragas_llm,
                                        )
                                    else:
                                        eval_res_all = await asyncio.wait_for(
                                            asyncio.to_thread(
                                                evaluate,
                                                dataset=dataset,
                                                metrics=metrics_local,
                                                llm=ragas_llm,
                                            ),
                                            timeout=EVAL_TIMEOUT,
                                        )
                                    if hasattr(eval_res_all, "to_pandas"):
                                        eval_df = eval_res_all.to_pandas()
                                    elif hasattr(eval_res_all, "to_dict") and hasattr(eval_res_all, "columns"):
                                        eval_df = eval_res_all
                                    else:
                                        eval_df = None
                                except Exception:
                                    print("[警告] ragas.evaluate 一括評価フォールバック失敗")
                                    eval_df = None

                        # 評価メトリクスの定義と結果整形（answer_similarity を含む）
                        metrics_keys = [
                            "faithfulness",
                            "answer_relevancy",
                            "context_precision",
                            "context_recall",
                            "answer_correctness",
                            "answer_similarity",
                        ]
                        metrics_per_qa = []
                        metrics_avg = {k: 0.0 for k in metrics_keys}
                        try:
                            if eval_df is not None:
                                try:
                                    rows = eval_df.to_dict(orient="records")
                                except Exception:
                                    rows = []
                                for r in rows:
                                    item = {k: safe_val(r.get(k, 0.0)) for k in metrics_keys}
                                    metrics_per_qa.append(item)
                                # 平均値を計算（列が無い場合は0）
                                for k in metrics_keys:
                                    try:
                                        if hasattr(eval_df, "columns") and k in list(eval_df.columns):
                                            metrics_avg[k] = safe_val(float(eval_df[k].mean()))
                                        else:
                                            metrics_avg[k] = 0.0
                                    except Exception:
                                        metrics_avg[k] = 0.0
                            else:
                                # タイムアウトや失敗時のフォールバック（質問数ぶんの0レコード）
                                metrics_per_qa = [{k: 0.0 for k in metrics_keys} for _ in range(len(questions))]
                                metrics_avg = {k: 0.0 for k in metrics_keys}
                        except Exception:
                            # 万一の整形失敗時も0で埋める
                            metrics_per_qa = [{k: 0.0 for k in metrics_keys} for _ in range(len(questions))]
                            metrics_avg = {k: 0.0 for k in metrics_keys}
                        # 総合スコアの計算（重み付けは従来比率を踏襲）
                        overall_score = (
                            metrics_avg["answer_relevancy"] * 0.25 +
                            metrics_avg["faithfulness"] * 0.25 +
                            metrics_avg["context_precision"] * 0.2 +
                            metrics_avg["context_recall"] * 0.2 +
                            metrics_avg["answer_correctness"] * 0.1
                        )
                        overall_score = safe_val(overall_score)
                        
                        # チャンク関連の統計情報
                        num_chunks = len(chunks)
                        avg_chunk_len = int(sum(len(c) for c in chunks) / num_chunks) if num_chunks > 0 else 0
                        
                        # 必須キーのリスト
                        required_keys = [
                            "overall_score",
                            "faithfulness",
                            "answer_relevancy",
                            "answer_similarity",
                            "context_recall",
                            "context_precision",
                            "answer_correctness",
                            "avg_chunk_len",
                            "num_chunks",
                        ]
                        
                        print(f"[進捗] 評価メトリクスの計算が完了しました。総合スコア: {overall_score:.4f}")
                        # 評価結果を格納する辞書を作成
                        response_dict = {
                            "embedding_model": embedding_model,
                            "chunk_size": chunk_size_val if chunk_method != "semantic" else None,
                            "chunk_overlap": chunk_overlap_val if chunk_method != "semantic" else None,
                            "chunk_method": chunk_method,
                            "overall_score": overall_score,
                            "faithfulness": metrics_avg["faithfulness"],
                            "answer_relevancy": metrics_avg["answer_relevancy"],
                            "answer_similarity": metrics_avg.get("answer_similarity", 0.0),
                            "context_recall": metrics_avg["context_recall"],
                            "context_precision": metrics_avg["context_precision"],
                            "answer_correctness": metrics_avg["answer_correctness"],
                            "chunk_strategy": chunk_strategy,
                            "num_chunks": num_chunks,
                            "avg_chunk_len": avg_chunk_len,
                            "metrics": metrics_per_qa
                        }
                        
                        # セマンティックチャンキングの場合は類似度閾値を追加
                        if chunk_method == "semantic":
                            response_dict["similarity_threshold"] = similarity_threshold
                        
                        # 必須キーが含まれているか確認、なければデフォルト値を設定
                        for k in required_keys:
                            if k not in response_dict:
                                response_dict[k] = 0.0
                                
                        print(f"[進捗] チャンク方法 '{chunk_method}' の処理が完了しました。スコア: {overall_score:.4f}")
                        results.append(response_dict)
                    except Exception as e:
                        # エラー時も必ずエラー内容を返す
                        import traceback
                        error_detail = traceback.format_exc()
                        print(f"[エラー] チャンク方法 '{chunk_method}' の処理中にエラーが発生しました: {str(e)}")
                        traceback.print_exc()
                        results.append({
                            "error": str(e), 
                            "chunk_method": chunk_method,
                            "error_detail": error_detail,
                            "input_data": data
                        })
                
                print(f"[進捗] すべてのチャンク方法の評価が完了しました。結果数: {len(results)}")
                return results
            except Exception as e:
                # エラー時も必ずエラー内容を返す
                import traceback
                error_detail = traceback.format_exc()
                print(f"[重要エラー] evaluate_one_bulk処理全体で例外が発生: {str(e)}")
                traceback.print_exc()
                return {
                    "error": str(e), 
                    "error_detail": error_detail,
                    "input_data": data
                }

        # --- 本体分岐 ---
        print(f"[進捗] bulk_evaluate APIが呼び出されました")
        if isinstance(data, list):
            print(f"[進捗] リストデータを処理します。データ数: {len(data)}")
            results = []
            for i, d in enumerate(data):
                try:
                    print(f"[進捗] データ {i+1}/{len(data)} を処理中...")
                    if not isinstance(d, dict):
                        d = find_first_dict(d)
                    res = await evaluate_one_bulk(d)
                    results.append(res)
                    print(f"[進捗] データ {i+1}/{len(data)} の処理が完了しました")
                except Exception as e:
                    # 個別データでエラーが発生しても全体を止めず、エラー内容を追加
                    import traceback
                    error_detail = traceback.format_exc()
                    print(f"[エラー] データ {i+1}/{len(data)} の処理中にエラーが発生: {str(e)}")
                    traceback.print_exc()
                    results.append({
                        "error": str(e), 
                        "error_detail": error_detail,
                        "input_data": d
                    })
            print(f"[進捗] すべてのデータ処理が完了しました。結果数: {len(results)}")
            return results
        else:
            print(f"[進捗] 単一データを処理します")
            result = await evaluate_one_bulk(data)
            print(f"[進捗] 処理が完了しました")
            return result
    except Exception as e:
        # 異常時も辞書を直接返す（JSONResponse不使用）
        import traceback
        error_detail = traceback.format_exc()
        print(f"[重要エラー] bulk_evaluate全体例外: {str(e)}")
        traceback.print_exc()
        return {
            "error": str(e),
            "error_detail": error_detail
        }

# --- PDFアップロード＆QA自動生成API ---
from fastapi import UploadFile, File

@app.post("/uploadfile/")
async def uploadfile(file: UploadFile = File(...), question_llm_model: str = Form("mistral"), answer_llm_model: str = Form("mistral")):
    """
    PDFアップロード時にテキスト抽出→LLMで質問自動生成→LLMで回答自動生成まで行い、
    質問・回答セットを返すAPI。
    """
    # ■■ 最重要デバッグ情報 ■■
    print(f"[重要] uploadfile関数実行開始: ファイル名={file.filename}, サイズ={file.size if hasattr(file, 'size') else '不明'}")
    print(f"[重要] ファイル情報: {file=}, タイプ={type(file)}")
    import io
    try:
        try:
            try:
                # 1. PDFからテキスト抽出
                contents = await file.read()
                print(f"[重要] ファイル読み込み完了: {len(contents)}バイト")
                
                # BytesIOでラップして再利用可能なストリームを作成
                pdf_stream = io.BytesIO(contents)
                print(f"[重要] BytesIOストリーム作成完了: {pdf_stream.getbuffer().nbytes}バイト")
                
                # PyPDF2でPDF読み込み
                try:
                    reader = PdfReader(pdf_stream)
                    print(f"[重要] PdfReader初期化成功: {len(reader.pages)}ページ")
                    
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        text += page_text
                        print(f"[重要] ページ抽出: {len(page_text)}文字")
                    sample_text = text[:3000] if len(text) > 3000 else text
                    print(f"[重要] PDF抽出完了: 合計{len(text)}文字, サンプル={sample_text[:100]}...")
                except Exception as pdf_error:
                    print(f"[重要] PDF処理エラー: {pdf_error}")
                    # エラーでもdict形式で返す
                    return {"error": f"PDF処理エラー: {str(pdf_error)}"}
            except Exception as e:
                print(f"[重要] PDF処理エラー: {e}")
                return {"error": f"PDF処理エラー: {str(e)}"}
        except Exception as e:
            print(f"[重要] PDF処理エラー: {e}")
            return {"error": f"PDF処理エラー: {str(e)}"}

        # 2. LLMで質問セット自動生成（GPT-OSS固定）
        print("[重要] LLM質問生成開始 (GPT-OSS固定)")
        llm_q_instance = get_llm("gpt-oss")
        prompt_q = f"""
以下の内容に関する代表的な質問を日本語で5つ作成してください。\n---\n{text[:1500]}\n---\n質問：
"""
        try:
            questions_resp = llm_q_instance.invoke(prompt_q)
            print(f"[重要] LLM質問生成レスポンス取得: {len(questions_resp.content)}文字")
            questions = [q.strip() for q in questions_resp.content.split('\n') if q.strip()]
            print(f"[重要] 質問リスト生成完了: {len(questions)}件")
        except Exception as e:
            print(f"[重要] LLM質問生成例外: {e}")
            questions = []

        # 質問が空の場合やLLM失敗時はtext内から箇条書き・QAペアを抽出
        if not questions:
            import re
            print("[重要] 正規表現によるQA/箇条書き抽出開始")
            bullets = re.findall(r'^[\*\-\d\.]+\s*(.+)', text, re.MULTILINE)
            print(f"[重要] 箇条書き抽出結果: {len(bullets)}件")
            qas = re.findall(r'Q[\d：: ]*(.+?)\nA[\d：: ]*(.+?)(?=\nQ|\n\Z)', text, re.DOTALL)
            print(f"[重要] QA形式抽出結果: {len(qas)}件")
            if qas:
                questions = [q.strip() for q, a in qas]
                answers = [a.strip() for q, a in qas]
                print(f"[重要] QA形式から抽出: {len(questions)}件")
            elif bullets:
                questions = bullets[:5]
                answers = ["該当内容を本文から要約してください。"] * len(questions)
                print(f"[重要] 箇条書きから抽出: {len(questions)}件")
            else:
                # 各段落の先頭文を質問化
                paras = [p.strip() for p in text.split('\n') if p.strip()]
                questions = [f"{p[:20]}について説明してください。" for p in paras[:5]]
                answers = ["該当内容を本文から要約してください。"] * len(questions)
                print(f"[重要] 段落先頭文から生成: {len(questions)}件")
        else:
            # 3. LLMで回答セット自動生成（GPT-OSS固定）
            print("[重要] LLM回答生成開始 (GPT-OSS固定)")
            answers = []
            llm_a_instance = get_llm("gpt-oss")
            for i, q in enumerate(questions):
                try:
                    prompt_a = f"""
以下の内容に基づいて、次の質問に日本語で簡潔に答えてください。\n---\n{sample_text}\n---\n質問: {q}\n回答：
"""
                    answer_resp = llm_a_instance.invoke(prompt_a)
                    print(f"[重要] LLM回答{i+1}生成完了: {len(answer_resp.content)}文字")
                    answer = answer_resp.content.strip().split('\n')[0]
                    answers.append(answer)
                except Exception as e:
                    print(f"[重要] LLM回答{i+1}生成例外: {e}")
                    answers.append("該当内容を本文から要約してください。")

        # --- 最終ガード: questions/answersが空なら必ずダミー値を返す ---
        if not questions or not answers:
            print("[重要] ダミーQAセットを返却（questions/answersが空）")
            questions = ["この文書の主題は何ですか？"]
            answers = ["本文を要約してください。"]

        # --- qa_meta を生成（回答長の正規化スコア + ダミー回答フラグ）---
        try:
            max_len = max((len(a) for a in answers), default=1)
            dummy_patterns = ["該当内容を本文から要約", "本文を要約して"]
            qa_meta = []
            for a in answers:
                norm_len = (len(a) / max_len) if max_len else 0.0
                is_dummy = any(pat in a for pat in dummy_patterns)
                qa_meta.append({
                    "score": float(round(norm_len, 3)),
                    "is_auto_fixed": False,
                    "is_dummy_answer": bool(is_dummy),
                    "candidates": [a],
                    "candidate_scores": [float(round(norm_len, 3))]
                })
        except Exception as e:
            print(f"[警告] qa_meta生成時に例外: {e}。全件デフォルト値を設定します")
            qa_meta = [{
                "score": 1.0,
                "is_auto_fixed": False,
                "is_dummy_answer": False,
                "candidates": [a],
                "candidate_scores": [1.0]
            } for a in answers]

        # 4. 結果を辞書形式で返却（正常時は全キー、JSONResponseは使わない）
        print(f"[重要] API返却直前: {len(questions)}質問, {len(answers)}回答")
        for i, (q, a) in enumerate(zip(questions, answers)):
            print(f"[重要] Q{i+1}: {q}")
            print(f"[重要] A{i+1}: {a}")
        
        # dictを直接返す（JSONResponse不使用）
        return {
            "text": sample_text,
            "questions": questions,
            "answers": answers,
            "qa_meta": qa_meta
        }
    except Exception as e:
        # 異常時も辞書を直接返す（JSONResponse不使用）
        print(f"[重要] uploadfile全体例外: {e}")
        return {"error": str(e)}

# --- モデル・戦略リスト取得API（YAMLファイルを返す） ---
from fastapi.responses import JSONResponse

@app.get("/list_models")
def list_models():
    """
    models.yamlの内容を {"models": [...]} 形式で返すAPI。エラー時はprintログも出し、説明付きで返却。
    """
    import os
    try:
        # デバッグ用: カレントディレクトリとファイル一覧を表示
        print(f"[DEBUG] os.getcwd() = {os.getcwd()}")
        print(f"[DEBUG] os.listdir('.') = {os.listdir('.')}")
        abs_path = os.path.abspath("models.yaml")
        print(f"[DEBUG] models.yaml abs path = {abs_path}")
        print(f"[DEBUG] models.yaml exists = {os.path.exists(abs_path)}")
        # 読み込み前
        models_dict = load_models_yaml()
        print(f"[DEBUG] models_dict loaded: {models_dict}")
        if not models_dict or "models" not in models_dict:
            print("[list_models ERROR] models.yamlに'models'キーがありません")
            return JSONResponse(status_code=404, content={"error": "models.yamlに'models'キーがありません"})
        
        # モデルをカテゴリー別に分類
        categorized_models = {
            "LLM": [m for m in models_dict["models"] if m.get("category") == "LLM"],
            "Embedding": [m for m in models_dict["models"] if m.get("category") == "Embedding"]
        }
        
        print(f"[DEBUG] categorized_models: {categorized_models}")
        return JSONResponse(content=categorized_models)
    except Exception as e:
        print(f"[list_models ERROR] {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/list_strategies")
def list_strategies():
    """
    strategies.yamlの内容を返すAPI。エラー時は説明付きで返却。
    """
    try:
        strategies = load_strategies_yaml()
        return JSONResponse(content=strategies)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
