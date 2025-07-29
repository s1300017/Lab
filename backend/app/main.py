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

# 評価履歴管理APIを統合
try:
    from .evaluation_history_api import router as history_router
    app.include_router(history_router, prefix="/api/v1", tags=["evaluation_history"])
    print(f"[{jst_now_str()}] [INFO] 評価履歴管理APIを統合しました")
except ImportError as e:
    print(f"[{jst_now_str()}] [WARNING] 評価履歴管理APIのインポートに失敗: {e}")

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
                            experiment_id INTEGER,
                            chunks_details JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                    print(f"[{jst_now_str()}] [INFO] embeddingsテーブルを作成しました")
                else:
                    print(f"[{jst_now_str()}] [INFO] embeddingsテーブルは既に存在します")
                    
                    # 既存のembeddingsテーブルにexperiment_idカラムが存在するかチェック
                    column_check = conn.execute(text("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'embeddings' AND column_name = 'experiment_id'
                        )
                    """))
                    experiment_id_exists = column_check.scalar()
                    
                    if not experiment_id_exists:
                        print(f"[{jst_now_str()}] [INFO] embeddingsテーブルにexperiment_idカラムを追加します")
                        conn.execute(text("ALTER TABLE embeddings ADD COLUMN experiment_id INTEGER"))
                        print(f"[{jst_now_str()}] [INFO] experiment_idカラムを追加しました")
                    else:
                        print(f"[{jst_now_str()}] [INFO] experiment_idカラムは既に存在します")
                    
                    # chunks_detailsカラムの存在チェック
                    chunks_details_check = conn.execute(text("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'embeddings' AND column_name = 'chunks_details'
                        )
                    """))
                    chunks_details_exists = chunks_details_check.scalar()
                    
                    if not chunks_details_exists:
                        print(f"[{jst_now_str()}] [INFO] embeddingsテーブルにchunks_detailsカラムを追加します")
                        conn.execute(text("ALTER TABLE embeddings ADD COLUMN chunks_details JSONB"))
                        print(f"[{jst_now_str()}] [INFO] chunks_detailsカラムを追加しました")
                    else:
                        print(f"[{jst_now_str()}] [INFO] chunks_detailsカラムは既に存在します")
                
                # experimentsテーブルの作成
                result_exp = conn.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'experiments');"
                ))
                exp_table_exists = result_exp.scalar()
                
                if not exp_table_exists:
                    print(f"[{jst_now_str()}] [INFO] experimentsテーブルを作成します")
                    conn.execute(text("""
                        CREATE TABLE experiments (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            experiment_name TEXT,
                            file_id TEXT,
                            file_name TEXT,
                            parameters JSONB,
                            status TEXT DEFAULT 'running',
                            total_combinations INTEGER DEFAULT 0,
                            completed_combinations INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                    print(f"[{jst_now_str()}] [INFO] experimentsテーブルを作成しました")
                else:
                    print(f"[{jst_now_str()}] [INFO] experimentsテーブルは既に存在します")
                    
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



# --- PDFクレンジング関数 ---
def cleanse_pdf_text(text: str) -> str:
    """
    PDFテキストから表記号を日本語に変換し、ノイズ行や連続空白行を除去します。
    表記号変換はAIが扱いやすいように日本語へ置換します。
    Qiita記事（https://qiita.com/UKI_datascience/items/ba610c83c8f942f4b538）準拠。
    """
    import re
    # 1. 表記号→日本語変換
    table_symbol_map = {
        "│": "たて", "┃": "ふとたて", "─": "よこ", "━": "ふとよこ", "┏": "ひだりうえ", "┓": "みぎうえ",
        "┗": "ひだりした", "┛": "みぎした", "├": "ひだり", "┤": "みぎ", "┬": "うえ", "┴": "した", "┼": "てん",
        "|": "たて", "-": "よこ", "+": "てん", "＝": "よこ", "＝": "ふとよこ"
    }
    def replace_table_symbols(s):
        for k, v in table_symbol_map.items():
            s = s.replace(k, v)
        return s
    text = replace_table_symbols(text)
    # 2. ノイズ行除去（Qiita記事例: 表形式や記号が多い行・短い行など）
    lines = text.split('\n')
    cleansed = []
    for line in lines:
        # 罫線や記号が多い行の除去
        if re.match(r'^[\sたてよこふとてんうえしたひだりみぎ]+$', line):
            continue
        # 3文字以下の短い行も除去（必要なら）
        if len(line.strip()) <= 2:
            continue
        cleansed.append(line)
    # 3. 連続空白行の削除
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
from langchain_community.chat_models import ChatOllama
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
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,  # 追加: 回答の正確性指標

)
from datasets import Dataset

# モデルパスの設定
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LOCAL_MODEL_PATH = Path("/app/models/BAAI_bge-small-en-v1.5")

# セキュリティ注意: 本番環境ではAPIキーの表示は避けてください
import logging
logging.basicConfig(level=logging.INFO)
logging.info(f"[起動時] OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")


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
    モデル名に応じてLLMインスタンスを返す。
    models.yamlのname（例: 'mistral', 'gpt-4o-mini'）にも対応。
    未対応モデルは詳細付きで例外。
    """
    # OpenAI系モデル名はすべてこの分岐で返す
    openai_models = ["gpt-4o", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    if model_name in openai_models:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        # モデル名に応じてChatOpenAIインスタンスを返す
        return ChatOpenAI(model=model_name, temperature=0, openai_api_key=openai_api_key)
    elif model_name == "ollama_llama2":
        # llama2:7bモデルをOllamaで呼び出す
        return ChatOllama(model="llama2:7b", base_url="http://host.docker.internal:11434", temperature=0)
    elif model_name == "llama3":
        # llama3:latestモデルをOllamaで呼び出す
        return ChatOllama(model="llama3:latest", base_url="http://host.docker.internal:11434", temperature=0)
    elif model_name == "mistral":
        # mistral:latestモデルをOllamaで呼び出す
        return ChatOllama(model="mistral:latest", base_url="http://host.docker.internal:11434", temperature=0)
    else:
        # 日本語で詳細も返す
        raise ValueError(f"未対応のLLMモデルが指定されました: {model_name}")


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
    
    # OpenAIモデルのマッピング
    openai_models = {
        "gpt-4o": "text-embedding-ada-002",  # 旧モデル名との互換性のため
        "text-embedding-3-small": "text-embedding-3-small",
        "text-embedding-3-large": "text-embedding-3-large",
        "text-embedding-ada-002": "text-embedding-ada-002"
    }
    
    if model_name in openai_models:
        return OpenAIEmbeddings(
            model=openai_models[model_name],
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # ローカルHuggingFaceモデルのマッピング（GPU加速対応）
    local_hf_models = {
        "huggingface_bge_small": "/app/local_models/models/bge-small-en-v1.5",
        "huggingface_bge_large": "/app/local_models/models/bge-large-en-v1.5",
        "huggingface_miniLM": "/app/local_models/models/all-MiniLM-L6-v2",
        "huggingface_mpnet_base": "/app/local_models/models/all-mpnet-base-v2",
        "huggingface_multi_qa_minilm": "/app/local_models/models/multi-qa-MiniLM-L6-cos-v1",
        "huggingface_multi_qa_mpnet": "/app/local_models/models/multi-qa-mpnet-base-dot-v1",
        "huggingface_paraphrase_multilingual": "/app/local_models/models/paraphrase-multilingual-MiniLM-L12-v2",
        "huggingface_distiluse_multilingual": "/app/local_models/models/distiluse-base-multilingual-cased-v2",
        "huggingface_xlm_r": "/app/local_models/models/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
    }
    
    if model_name in local_hf_models:
        local_path = local_hf_models[model_name]
        print(f"[INFO] ローカル埋め込みモデルを使用: {local_path} (device: {device})")
        
        # GPU加速対応のHuggingFaceEmbeddingsを作成
        common_kwargs = {
            'model_kwargs': {
                'device': device,  # Metal GPU (mps) または CUDA
                'trust_remote_code': True
            },
            'encode_kwargs': {
                'normalize_embeddings': True,
                'device': device  # エンコード時もGPUを使用
            }
        }
        
        return HuggingFaceEmbeddings(
            model_name=local_path,
            **common_kwargs
        )
    
    raise ValueError(f"Unsupported embedding model: {model_name}")


# Default models（モデルが未ダウンロードでもサーバーが起動できるように修正）
current_llm = None
current_embeddings = None

# デフォルトでHuggingFaceのモデルを使用
try:
    # まずは軽量モデルを試す
    current_llm = get_llm("ollama_llama2")
except Exception as e:
    import logging
    logging.warning(f"LLM初期化失敗 (ollama_llama2): {e}")
    try:
        # 代替モデルを試す
        current_llm = get_llm("mistral")
    except Exception as e2:
        logging.warning(f"LLM初期化失敗 (mistral): {e2}")
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
    すべてのDBデータを完全削除するAPI。
    - ベクターストアコレクションの削除
    - embeddingsテーブルの全データ削除
    - experimentsテーブルの全データ削除
    """
    try:
        print(f"[{jst_now_str()}] [INFO] DB全体リセット開始")
        results = []
        
        # 1. PostgreSQLテーブルのデータ削除
        try:
            with engine.connect() as conn:
                with conn.begin():
                    # embeddingsテーブルの全データ削除
                    embeddings_result = conn.execute(text("DELETE FROM embeddings"))
                    embeddings_count = embeddings_result.rowcount
                    results.append(f"embeddingsテーブル: {embeddings_count}件削除")
                    print(f"[{jst_now_str()}] [INFO] embeddingsテーブル: {embeddings_count}件削除")
                    
                    # experimentsテーブルの全データ削除
                    experiments_result = conn.execute(text("DELETE FROM experiments"))
                    experiments_count = experiments_result.rowcount
                    results.append(f"experimentsテーブル: {experiments_count}件削除")
                    print(f"[{jst_now_str()}] [INFO] experimentsテーブル: {experiments_count}件削除")
                    
                    # シーケンスのリセット
                    conn.execute(text("ALTER SEQUENCE embeddings_id_seq RESTART WITH 1"))
                    conn.execute(text("ALTER SEQUENCE experiments_id_seq RESTART WITH 1"))
                    results.append("シーケンスリセット完了")
                    print(f"[{jst_now_str()}] [INFO] シーケンスリセット完了")
        except Exception as db_error:
            error_msg = f"PostgreSQLテーブル削除エラー: {str(db_error)}"
            results.append(error_msg)
            print(f"[{jst_now_str()}] [ERROR] {error_msg}")
        
        # 2. ベクターストアコレクションの削除
        if LOCAL_MODEL_PATH.exists():
            # 削除対象embeddingモデルリスト
            embedding_models = ["huggingface_bge_small", "huggingface_bge_large", "gpt-4o", "text-embedding-3-large"]
            for emb_model in embedding_models:
                try:
                    if emb_model in ["huggingface_bge_small", "huggingface_bge_large"]:
                        dummy_embeddings = HuggingFaceEmbeddings(
                            model_name=str(LOCAL_MODEL_PATH),
                            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                            encode_kwargs={'normalize_embeddings': True}
                        )
                    elif emb_model in ["gpt-4o", "text-embedding-3-large"]:
                        from langchain_openai import OpenAIEmbeddings
                        dummy_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
                    else:
                        continue
                    
                    vectorstore = PGVector.from_documents(
                        documents=[],
                        embedding=dummy_embeddings,
                        collection_name=get_collection_name(emb_model)
                    )
                    vectorstore.delete_collection()
                    results.append(f"{emb_model}コレクション: 削除成功")
                    print(f"[{jst_now_str()}] [INFO] {emb_model}コレクション削除成功")
                except Exception as e:
                    error_msg = f"{emb_model}コレクション: 削除失敗 ({str(e)})"
                    results.append(error_msg)
                    print(f"[{jst_now_str()}] [ERROR] {error_msg}")
        else:
            results.append(f"モデルが見つかりません: {LOCAL_MODEL_PATH}. ベクターストアコレクションの削除をスキップ")
        
        print(f"[{jst_now_str()}] [INFO] DB全体リセット完了")
        return {
            "status": "success",
            "message": "全DBデータを削除しました。",
            "details": results,
            "model_exists": LOCAL_MODEL_PATH.exists()
        }
    except Exception as e:
        error_msg = f"DB全体削除時エラー: {str(e)}"
        print(f"[{jst_now_str()}] [ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": error_msg,
            "model_exists": LOCAL_MODEL_PATH.exists()
        }

@app.get("/models/")
def get_available_models():
    """
    利用可能なモデルと現在のモデル状態を返します。
    """
    # HuggingFaceキャッシュディレクトリでの存在チェック
    hf_cache_path = Path("/app/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5")
    model_exists = hf_cache_path.exists() or LOCAL_MODEL_PATH.exists()
    
    # 実際に存在するパスを使用
    actual_path = hf_cache_path if hf_cache_path.exists() else LOCAL_MODEL_PATH
    
    model_info = {
        "model_name": str(MODEL_NAME),
        "local_path": str(LOCAL_MODEL_PATH),
        "hf_cache_path": str(hf_cache_path),
        "actual_path": str(actual_path),
        "exists": model_exists,
        "size_mb": (
            sum(f.stat().st_size for f in actual_path.glob('**/*') if f.is_file()) / (1024 * 1024)
        ) if model_exists else 0
    }
    
    return {
        "llm_models": ["llama3", "mistral", "ollama_llama2", "gpt-4o"],
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
        MAX_PARALLEL_TASKS = 5  # APIリクエスト制限に基づいて調整
        semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)

        async def evaluate_one_bulk(data):
            try:
                print("[進捗] 評価データを処理中...")
                embedding_model = data.get("embedding_model")
                llm_model = data.get("llm_model", "mistral")  # デフォルトでmistralを使用
                chunk_methods = data.get("chunk_methods", [data.get("chunk_method", "recursive")])
                chunk_sizes = data.get("chunk_sizes", [data.get("chunk_size", 1000)])
                chunk_overlaps = data.get("chunk_overlaps", [data.get("chunk_overlap", 0)])
                
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
                    'huggingface_xlm_r'
                }
                
                if embedding_model not in supported_models:
                    raise ValueError(f"未サポートの埋め込みモデルが指定されました: {embedding_model}")
                    
                # モデル名がopenaiの場合は、最新モデルを使用するように警告
                if embedding_model == "gpt-4o":
                    print("警告: 'openai' モデルは非推奨です。代わりに 'text-embedding-3-small' または 'text-embedding-3-large' の使用を検討してください。")

                qa_meta = []  # ←必ず初期化（関数のtryより前に移動）
                questions = data.get("questions")
                # ground_truthキーまたはanswersキーのどちらかを使用（互換性のため）
                answers = data.get("ground_truth", data.get("answers"))
                print(f"[DEBUG] qa_meta前: questions={questions}, answers={answers}, len_q={len(questions)}, len_a={len(answers)}")

                # --- uploadfileと同等の日本語質問フィルタを適用 ---
                def is_valid_japanese_question(q):
                    import re
                    if not re.search(r'[ぁ-んァ-ン一-龥]', q):
                        return False
                    exclude_patterns = [
                        r'^Q\d*[:：]?\s*$', r'^A\d*[:：]?\s*$', r'以下の\d*つの質問', r'代表的な質問', r'Here are', r'英語', r'質問です', r'answers? to', r'^一意回答$', r'^A[:：]', r'^Q[:：]',
                        r'^\d+[.．、)]?$', r'^\s*$', r'^\W+$'
                    ]
                    for pat in exclude_patterns:
                        if re.search(pat, q):
                            return False
                    if re.match(r'^[\d\.\s]*$', q):
                        return False
                    if len(q.strip()) < 6:
                        return False
                    return True

                # 質問・回答リストにフィルタ適用＆5件制限
                if questions and answers and len(questions) == len(answers):
                    filtered = [(q.strip(), a.strip()) for q, a in zip(questions, answers) if is_valid_japanese_question(q)]
                    filtered = filtered[:5]  # 5件にtruncate
                    questions = [q for q, a in filtered]
                    answers = [a for q, a in filtered]
                elif questions:
                    filtered = [q.strip() for q in questions if is_valid_japanese_question(q)]
                    filtered = filtered[:5]
                    questions = filtered
                    answers = [""] * len(questions)  # 回答がない場合は空欄
                else:
                    questions = []
                    answers = []

                print(f"[DEBUG] フィルタ後: questions={questions}, answers={answers}, len_q={len(questions)}, len_a={len(answers)}")

                # --- フィルタ後に5件未満 or 空なら即エラー ---
                if not questions or len(questions) < 5:
                    raise ValueError("有効な日本語質問が5件未満です。英語や無効な質問が混入していないか確認してください。アップロード時のPDFやQAセットを見直してください。")
                try:
                    import pandas as pd
                    if questions and answers and len(questions) == len(answers):
                        print("[DEBUG] qa_metaスコア計算ロジックに入る")
                    else:
                        print("[DEBUG] qa_metaスコア計算条件を満たさずスキップ")
                    if questions and answers and len(questions) == len(answers):
                        qa_df = pd.DataFrame({"question": questions, "answer": answers})
                        print(f"[DEBUG] qa_df内容: {qa_df}")
                        # 出現回数スコア
                        qa_df["count_score"] = qa_df.groupby(["question", "answer"])['answer'].transform('count')
                        qa_df["len_score"] = qa_df["answer"].apply(len)
                        qa_df["len_score"] = (qa_df["len_score"] - qa_df["len_score"].min()) / (qa_df["len_score"].max() - qa_df["len_score"].min() + 1e-6)
                        qa_df["total_score"] = qa_df["count_score"] + qa_df["len_score"]
                        for q, group in qa_df.groupby("question"):
                            print(f"[DEBUG] groupbyループ: q={q}, group={group}")
                            candidates = group[["answer", "total_score"]].to_dict("records")
                            # スコア最大の回答を選択
                            best_idx = group["total_score"].idxmax()
                            best_answer = group.loc[best_idx, "answer"]
                            best_score = group.loc[best_idx, "total_score"]
                            is_auto_fixed = len(group) > 1
                            qa_meta.append({
                                "score": float(best_score),
                                "is_auto_fixed": bool(is_auto_fixed),
                                "candidates": [c["answer"] for c in candidates],
                                "candidate_scores": [float(c["total_score"]) for c in candidates]
                            })
                    else:
                        # 質問・回答が空や不一致の場合はqa_metaも空リスト
                        qa_meta = []
                except Exception as e:
                    print(f"[警告] QA矛盾自動修正処理で例外: {e}")
                    qa_meta = []
                
                if not questions or not answers:
                    raise ValueError("questions/answersが指定されていません。PDFアップロード時の自動生成結果をそのまま送信してください。")
                if not (sample_text and questions and answers):
                    raise ValueError("PDFアップロードとQA自動生成を先に実施してください（text, questions, answers必須）。")

                results = []
                # embedding_modelのインスタンスを一度だけロードし再利用
                print(f"[進捗] 埋め込みモデル '{embedding_model}' をロード中...")
                embedder = get_embeddings(embedding_model)
                
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
                        
                        # ベクトルストア構築
                        vectorstore = PGVector.from_documents(
                            documents=[],  # 空で初期化
                            embedding=embedder,
                            collection_name=get_collection_name(embedding_model)
                        )
                        # チャンクをベクトルストアに追加（大量の場合はバッチ処理）
                        await asyncio.to_thread(vectorstore.add_texts, texts=chunks)
                        retriever = vectorstore.as_retriever()

                        # RAG回答生成＆コンテキスト取得
                        contexts = []
                        
                        # PDFアップロード時の回答がある場合はそれを使用、ない場合は新しく生成
                        if answers and len(answers) == len(questions):
                            print(f"[進捗] PDFアップロード時の回答を使用（{len(answers)}個の回答）")
                            pred_answers = answers  # PDFアップロード時の回答を使い回し
                            
                            # コンテキスト取得のみ実行
                            async def get_context_only(q):
                                async with semaphore:
                                    retrieved_docs = await asyncio.to_thread(retriever.get_relevant_documents, q)
                                    return [doc.page_content for doc in retrieved_docs]
                            
                            print(f"[進捗] コンテキスト取得のみ実行（{len(questions)}個の質問）...")
                            contexts = await asyncio.gather(*[get_context_only(q) for q in questions])
                        else:
                            print(f"[進捗] 新しいRAG回答を生成（{len(questions)}個の質問）...")
                            pred_answers = []
                            
                            # 各質問に対して非同期でコンテキスト取得と回答生成を行う
                            async def get_context_and_answer(q):
                                async with semaphore:  # セマフォで並列処理数を制限
                                    # 各質問ごとにリトリーバーで文脈取得（非同期化）
                                    retrieved_docs = await asyncio.to_thread(retriever.get_relevant_documents, q)
                                    context_texts = [doc.page_content for doc in retrieved_docs]
                                    # LLMインスタンス・プロンプト生成
                                    llm_instance = get_llm(llm_model)  # リクエストで指定されたLLMモデルを使用
                                    prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:\n{context}\n\nQuestion: {question}""")
                                    chain = (
                                        {"context": lambda _: context_texts, "question": lambda _: q}
                                        | prompt
                                        | llm_instance
                                        | StrOutputParser()
                                    )
                                    # 非同期で回答生成
                                    answer = await chain.ainvoke(q)
                                    return context_texts, answer
                            
                            # 非同期で全質問の回答を生成
                            results_list = await asyncio.gather(*[get_context_and_answer(q) for q in questions])
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
                        dataset = Dataset.from_dict(dataset_dict)
                        llm_instance_eval = get_llm(llm_model)  # リクエストで指定されたLLMモデルを使用
                        
                        # RAGASメトリクスを埋め込みモデルで初期化（OpenAI Embeddingsを回避）
                        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness
                        
                        # メトリクスを埋め込みモデルで初期化
                        metrics_with_embeddings = []
                        for metric in [faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness]:
                            # メトリクスに埋め込みモデルを設定
                            if hasattr(metric, 'embeddings'):
                                metric.embeddings = embedder
                            metrics_with_embeddings.append(metric)
                        
                        # 評価関数を非同期化
                        async def eval_one(idx):
                            async with semaphore:  # セマフォで並列処理数を制限
                                print(f"[進捗] 評価 {idx+1}/{len(questions)} 件目を処理中...") 
                                single_dataset = Dataset.from_dict({
                                    "question": [questions[idx]],
                                    "answer": [pred_answers[idx]],
                                    "contexts": [contexts[idx]],
                                    "ground_truth": [answers[idx]],
                                })
                                return idx, evaluate(
                                    dataset=single_dataset,
                                    metrics=metrics_with_embeddings,
                                    llm=llm_instance_eval,
                                )
                        
                        # 非同期で評価を実行
                        eval_results_with_idx = await asyncio.gather(*[eval_one(i) for i in range(len(questions))])
                        # 結果を元の順番に整理
                        eval_results = [None] * len(questions)
                        for idx, result in eval_results_with_idx:
                            eval_results[idx] = result

                        # 評価メトリクスの定義
                        metrics_keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
                        metrics_sum = {k: 0.0 for k in metrics_keys}
                        metrics_count = {k: 0 for k in metrics_keys}
                        metrics_per_qa = []
                        
                        # 各質問のメトリクスを収集
                        for idx, res in enumerate(eval_results):
                            scores = res.scores if hasattr(res, "scores") else {}
                            # scoresがlist型ならdictに変換（防御的処理）
                            if isinstance(scores, list):
                                if len(scores) > 0 and isinstance(scores[0], dict):
                                    scores = scores[0]
                                else:
                                    scores = {}
                                    
                            # 各質問ごとのメトリクスを格納
                            qa_metric = {
                                "question": questions[idx],
                                "answer": pred_answers[idx],
                                "faithfulness": safe_val(scores.get("faithfulness", 0.0)),
                                "answer_relevancy": safe_val(scores.get("answer_relevancy", 0.0)),
                                "context_recall": safe_val(scores.get("context_recall", 0.0)),
                                "context_precision": safe_val(scores.get("context_precision", 0.0)),
                                "answer_correctness": safe_val(scores.get("answer_correctness", 0.0)),
                            }
                            metrics_per_qa.append(qa_metric)
                            
                            # 合計を計算
                            for k in metrics_keys:
                                if k in scores and isinstance(scores[k], (float, int)):
                                    metrics_sum[k] += float(scores[k])
                                    metrics_count[k] += 1
                        
                        # 平均値を計算
                        metrics_avg = {k: safe_val(metrics_sum[k] / metrics_count[k] if metrics_count[k] > 0 else 0.0) for k in metrics_keys}
                        
                        # 総合スコアの計算
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
                            "overall_score", "faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness", "avg_chunk_len", "num_chunks"
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
                        
                        # 評価結果をデータベースに永続化
                        try:
                            # 実験セッションIDを生成（ユニークな実験を識別）
                            import uuid
                            session_id = data.get('session_id', str(uuid.uuid4()))
                            experiment_name = f"{embedding_model}_{chunk_method}_{chunk_size_val}_{chunk_overlap_val}"
                            
                            with engine.connect() as db_conn:
                                with db_conn.begin():
                                    # 1. 実験セッションをexperimentsテーブルに記録
                                    exp_result = db_conn.execute(text("""
                                        INSERT INTO experiments (
                                            session_id, experiment_name, file_id, file_name, parameters, 
                                            status, total_combinations, completed_combinations
                                        ) VALUES (
                                            :session_id, :experiment_name, :file_id, :file_name, :parameters,
                                            'completed', 1, 1
                                        ) RETURNING id
                                    """), {
                                        'session_id': session_id,
                                        'experiment_name': experiment_name,
                                        'file_id': data.get('file_id', 'unknown'),
                                        'file_name': data.get('file_name', 'unknown'),
                                        'parameters': json.dumps({
                                            'embedding_model': embedding_model,
                                            'chunk_method': chunk_method,
                                            'chunk_size': chunk_size_val,
                                            'chunk_overlap': chunk_overlap_val,
                                            'similarity_threshold': similarity_threshold if chunk_method == 'semantic' else None
                                        })
                                    })
                                    experiment_id = exp_result.fetchone()[0]
                                    
                                    # チャンク詳細情報を準備
                                    chunks_details = {
                                        'chunk_method': chunk_method,
                                        'chunk_strategy': chunk_strategy,
                                        'total_chunks': len(chunks),
                                        'chunks': [
                                            {
                                                'index': i,
                                                'content': chunk[:200] + '...' if len(chunk) > 200 else chunk,  # 最初の200文字のみ保存
                                                'length': len(chunk)
                                            }
                                            for i, chunk in enumerate(chunks[:10])  # 最初の10チャンクのみ保存
                                        ],
                                        'parameters': {
                                            'chunk_size': chunk_size_val if chunk_method != 'semantic' else None,
                                            'chunk_overlap': chunk_overlap_val if chunk_method != 'semantic' else None,
                                            'similarity_threshold': similarity_threshold if chunk_method == 'semantic' else None
                                        }
                                    }
                                    
                                    # 2. 評価結果をembeddingsテーブルに保存
                                    db_conn.execute(text("""
                                        INSERT INTO embeddings (
                                            text, embedding_model, chunk_strategy, chunk_size, chunk_overlap,
                                            avg_chunk_len, num_chunks, overall_score, faithfulness, answer_relevancy,
                                            context_recall, context_precision, answer_correctness, experiment_id, chunks_details
                                        ) VALUES (
                                            :text, :embedding_model, :chunk_strategy, :chunk_size, :chunk_overlap,
                                            :avg_chunk_len, :num_chunks, :overall_score, :faithfulness, :answer_relevancy,
                                            :context_recall, :context_precision, :answer_correctness, :experiment_id, :chunks_details
                                        )
                                    """), {
                                        'text': sample_text[:1000],  # テキストは最初の1000文字のみ保存
                                        'embedding_model': embedding_model,
                                        'chunk_strategy': chunk_strategy,
                                        'chunk_size': chunk_size_val if chunk_method != 'semantic' else None,
                                        'chunk_overlap': chunk_overlap_val if chunk_method != 'semantic' else None,
                                        'avg_chunk_len': avg_chunk_len,
                                        'num_chunks': num_chunks,
                                        'overall_score': overall_score,
                                        'faithfulness': metrics_avg['faithfulness'],
                                        'answer_relevancy': metrics_avg['answer_relevancy'],
                                        'context_recall': metrics_avg['context_recall'],
                                        'context_precision': metrics_avg['context_precision'],
                                        'answer_correctness': metrics_avg['answer_correctness'],
                                        'experiment_id': experiment_id,
                                        'chunks_details': json.dumps(chunks_details, ensure_ascii=False)
                                    })
                                    
                                    print(f"[進捗] 評価結果をデータベースに保存しました。experiment_id: {experiment_id}")
                                    
                                    # response_dictに実験IDを追加
                                    response_dict['experiment_id'] = experiment_id
                                    response_dict['session_id'] = session_id
                                    
                        except Exception as db_error:
                            print(f"[警告] データベース保存エラー: {str(db_error)}")
                            # データベースエラーがあっても評価結果は返却する
                            import traceback
                            traceback.print_exc()
                        
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
from fastapi import UploadFile, File, Form

@app.post("/uploadfile/")
async def uploadfile(
    file: UploadFile = File(...),
    question_llm_model: str = Form("llama3"),
    answer_llm_model: str = Form("llama3")
):
    """
    PDFアップロード時にテキスト抽出→LLMで質問自動生成→LLMで回答自動生成まで行い、
    質問・回答セットを返すAPI。
    """
    # ■■ 最重要デバッグ情報 ■■
    print(f"[重要] uploadfile関数実行開始: ファイル名={file.filename}, サイズ={file.size if hasattr(file, 'size') else '不明'}")
    print(f"[重要] ファイル情報: {file=}, タイプ={type(file)}")
    print(f"[重要] LLMモデル選択: 質問生成={question_llm_model}, 回答生成={answer_llm_model}")
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

        # 2. LLMで質問セット自動生成
        print(f"[重要] LLM質問生成開始: モデル={question_llm_model}")
        llm_instance = get_llm(question_llm_model)
        prompt_q = f"""
以下の内容に関する代表的な質問を日本語で5つ作成してください。各質問は改行で区切ってください。

---
{text[:1500]}
---

質問：
"""
        try:
            # タイムアウトを設定してLLM呼び出し
            import signal
            import time
            
            def timeout_handler(signum, frame):
                raise TimeoutError("LLM質問生成がタイムアウトしました")
            
            # 20秒のタイムアウトを設定（GPU加速で高速化）
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(20)
            
            start_time = time.time()
            questions_resp = llm_instance.invoke(prompt_q)
            signal.alarm(0)  # タイムアウトをクリア
            
            elapsed_time = time.time() - start_time
            print(f"[重要] LLM質問生成レスポンス取得: {len(questions_resp.content)}文字 (実行時間: {elapsed_time:.2f}秒)")
            questions = [q.strip() for q in questions_resp.content.split('\n') if q.strip()]
            answers = []  # LLM回答生成で後で埋める
            print(f"[重要] 質問リスト生成完了: {len(questions)}件")
        except TimeoutError as e:
            signal.alarm(0)  # タイムアウトをクリア
            print(f"[重要] LLM質問生成タイムアウト: {e}")
            # フォールバック処理に進む
            questions = []
        except Exception as e:
            signal.alarm(0)  # タイムアウトをクリア
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
                print(f"[重要] 箇条書きから抽出: {len(questions)}件")
                # 箇条書きから抽出した質問に対してもLLM回答生成を実行
                answers = []  # 後でLLM回答生成で埋める
            else:
                # 各段落の先頭文を質問化
                paras = [p.strip() for p in text.split('\n') if p.strip()]
                questions = [f"{p[:20]}について説明してください。" for p in paras[:5]]
                print(f"[重要] 段落先頭文から生成: {len(questions)}件")
                # 段落から生成した質問に対してもLLM回答生成を実行
                answers = []  # 後でLLM回答生成で埋める
        
        # 質問が生成された場合は、LLMで回答セットを自動生成
        if questions and not answers:
            # 3. LLMで回答セット自動生成
            print(f"[重要] LLM回答生成開始: モデル={answer_llm_model}")
            # 回答生成用のLLMインスタンスを取得
            try:
                answer_llm_instance = get_llm(answer_llm_model)
            except Exception as e:
                print(f"[重要] LLMインスタンス取得失敗: {e}")
                # LLMが使用できない場合は簡易回答でフォールバック
                answers = [f"{q}についての詳細は本文を参照してください。" for q in questions]
                print(f"[重要] 簡易回答でフォールバック: {len(answers)}件")
            else:
                answers = []
                for i, q in enumerate(questions):
                    try:
                        prompt_a = f"""
以下の内容に基づいて、次の質問に日本語で簡潔に答えてください。

---
{sample_text}
---

質問: {q}
回答："""
                        # タイムアウトを設定してLLM呼び出し
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"LLM回答{i+1}生成がタイムアウトしました")
                        
                        # 20秒のタイムアウトを設定（GPU加速で高速化）
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(20)
                        
                        start_time = time.time()
                        answer_resp = answer_llm_instance.invoke(prompt_a)
                        signal.alarm(0)  # タイムアウトをクリア
                        
                        elapsed_time = time.time() - start_time
                        print(f"[重要] LLM回答{i+1}生成完了: {len(answer_resp.content)}文字 (実行時間: {elapsed_time:.2f}秒)")
                        answer = answer_resp.content.strip().split('\n')[0]
                        answers.append(answer)
                    except TimeoutError as e:
                        signal.alarm(0)  # タイムアウトをクリア
                        print(f"[重要] LLM回答{i+1}生成タイムアウト: {e}")
                        # タイムアウト時は簡易的な回答を生成
                        simple_answer = f"{q}に関する内容を本文から要約してください。"
                        answers.append(simple_answer)
                    except Exception as e:
                        signal.alarm(0)  # タイムアウトをクリア
                        print(f"[重要] LLM回答{i+1}生成例外: {e}")
                        answers.append("該当内容を本文から要約してください。")

        # --- 最終ガード: questions/answersが空なら必ずダミー値を返す ---
        if not questions or not answers:
            print("[重要] ダミーQAセットを返却（questions/answersが空）")
            questions = ["この文書の主題は何ですか？"]
            answers = ["本文を要約してください。"]

        # 4. qa_metaを生成（信頼性スコア計算とダミー回答フラグ付き）
        print(f"[重要] qa_meta生成開始: {len(questions)}質問, {len(answers)}回答")
        
        try:
            import pandas as pd
            if questions and answers and len(questions) == len(answers):
                # pandasで信頼性スコアを計算
                qa_df = pd.DataFrame({"question": questions, "answer": answers})
                print(f"[重要] qa_df内容:\n{qa_df}")
                
                # 出現回数スコア計算
                qa_df["count_score"] = qa_df.groupby(["question", "answer"])['answer'].transform('count')
                
                # 回答長スコア計算
                qa_df["len_score"] = qa_df["answer"].apply(len)
                if qa_df["len_score"].max() > qa_df["len_score"].min():
                    qa_df["len_score"] = (qa_df["len_score"] - qa_df["len_score"].min()) / (qa_df["len_score"].max() - qa_df["len_score"].min())
                else:
                    qa_df["len_score"] = 0.5  # 全て同じ長さの場合
                
                # 総合スコア計算
                qa_df["total_score"] = qa_df["count_score"] + qa_df["len_score"]
                
                print(f"[重要] スコア計算結果:\n{qa_df[['question', 'answer', 'count_score', 'len_score', 'total_score']]}")
                
                # qa_metaを質問ごとに生成
                qa_meta = []
                for q, group in qa_df.groupby("question"):
                    print(f"[重要] groupbyループ: q={q}, group=\n{group}")
                    candidates = group[["answer", "total_score"]].to_dict("records")
                    best_idx = group["total_score"].idxmax()
                    best_answer = group.loc[best_idx, "answer"]
                    best_score = group.loc[best_idx, "total_score"]
                    is_auto_fixed = len(group) > 1
                    
                    # ダミー回答判定パターンを拡張
                    dummy_patterns = [
                        "該当内容を本文から要約してください。",
                        "本文を要約してください。",
                        "以下の条件が必要です：",
                        "条件が必要です",
                        "条件は以下の通りです",
                        "条件について説明してください"
                    ]
                    is_dummy_answer = any(pattern in best_answer for pattern in dummy_patterns) or best_answer.endswith("です：") or len(best_answer.strip()) < 10
                    
                    qa_meta.append({
                        "score": float(best_score),
                        "is_auto_fixed": bool(is_auto_fixed),
                        "is_dummy_answer": bool(is_dummy_answer),
                        "candidates": [c["answer"] for c in candidates],
                        "candidate_scores": [float(c["total_score"]) for c in candidates]
                    })
                    
                print(f"[重要] qa_meta生成結果: {qa_meta}")
            else:
                print(f"[警告] qa_meta生成スキップ: questions={len(questions)}, answers={len(answers)}")
                qa_meta = []
        except Exception as e:
            print(f"[警告] qa_meta生成例外: {e}")
            import traceback
            traceback.print_exc()
            qa_meta = []
        
        # qa_metaが空の場合のフォールバック
        if not qa_meta and questions and answers and len(questions) == len(answers):
            print("[重要] qa_metaが空なのでフォールバック処理を実行")
            dummy_patterns = [
                "該当内容を本文から要約してください。",
                "本文を要約してください。",
                "以下の条件が必要です：",
                "条件が必要です",
                "条件は以下の通りです",
                "条件について説明してください"
            ]
            qa_meta = [
                {
                    "score": 1.0 + (len(a) / 100.0),  # 回答長に応じてスコアを変化
                    "is_auto_fixed": False,
                    "is_dummy_answer": any(pattern in a for pattern in dummy_patterns) or a.endswith("です：") or len(a.strip()) < 10,
                    "candidates": [a],
                    "candidate_scores": [1.0 + (len(a) / 100.0)]
                }
                for a in answers
            ]
        
        # 5. 結果を辞書形式で返却（正常時は全キー、JSONResponseは使わない）
        print(f"[重要] API返却直前: {len(questions)}質問, {len(answers)}回答, {len(qa_meta)}meta")
        for i, (q, a) in enumerate(zip(questions, answers)):
            print(f"[重要] Q{i+1}: {q}")
            print(f"[重要] A{i+1}: {a}")
            if i < len(qa_meta):
                meta = qa_meta[i]
                print(f"[重要] Meta{i+1}: score={meta.get('score', 'N/A')}, is_dummy={meta.get('is_dummy_answer', 'N/A')}, is_auto_fixed={meta.get('is_auto_fixed', 'N/A')}")
        
        # dictを直接返す（JSONResponse不使用）
        result = {
            "text": sample_text,
            "questions": questions,
            "answers": answers
        }
        print(f"[重要] 最終返却結果: qa_meta長={len(qa_meta)}, サンプル={qa_meta[:1] if qa_meta else 'None'}")
        return result
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

# --- 実験管理API ---

@app.get("/api/v1/experiments/")
def get_experiments():
    """
    実験一覧を取得するAPI
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    id, session_id, experiment_name, file_name, 
                    parameters, status, total_combinations, completed_combinations,
                    created_at, updated_at
                FROM experiments 
                ORDER BY created_at DESC
            """))
            
            experiments = []
            for row in result:
                exp = {
                    "id": row[0],
                    "session_id": row[1],
                    "experiment_name": row[2],
                    "file_name": row[3],
                    "parameters": row[4] if isinstance(row[4], dict) else (json.loads(row[4]) if row[4] and isinstance(row[4], str) else {}),
                    "status": row[5],
                    "total_combinations": row[6],
                    "completed_combinations": row[7],
                    "created_at": row[8].isoformat() if row[8] else None,
                    "updated_at": row[9].isoformat() if row[9] else None
                }
                experiments.append(exp)
            
            return JSONResponse(content={"experiments": experiments})
    except Exception as e:
        print(f"[エラー] 実験一覧取得エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/v1/experiments/{experiment_id}/detailed_results/")
def get_experiment_results(experiment_id: int):
    """
    指定した実験の詳細結果を取得するAPI（チャンク詳細情報含む）
    """
    try:
        with engine.connect() as conn:
            # 実験情報を取得
            exp_result = conn.execute(text("""
                SELECT session_id, experiment_name, file_name, parameters, status, 
                       total_combinations, completed_combinations, created_at
                FROM experiments 
                WHERE id = :experiment_id
            """), {"experiment_id": experiment_id})
            
            exp_row = exp_result.fetchone()
            
            if not exp_row:
                return {"error": "実験が見つかりません"}
            
            # parametersの安全なパース
            parameters = {}
            if exp_row[3] is not None:
                if isinstance(exp_row[3], dict):
                    parameters = exp_row[3]
                elif isinstance(exp_row[3], str):
                    try:
                        parameters = json.loads(exp_row[3])
                    except (json.JSONDecodeError, TypeError):
                        parameters = {}
            
            experiment_info = {
                "session_id": exp_row[0],
                "experiment_name": exp_row[1],
                "file_name": exp_row[2],
                "parameters": parameters,
                "status": exp_row[4],
                "total_combinations": exp_row[5],
                "completed_combinations": exp_row[6],
                "created_at": exp_row[7].isoformat() if exp_row[7] else None
            }
            
            # 評価結果を取得（チャンク詳細情報含む）
            results_query = conn.execute(text("""
                SELECT 
                    id, text, embedding_model, chunk_strategy, chunk_size, chunk_overlap,
                    avg_chunk_len, num_chunks, overall_score, faithfulness, answer_relevancy,
                    context_recall, context_precision, answer_correctness, chunks_details,
                    created_at
                FROM embeddings 
                WHERE experiment_id = :experiment_id
                ORDER BY created_at ASC
            """), {"experiment_id": experiment_id})
            
            results = []
            for row in results_query:
                # chunks_detailsの安全なパース
                chunks_details = None
                if row[14] is not None:
                    if isinstance(row[14], dict):
                        chunks_details = row[14]
                    elif isinstance(row[14], str):
                        try:
                            chunks_details = json.loads(row[14])
                        except (json.JSONDecodeError, TypeError):
                            chunks_details = None
                
                result_data = {
                    "id": row[0],
                    "text": row[1],
                    "embedding_model": row[2],
                    "chunk_strategy": row[3],
                    "chunk_size": row[4],
                    "chunk_overlap": row[5],
                    "avg_chunk_len": row[6],
                    "num_chunks": row[7],
                    "overall_score": row[8],
                    "faithfulness": row[9],
                    "answer_relevancy": row[10],
                    "context_recall": row[11],
                    "context_precision": row[12],
                    "answer_correctness": row[13],
                    "chunks_details": chunks_details,
                    "created_at": row[15].isoformat() if row[15] else None
                }
                results.append(result_data)
            
            return {
                "experiment": experiment_info,
                "results": results,
                "total_results": len(results)
            }
    except Exception as e:
        print(f"[エラー] 実験結果取得エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.delete("/api/v1/experiments/{experiment_id}/")
def delete_experiment(experiment_id: int):
    """
    指定した実験を削除するAPI
    """
    try:
        with engine.connect() as conn:
            with conn.begin():
                # 関連する評価結果を削除
                conn.execute(text("""
                    DELETE FROM embeddings WHERE experiment_id = :experiment_id
                """), {"experiment_id": experiment_id})
                
                # 実験を削除
                result = conn.execute(text("""
                    DELETE FROM experiments WHERE id = :experiment_id
                """), {"experiment_id": experiment_id})
                
                if result.rowcount == 0:
                    return JSONResponse(status_code=404, content={"error": "実験が見つかりません"})
                
                return JSONResponse(content={"message": "実験を削除しました"})
    except Exception as e:
        print(f"[エラー] 実験削除エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/v1/experiments/statistics/")
def get_experiments_statistics():
    """
    実験の統計情報を取得するAPI
    """
    try:
        with engine.connect() as conn:
            # 実験数と結果数の統計
            stats_result = conn.execute(text("""
                SELECT 
                    COUNT(DISTINCT e.id) as total_experiments,
                    COUNT(em.id) as total_results,
                    AVG(em.overall_score) as avg_overall_score,
                    MAX(em.overall_score) as max_overall_score,
                    MIN(em.overall_score) as min_overall_score
                FROM experiments e
                LEFT JOIN embeddings em ON e.id = em.experiment_id
            """))
            
            stats_row = stats_result.fetchone()
            
            # モデル別統計
            model_stats = conn.execute(text("""
                SELECT 
                    embedding_model,
                    COUNT(*) as count,
                    AVG(overall_score) as avg_score,
                    MAX(overall_score) as max_score,
                    MIN(overall_score) as min_score
                FROM embeddings
                GROUP BY embedding_model
                ORDER BY avg_score DESC
            """))
            
            model_statistics = []
            for row in model_stats:
                model_statistics.append({
                    "model": row[0],
                    "count": row[1],
                    "avg_score": float(row[2]) if row[2] else 0,
                    "max_score": float(row[3]) if row[3] else 0,
                    "min_score": float(row[4]) if row[4] else 0
                })
            
            # チャンク戦略別統計
            strategy_stats = conn.execute(text("""
                SELECT 
                    chunk_strategy,
                    COUNT(*) as count,
                    AVG(overall_score) as avg_score,
                    MAX(overall_score) as max_score,
                    MIN(overall_score) as min_score
                FROM embeddings
                GROUP BY chunk_strategy
                ORDER BY avg_score DESC
            """))
            
            strategy_statistics = []
            for row in strategy_stats:
                strategy_statistics.append({
                    "strategy": row[0],
                    "count": row[1],
                    "avg_score": float(row[2]) if row[2] else 0,
                    "max_score": float(row[3]) if row[3] else 0,
                    "min_score": float(row[4]) if row[4] else 0
                })
            
            return JSONResponse(content={
                "overall": {
                    "total_experiments": stats_row[0] if stats_row[0] else 0,
                    "total_results": stats_row[1] if stats_row[1] else 0,
                    "avg_overall_score": float(stats_row[2]) if stats_row[2] else 0,
                    "max_overall_score": float(stats_row[3]) if stats_row[3] else 0,
                    "min_overall_score": float(stats_row[4]) if stats_row[4] else 0
                },
                "by_model": model_statistics,
                "by_strategy": strategy_statistics
            })
    except Exception as e:
        print(f"[エラー] 統計情報取得エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
