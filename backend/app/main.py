from datetime import datetime
from pytz import timezone
import os

def jst_now_str():
    return datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')

from fastapi import FastAPI, UploadFile, File, HTTPException
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
async def uploadfile(file: UploadFile = File(...)):
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
            sample_text = text[:3000] if len(text) > 3000 else text
            print(f"[重要] PDF抽出完了: 合計{len(text)}文字, サンプル={sample_text[:100]}...")
        except Exception as pdf_error:
            print(f"[重要] PDF処理エラー: {pdf_error}")
            return {"error": f"PDF処理エラー: {str(pdf_error)}"}
        print("[重要] LLM質問生成開始")
        llm_instance = get_llm("openai")
        prompt_q = f"""
以下の内容に関する代表的な質問を日本語で5つ作成してください。\n---\n{text[:1500]}\n---\n質問：
"""
        try:
            questions_resp = llm_instance.invoke(prompt_q)
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
            for i, q in enumerate(questions):
                try:
                    prompt_a = f"""
以下の内容に基づいて、次の質問に日本語で簡潔に答えてください。\n---\n{sample_text}\n---\n質問: {q}\n回答：
"""
                    answer_resp = llm_instance.invoke(prompt_a)
                    print(f"[重要] LLM回答{i+1}生成完了: {len(answer_resp.content)}文字")
                    answer = answer_resp.content.strip().split('\n')[0]
                    answers.append(answer)
                except Exception as e:
                    print(f"[重要] LLM回答{i+1}生成例外: {e}")
                    answers.append("該当内容を本文から要約してください。")
        if not questions or not answers:
            print("[重要] ダミーQAセットを返却（questions/answersが空）")
            questions = ["この文書の主題は何ですか？"]
            answers = ["本文を要約してください。"]
        print(f"[重要] API返却直前: questions={questions}, answers={answers}")
        # 4. 抽出データ保存
        extracted_path = EXTRACTED_DIR / f"{file_id}.json"
        with open(extracted_path, "w", encoding="utf-8") as f_json:
            json.dump({
                "text": sample_text,
                "questions": questions,
                "answers": answers,
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
            "file_name": file.filename,  # ←file_nameで統一
        }
    except Exception as e:
        print(f"[重要] uploadfile全体例外: {e}")
        return {"error": str(e)}

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

# 設定ファイルのパス
MODELS_YAML_PATH = Path(__file__).parent.parent / "models.yaml"
STRATEGIES_YAML_PATH = Path(__file__).parent.parent / "strategies.yaml"

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
    文単位で分割
    """
    import nltk
    return nltk.sent_tokenize(text)

def paragraph_chunk_text(text):
    """
    段落単位で分割（空行または改行2つで区切る）
    """
    import re
    paras = re.split(r'\n\s*\n', text)
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
    
    # 文単位で分割
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [text]
        
    print(f"セマンティックチャンキング: {len(sentences)}文を処理中...")
    
    # 各文のembedding取得
    if embedding_model is None:
        raise ValueError("embedding_modelが指定されていません")
    
    # バッチ処理で埋め込みを取得（大量の文がある場合に備えて）
    batch_size = 32
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    
    # チャンク生成
    chunks = []
    current_chunk = []
    
    for i in range(len(sentences)):
        current_sentence = sentences[i]
        current_embedding = np.array(embeddings[i]).reshape(1, -1)
        
        # 現在のチャンクが空の場合は追加
        if not current_chunk:
            current_chunk.append(current_sentence)
            continue
            
        # 現在のチャンクの最後の文との類似度を計算
        last_embedding = np.array(embeddings[i-1]).reshape(1, -1)
        similarity = cosine_similarity(last_embedding, current_embedding)[0][0]
        
        # 類似度が閾値より低い場合にのみ新しいチャンクを開始
        if similarity < similarity_threshold:
            if current_chunk:  # 現在のチャンクを保存
                chunks.append(' '.join(current_chunk))
                current_chunk = [current_sentence]  # 新しいチャンクを開始
        else:
            current_chunk.append(current_sentence)
    
    # 最後のチャンクを追加
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
    if model_name in ["openai", "gpt-4o-mini"]:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        # gpt-4o-mini も openai も同じChatOpenAIで返す
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    elif model_name in ["ollama_llama2", "mistral"]:
        # mistral も ollama_llama2 もOllamaで返す
        return OllamaLLM(model="mistral", base_url="http://ollama:11434")
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
    if model_name == "openai":
        return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    elif model_name == "huggingface_bge_small":
        # ローカルのモデルを使用
        if not LOCAL_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {LOCAL_MODEL_PATH}. "
                "Please make sure the model is correctly downloaded and mounted."
            )
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={
                'device': device,  # 自動判定したデバイスを指定
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
    elif model_name == "huggingface_miniLM":
        # 軽量・多言語対応モデル（初回のみ自動DL）
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': device,  # 自動判定したデバイスを指定
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
    else:
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
    embedding_model: str # Added for dynamic selection

class QueryRequest(BaseModel):
    query: str
    llm_model: str = "mistral"  # デフォルト値を設定
    embedding_model: str = "huggingface_bge_small"  # デフォルト値を設定

class EvalRequest(BaseModel):
    questions: list[str]
    answers: list[str]
    contexts: list[list[str]]
    llm_model: str # Added for dynamic selection
    embedding_model: str # Added for dynamic selection
    include_overlap_metrics: bool = False

class ModelSelection(BaseModel):
    llm_model: str
    embedding_model: str

# --- API Endpoints ---


@app.post("/chunk/")
def chunk_text(request: ChunkRequest):
    """
    chunk_methodに応じて適切な方法でテキストをチャンク分割
    - recursive: 再帰的にテキストを分割（デフォルト）
    - semantic: 意味的なまとまりで分割（embeddingモデルが必要）
    """
    if request.chunk_method == 'semantic':
        # embedding_modelが指定されていることを確認
        if not request.embedding_model:
            raise HTTPException(
                status_code=400,
                detail="semanticチャンキングにはembedding_modelの指定が必要です"
            )
        try:
            # 埋め込みモデルを取得
            embedder = get_embeddings(request.embedding_model)
            print(f"セマンティックチャンキングを開始します（chunk_sizeとchunk_overlapは無視されます）...")
            # セマンティックチャンキングを実行（chunk_sizeとchunk_overlapは無視）
            chunks = semantic_chunk_text(
                text=request.text,
                chunk_size=None,  # 無視される
                chunk_overlap=None,  # 無視される
                embedding_model=embedder
            )
            return {"chunks": chunks}
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"セマンティックチャンキング中にエラーが発生しました: {str(e)}"
            )
            
    elif request.chunk_method == 'recursive':
        # 再帰的チャンキング（デフォルト）
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                length_function=len,
            )
            chunks = text_splitter.split_text(request.text)
            return {"chunks": chunks}
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"テキストのチャンキング中にエラーが発生しました: {str(e)}"
            )
    else:
        # 未対応のチャンキング方法が指定された場合
        raise HTTPException(
            status_code=400,
            detail=f"未対応のchunk_method: {request.chunk_method}。'recursive' または 'semantic' を指定してください。"
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
        vectorstore.add_texts(texts=request.chunks)
        return {"message": f"Successfully embedded and stored {len(request.chunks)} chunks using {request.embedding_model}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
def query_rag(request: QueryRequest):
    try:
        # モデルを初期化
        llm_instance = get_llm(request.llm_model)
        embeddings_instance = get_embeddings(request.embedding_model)

        # データベースからテキストを取得
        with SessionLocal() as session:
            result = session.execute(
                text("SELECT text FROM embeddings WHERE chunk_strategy = 'test'")
            )
            texts = [row[0] for row in result.fetchall()]
            
            if not texts:
                return {
                    "answer": "データベースにテストデータが見つかりません。先にテストデータを挿入してください。", 
                    "contexts": []
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

@app.post("/evaluate/")
def evaluate_ragas(request: EvalRequest):
    try:
        llm_instance = get_llm(request.llm_model)
        embeddings_instance = get_embeddings(request.embedding_model)

        dataset_dict = {
            "question": request.questions,
            "answer": request.answers,
            "contexts": request.contexts,
            "ground_truth": request.answers 
        }
        dataset = Dataset.from_dict(dataset_dict)

        # 評価メトリクスの定義
        metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]

        # オーバーラップメトリクスを追加
        if request.include_overlap_metrics:
            # 拡張されたオーバーラップメトリクスを計算
            overlap_metrics = calculate_overlap_metrics(request.contexts, embeddings_instance)
            
            # 各質問のコンテキストに対してオーバーラップを計算
            per_question_overlaps = []
            for ctx_list in request.contexts:
                metrics = calculate_overlap_metrics(ctx_list, embeddings_instance)
                per_question_overlaps.append(metrics)
            
            # 平均オーバーラップを計算
            avg_overlap = overlap_metrics["overlap_ratio"]
            avg_adjacent_overlap = overlap_metrics.get("avg_adjacent_overlap", 0.0)
            semantic_overlap = overlap_metrics.get("semantic_overlap", 0.0)
        else:
            avg_overlap = 0.0
            avg_adjacent_overlap = 0.0
            semantic_overlap = 0.0
            per_question_overlaps = [{"overlap_ratio": 0.0, "adjacent_overlap": [0.0], "semantic_overlap": 0.0} for _ in request.contexts]

        # 評価を実行
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm_instance,
            embeddings=embeddings_instance,
        )

        # 必須評価指標キーを全て含める（欠損時は0.0で埋める）
        required_keys = [
            "overall_score", "faithfulness", "answer_relevancy", 
            "context_recall", "context_precision", "answer_correctness", 
            "avg_chunk_len", "num_chunks"
        ]
        
        # ragasのscoresはdictで返る
        scores = result.scores if isinstance(result.scores, dict) else {}
        
        # 欠損キーを0.0で補完
        for k in required_keys:
            if k not in scores:
                scores[k] = 0.0
        
        # オーバーラップメトリクスを追加
        if request.include_overlap_metrics:
            scores["overlap_ratio"] = avg_overlap
            scores["avg_adjacent_overlap"] = avg_adjacent_overlap
            scores["semantic_overlap"] = semantic_overlap
            scores["per_question_overlaps"] = per_question_overlaps
        
        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        embedding_models = ["huggingface_bge_small", "openai"]
        results = []
        for emb_model in embedding_models:
            try:
                if emb_model == "huggingface_bge_small":
                    dummy_embeddings = HuggingFaceEmbeddings(
                        model_name=str(LOCAL_MODEL_PATH),
                        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                elif emb_model == "openai":
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
        "llm_models": ["ollama_llama2", "openai"],
        "embedding_models": ["huggingface_bge_small", "openai"],
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

        async def evaluate_one_bulk(data):
            try:
                embedding_model = data.get("embedding_model")
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
                questions = data.get("questions")
                answers = data.get("answers")
                if not questions or not answers:
                    raise ValueError("questions/answersが指定されていません。PDFアップロード時の自動生成結果をそのまま送信してください。")
                if not (sample_text and questions and answers):
                    raise ValueError("PDFアップロードとQA自動生成を先に実施してください（text, questions, answers必須）。")

                results = []
                # embedding_modelのインスタンスを一度だけロードし再利用
                embedder = get_embeddings(embedding_model)
                
                # chunk_method/chunk_size/chunk_overlapごとに完全に独立してチャンク分割→ベクトルストア→retriever→RAG回答生成→評価→スコア集計を実行
                for i in range(len(chunk_methods)):
                    try:
                        chunk_method = chunk_methods[i]
                        
                        # セマンティックチャンキングの場合、チャンクサイズとオーバーラップは無視する
                        if chunk_method == "semantic":
                            if not embedding_model:
                                results.append({
                                    "error": "セマンティックチャンキングにはembedding_modelの指定が必須です", 
                                    "chunk_method": chunk_method
                                })
                                continue
                                
                            print(f"セマンティックチャンキングを開始します（chunk_sizeとchunk_overlapは無視されます）...")
                            
                            # セマンティックチャンキングのパラメータを取得
                            semantic_params = data.get("semantic_params", {})
                            similarity_threshold = float(semantic_params.get("similarity_threshold", 0.7))
                            
                            print(f"セマンティックチャンキングを実行: similarity_threshold={similarity_threshold}")
                            chunks = semantic_chunk_text(
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
                            if chunk_method == "recursive":
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=chunk_sizes[i],
                                    chunk_overlap=chunk_overlaps[i],
                                    length_function=len,
                                )
                                chunks = text_splitter.split_text(sample_text)
                            elif chunk_method == "fixed":
                                chunks = fixed_chunk_text(
                                    sample_text, 
                                    chunk_size=chunk_sizes[i], 
                                    chunk_overlap=chunk_overlaps[i]
                                )
                            elif chunk_method == "sentence":
                                chunks = sentence_chunk_text(sample_text)
                            elif chunk_method == "paragraph":
                                chunks = paragraph_chunk_text(sample_text)
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

                        # ベクトルストア構築
                        vectorstore = PGVector.from_documents(
                            documents=[],  # 空で初期化
                            embedding=embedder,
                            collection_name=get_collection_name(embedding_model)
                        )
                        vectorstore.add_texts(texts=chunks)
                        retriever = vectorstore.as_retriever()

                        # RAG回答生成＆コンテキスト取得
                        contexts = []
                        pred_answers = []
                        import asyncio
                        async def get_context_and_answer(q):
                            # 各質問ごとにリトリーバーで文脈取得
                            retrieved_docs = retriever.get_relevant_documents(q)
                            context_texts = [doc.page_content for doc in retrieved_docs]
                            # LLMインスタンス・プロンプト生成
                            llm_instance = get_llm("openai")  # 必ずOpenAIモデルを使用
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
                        results_async = asyncio.get_event_loop()
                        results_list = results_async.run_until_complete(asyncio.gather(*[get_context_and_answer(q) for q in questions]))
                        for context_texts, answer in results_list:
                            contexts.append(context_texts)
                            pred_answers.append(answer)
                        # --- ここまで並列化 ---

                        # RAGAS等で自動評価
                        dataset_dict = {
                            "question": questions,
                            "answer": pred_answers,
                            "contexts": contexts,
                            "ground_truth": answers
                        }
                        dataset = Dataset.from_dict(dataset_dict)
                        llm_instance_eval = get_llm("openai")
                        async def eval_one(idx, llm_instance):
                            single_dataset = Dataset.from_dict({
                                "question": [questions[idx]],
                                "answer": [pred_answers[idx]],
                                "contexts": [contexts[idx]],
                                "ground_truth": [answers[idx]],
                            })
                            return evaluate(
                                dataset=single_dataset,
                                metrics=[faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness],
                                llm=llm_instance,
                            )
                        eval_results = results_async.run_until_complete(asyncio.gather(*[eval_one(i, llm_instance_eval) for i in range(len(questions))]))

                        metrics_keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
                        metrics_sum = {k: 0.0 for k in metrics_keys}
                        metrics_count = {k: 0 for k in metrics_keys}
                        for res in eval_results:
                            scores = res.scores if hasattr(res, "scores") else {}
                            for k in metrics_keys:
                                if k in scores and isinstance(scores[k], (float, int)):
                                    metrics_sum[k] += float(scores[k])
                                    metrics_count[k] += 1
                        metrics_avg = {k: safe_val(metrics_sum[k] / metrics_count[k] if metrics_count[k] > 0 else 0.0) for k in metrics_keys}
                        overall_score = (
                            metrics_avg["answer_relevancy"] * 0.25 +
                            metrics_avg["faithfulness"] * 0.25 +
                            metrics_avg["context_precision"] * 0.2 +
                            metrics_avg["context_recall"] * 0.2 +
                            metrics_avg["answer_correctness"] * 0.1
                        )
                        overall_score = safe_val(overall_score)
                        num_chunks = len(chunks)
                        avg_chunk_len = int(sum(len(c) for c in chunks) / num_chunks) if num_chunks > 0 else 0
                        required_keys = [
                            "overall_score", "faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness", "avg_chunk_len", "num_chunks"
                        ]
                        metrics_per_qa = []
                        total = len(eval_results)
                        for idx, res in enumerate(eval_results):
                            print(f"[進捗] 評価 {idx+1}/{total} 件目を処理中...")  # 進捗ログを出力
                            scores = res.scores if hasattr(res, "scores") else {}
                            # scoresがlist型ならdictに変換（防御的処理）
                            if isinstance(scores, list):
                                if len(scores) > 0 and isinstance(scores[0], dict):
                                    scores = scores[0]
                                else:
                                    scores = {}
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
                            for k in metrics_keys:
                                if k in scores and isinstance(scores[k], (float, int)):
                                    metrics_sum[k] += float(scores[k])
                                    metrics_count[k] += 1
                        metrics_avg = {k: safe_val(metrics_sum[k] / metrics_count[k] if metrics_count[k] > 0 else 0.0) for k in metrics_keys}
                        overall_score = (
                            metrics_avg["answer_relevancy"] * 0.25 +
                            metrics_avg["faithfulness"] * 0.25 +
                            metrics_avg["context_precision"] * 0.2 +
                            metrics_avg["context_recall"] * 0.2 +
                            metrics_avg["answer_correctness"] * 0.1
                        )
                        overall_score = safe_val(overall_score)
                        num_chunks = len(chunks)
                        avg_chunk_len = int(sum(len(c) for c in chunks) / num_chunks) if num_chunks > 0 else 0
                        required_keys = [
                            "overall_score", "faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness", "avg_chunk_len", "num_chunks"
                        ]
                        
                        # チャンク戦略を設定
                        if chunk_method == "semantic":
                            chunk_strategy = "semantic"
                        else:
                            chunk_size_val = chunk_sizes[i] if i < len(chunk_sizes) else chunk_sizes[0]
                            chunk_overlap_val = chunk_overlaps[i] if i < len(chunk_overlaps) else chunk_overlaps[0]
                            chunk_strategies = data.get("chunk_strategies", []) if isinstance(data, dict) else []
                            if chunk_strategies and i < len(chunk_strategies):
                                chunk_strategy = chunk_strategies[i]
                            else:
                                chunk_strategy = f"{chunk_method}-{chunk_size_val}-{chunk_overlap_val}"
                        
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
                        for k in required_keys:
                            if k not in response_dict:
                                response_dict[k] = 0.0
                        results.append(response_dict)
                    except Exception as e:
                        # エラー時も必ずエラー内容を返す
                        import traceback
                        traceback.print_exc()
                        results.append({"error": str(e), "input_data": data})
                return results
            except Exception as e:
                # エラー時も必ずエラー内容を返す
                import traceback
                traceback.print_exc()
                return {"error": str(e), "input_data": data}

        # --- 本体分岐 ---
        if isinstance(data, list):
            results = []
            for d in data:
                try:
                    if not isinstance(d, dict):
                        d = find_first_dict(d)
                    res = await evaluate_one_bulk(d)
                    results.append(res)
                except Exception as e:
                    # 個別データでエラーが発生しても全体を止めず、エラー内容を追加
                    import traceback
                    traceback.print_exc()
                    results.append({"error": str(e), "input_data": d})
            return results
        else:
            return await evaluate_one_bulk(data)
    except Exception as e:
        # 異常時も辞書を直接返す（JSONResponse不使用）
        print(f"[重要] bulk_evaluate全体例外: {e}")
        return {"error": str(e)}

# --- PDFアップロード＆QA自動生成API ---
from fastapi import UploadFile, File

@app.post("/uploadfile/")
async def uploadfile(file: UploadFile = File(...)):
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

        # 2. LLMで質問セット自動生成
        print("[重要] LLM質問生成開始")
        llm_instance = get_llm("openai")
        prompt_q = f"""
以下の内容に関する代表的な質問を日本語で5つ作成してください。\n---\n{text[:1500]}\n---\n質問：
"""
        try:
            questions_resp = llm_instance.invoke(prompt_q)
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
            # 3. LLMで回答セット自動生成
            print("[重要] LLM回答生成開始")
            answers = []
            for i, q in enumerate(questions):
                try:
                    prompt_a = f"""
以下の内容に基づいて、次の質問に日本語で簡潔に答えてください。\n---\n{sample_text}\n---\n質問: {q}\n回答：
"""
                    answer_resp = llm_instance.invoke(prompt_a)
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

        # 4. 結果を辞書形式で返却（正常時は全キー、JSONResponseは使わない）
        print(f"[重要] API返却直前: {len(questions)}質問, {len(answers)}回答")
        for i, (q, a) in enumerate(zip(questions, answers)):
            print(f"[重要] Q{i+1}: {q}")
            print(f"[重要] A{i+1}: {a}")
        
        # dictを直接返す（JSONResponse不使用）
        return {
            "text": sample_text,
            "questions": questions,
            "answers": answers
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
        print(f"[DEBUG] models.yamlのmodelsキー: {models_dict['models']}")
        return JSONResponse(content={"models": models_dict["models"]})
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
