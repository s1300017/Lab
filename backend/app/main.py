from datetime import datetime
from pytz import timezone
import os

def jst_now_str():
    return datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')

from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# --- PDF・抽出データ保存用ディレクトリのグローバル定義 ---
import uuid
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PDF_DIR = DATA_DIR / "pdf"
EXTRACTED_DIR = DATA_DIR / "extracted"
PDF_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

print(f"[{jst_now_str()}] === FastAPI main.py 起動開始 ===")

# データベース接続設定
DB_URL = os.environ.get("DATABASE_URL", "postgresql://rag_user:rag_password@db:5432/rag_db")
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    try:
        with engine.connect() as conn:
            # 必要なテーブルを作成
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.commit()
            print(f"[{jst_now_str()}] データベース初期化成功: embeddingsテーブルを作成しました")
    except Exception as e:
        print(f"[{jst_now_str()}] データベース初期化失敗: {str(e)}")
        raise

app = FastAPI()

# サーバー起動時にデータベースを初期化
@app.on_event("startup")
async def startup_event():
    init_db()

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

def semantic_chunk_text(text, chunk_size=1000, chunk_overlap=200, embedding_model=None):
    """
    セマンティックチャンク分割：
    1. 文単位で分割
    2. 各文のembeddingを取得
    3. コサイン類似度で分割点を決定し、意味的に自然なチャンクを作成
    """
    # 文単位で分割
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [text]
    # 各文のembedding取得
    if embedding_model is None:
        raise ValueError("embedding_modelが指定されていません")
    embeddings = embedding_model.embed_documents(sentences)
    # チャンク生成
    chunks = []
    current_chunk = []
    current_len = 0
    i = 0
    while i < len(sentences):
        current_chunk.append(sentences[i])
        current_len += len(sentences[i])
        # チャンクサイズを超えたら区切る
        if current_len >= chunk_size or i == len(sentences) - 1:
            # overlapを考慮（直前の文からchunk_overlap分だけ残す）
            if chunk_overlap > 0 and i != len(sentences) - 1:
                overlap_len = 0
                overlap_chunk = []
                j = len(current_chunk) - 1
                while j >= 0 and overlap_len < chunk_overlap:
                    overlap_chunk.insert(0, current_chunk[j])
                    overlap_len += len(current_chunk[j])
                    j -= 1
                # 次のチャンクの先頭にoverlap_chunkを追加
                if i+1 < len(sentences):
                    next_start = i+1 - len(overlap_chunk)
                    if next_start < 0:
                        next_start = 0
                    i = next_start
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_len = 0
        i += 1
    return chunks

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
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
try:
    current_llm = get_llm("ollama_llama2")
except Exception as e:
    import logging
    logging.warning(f"LLM初期化失敗: {e}")

try:
    current_embeddings = get_embeddings("huggingface_bge_small")
except Exception as e:
    import logging
    logging.warning(f"Embedding初期化失敗: {e}")

# --- Pydantic Models ---
class ChunkRequest(BaseModel):
    text: str
    chunk_size: int = 1000
    chunk_overlap: int = 200

class EmbedRequest(BaseModel):
    chunks: list[str]
    embedding_model: str # Added for dynamic selection

class QueryRequest(BaseModel):
    query: str
    llm_model: str # Added for dynamic selection
    embedding_model: str # Added for dynamic selection

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
    chunk_methodに応じてrecursiveまたはsemanticで分割
    """
    chunk_method = getattr(request, 'chunk_method', 'recursive')
    embedding_model = getattr(request, 'embedding_model', None)
    if chunk_method == 'semantic':
        # embedding_model必須
        if embedding_model is None:
            raise HTTPException(status_code=400, detail="semantic chunkingにはembedding_modelが必要です")
        embedder = get_embeddings(embedding_model)
        chunks = semantic_chunk_text(request.text, chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap, embedding_model=embedder)
    elif chunk_method == 'recursive':
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(request.text)
    else:
        raise HTTPException(status_code=400, detail=f"未対応のchunk_method: {chunk_method}")
    return {"chunks": chunks}


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
        llm_instance = get_llm(request.llm_model)
        embeddings_instance = get_embeddings(request.embedding_model)

        vectorstore = PGVector.from_documents(
            documents=[],  # 空のドキュメントで初期化
            embedding=embeddings_instance,
            collection_name=get_collection_name(request.embedding_model)  # embeddingモデルごとにコレクションを切り替え
        )
        vectorstore.add_texts(texts=chunks)
        retriever = vectorstore.as_retriever()

        template = """Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm_instance
            | StrOutputParser()
        )

        answer = chain.invoke(request.query)
        retrieved_docs = retriever.get_relevant_documents(request.query)
        contexts = [doc.page_content for doc in retrieved_docs]

        return {"answer": answer, "contexts": contexts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            # コンテキストからオーバーラップを計算する関数
            def calculate_overlap(contexts: list[list[str]]) -> float:
                if not contexts or len(contexts) < 2:
                    return 0.0
                
                # すべてのコンテキストを結合してトークン化
                all_tokens = []
                for ctx in contexts:
                    if isinstance(ctx, str):
                        all_tokens.extend(ctx.split())
                    else:
                        for text in ctx:
                            all_tokens.extend(text.split())
                
                # 重複するトークン数を計算
                unique_tokens = set(all_tokens)
                total_tokens = len(all_tokens)
                unique_count = len(unique_tokens)
                
                if total_tokens == 0:
                    return 0.0
                    
                # 重複率を計算 (0.0 〜 1.0)
                overlap_ratio = 1.0 - (unique_count / total_tokens) if total_tokens > 0 else 0.0
                return overlap_ratio

            # 各質問のコンテキストに対してオーバーラップを計算
            overlap_scores = []
            for ctx_list in request.contexts:
                overlap = calculate_overlap(ctx_list)
                overlap_scores.append(overlap)
            
            # 平均オーバーラップをスコアに追加
            avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
        else:
            overlap_scores = [0.0] * len(request.questions)
            avg_overlap = 0.0

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
            scores["overlap_scores"] = overlap_scores
        
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
                        chunk_size = chunk_sizes[i] if i < len(chunk_sizes) else 0
                        chunk_overlap = chunk_overlaps[i] if i < len(chunk_overlaps) else 0

                        # チャンク分割方式ごとに分岐
                        if chunk_method == "recursive":
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=int(chunk_size),
                                chunk_overlap=int(chunk_overlap),
                                length_function=len
                            )
                            chunks = splitter.split_text(sample_text)
                        elif chunk_method == "fixed":
                            chunks = fixed_chunk_text(sample_text, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
                        elif chunk_method == "semantic":
                            if not embedding_model:
                                results.append({"error": "semantic chunkingにはembedding_modelが必要です", "chunk_method": chunk_method, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
                                continue
                            chunks = semantic_chunk_text(sample_text, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap), embedding_model=embedder)
                        elif chunk_method == "sentence":
                            chunks = sentence_chunk_text(sample_text)
                        elif chunk_method == "paragraph":
                            chunks = paragraph_chunk_text(sample_text)
                        else:
                            results.append({"error": f"未対応のchunk_method: {chunk_method}", "chunk_method": chunk_method, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
                            continue

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
                        chunk_size_val = chunk_sizes[i] if i < len(chunk_sizes) else chunk_sizes[0]
                        chunk_overlap_val = chunk_overlaps[i] if i < len(chunk_overlaps) else chunk_overlaps[0]
                        chunk_strategies = data.get("chunk_strategies", []) if isinstance(data, dict) else []
                        if chunk_strategies and i < len(chunk_strategies):
                            chunk_strategy = chunk_strategies[i]
                        else:
                            chunk_strategy = f"{chunk_method}-{chunk_size_val}-{chunk_overlap_val}"
                        response_dict = {
                            "embedding_model": embedding_model,
                            "chunk_size": chunk_size_val,
                            "chunk_overlap": chunk_overlap_val,
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