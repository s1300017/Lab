"""
RAGASを使用してRAGシステムを評価するスクリプト
Ollamaのgpt-oss:20bモデルと埋め込みモデルを使用
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# RAGAS関連のインポート
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas import RunConfig

# LangChain関連のインポート
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 環境変数の読み込み
load_dotenv()

@dataclass
class RAGASEvaluator:
    """RAGシステムをRAGASで評価するクラス"""
    
    def __init__(
        self,
        model_name: str = "gpt-oss:20b",
        embedding_model_name: str = "nomic-embed-text",
        temperature: float = 0.1,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """初期化"""
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # モデルを初期化
        self.llm = Ollama(
            model=model_name,
            temperature=temperature
        )
        
        # 埋め込みモデルを初期化
        self.embeddings = OllamaEmbeddings(
            model=embedding_model_name
        )
        
        # テキスト分割器を初期化
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # ベクトルストア
        self.vectorstore = None
    
    def load_documents(self, file_path: str) -> List[Document]:
        """ドキュメントを読み込む"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # ドキュメントをチャンクに分割
        docs = self.text_splitter.create_documents([text])
        return docs
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """ベクトルストアを作成"""
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
    
    def retrieve_contexts(self, query: str, k: int = 3) -> List[str]:
        """クエリに関連するコンテキストを取得"""
        if self.vectorstore is None:
            raise ValueError("ベクトルストアが初期化されていません。先にcreate_vectorstoreを実行してください。")
        
        # 類似ドキュメントを検索
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def evaluate_rag_system(
        self,
        test_data_path: str,
        output_dir: str = "evaluation_results",
        k: int = 3
    ) -> Dict[str, Any]:
        """RAGシステムを評価"""
        # テストデータを読み込み
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        
        # 評価結果を保存するディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 評価用データを準備
        questions = []
        ground_truths = []
        contexts_list = []
        model_answers = []
        
        print("評価を開始します...")
        for item in tqdm(test_data, desc="評価を準備中"):
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            # コンテキストを取得
            if "contexts" in item and item["contexts"]:
                contexts = item["contexts"][:k]  # 最大k個のコンテキストを使用
            else:
                # ベクトルストアからコンテキストを取得
                contexts = self.retrieve_contexts(question, k=k)
            
            # モデルに回答を生成させる
            context_text = "\n\n".join(contexts)
            prompt = f"""以下のコンテキストに基づいて質問に答えてください。
            
            コンテキスト:
            {context_text}
            
            質問: {question}
            回答: """
            
            model_answer = self.llm(prompt).strip()
            
            questions.append(question)
            ground_truths.append(ground_truth)
            contexts_list.append(contexts)
            model_answers.append(model_answer)
        
        # 評価を実行
        print("評価を実行中...")
        from datasets import Dataset
        
        # データフレームを作成
        # answer_correctnessメトリックには'reference'カラムが必要
        df = pd.DataFrame({
            "question": questions,
            "answer": model_answers,  # モデルが生成した回答
            "reference": ground_truths,  # 正解回答
            "contexts": contexts_list
        })
        
        # Datasetオブジェクトに変換
        dataset = Dataset.from_pandas(df)
        
        # 評価設定をカスタマイズ
        run_config = RunConfig(
            timeout=300,  # タイムアウトを300秒（5分）に設定
            max_workers=2  # 並列ワーカー数を減らす
        )

        # 評価を実行
        result = evaluate(
            dataset,
            metrics=[
                answer_relevancy,
                answer_correctness,
                answer_similarity,
                context_precision,
                context_recall,
                faithfulness,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
            run_config=run_config  # カスタム設定を適用
        )
        
        # 結果を保存
        output_file = os.path.join(output_dir, "evaluation_results.json")
        result.to_pandas().to_json(output_file, orient="records", force_ascii=False, indent=2)
        print(f"評価結果を {output_file} に保存しました。")
        
        return result

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RAGシステムをRAGASで評価")
    parser.add_argument("--test-data", "-t", type=str, required=True, help="テストデータのパス")
    parser.add_argument("--documents", "-d", type=str, required=True, help="評価対象のドキュメントのパス")
    parser.add_argument("--output-dir", "-o", type=str, default="evaluation_results", help="評価結果の出力ディレクトリ")
    parser.add_argument("--model", "-m", type=str, default="gpt-oss:20b", help="使用するOllamaモデル名")
    parser.add_argument("--embedding-model", "-e", type=str, default="nomic-embed-text", help="使用する埋め込みモデル名")
    parser.add_argument("--chunk-size", type=int, default=1000, help="ドキュメントのチャンクサイズ")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="チャンク間のオーバーラップサイズ")
    parser.add_argument("--top-k", type=int, default=3, help="取得するコンテキストの数")
    
    args = parser.parse_args()
    
    # 評価器を初期化
    print("評価器を初期化しています...")
    evaluator = RAGASEvaluator(
        model_name=args.model,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # ドキュメントを読み込んでベクトルストアを作成
    print("ドキュメントを読み込んでいます...")
    documents = evaluator.load_documents(args.documents)
    evaluator.create_vectorstore(documents)
    
    # 評価を実行
    print("評価を開始します...")
    result = evaluator.evaluate_rag_system(
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        k=args.top_k
    )
    
    # 結果を表示
    print("\n評価結果:")
    print(result)

if __name__ == "__main__":
    main()