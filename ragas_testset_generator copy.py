"""
RAGASのテストセットをOllamaのGPTモデルを使用して生成するスクリプト

このスクリプトは、RAG（Retrieval-Augmented Generation）システムの評価用に
テストセットを生成します。Ollamaで実行可能なGPTモデルを使用します。

主な機能:
- ドキュメントからのテストケース生成
- 複数の質問タイプのサポート
- カスタマイズ可能なパラメータ
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import random
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 環境変数の読み込み
from dotenv import load_dotenv
load_dotenv()

# デフォルトの設定
DEFAULT_MODEL_NAME = "gpt-oss:20b"  # Ollamaで使用するデフォルトのモデル名
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_QUESTIONS = 5
DEFAULT_OUTPUT_FILE = "ragas_testset.jsonl"

@dataclass
class TestExample:
    """テストケースを表すデータクラス"""
    question: str
    ground_truth: str
    context: List[str]
    question_type: str = "single_hop"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "question": self.question,
            "ground_truth": self.generate_ground_truth(),
            "contexts": self.context,
            "question_type": self.question_type,
            "metadata": self.metadata or {}
        }
    
    def generate_ground_truth(self) -> str:
        """RAGAS用のground_truthを生成"""
        return self.ground_truth

class QuestionGenerator:
    """Ollamaを使用して質問を生成するクラス"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, temperature: float = DEFAULT_TEMPERATURE):
        """初期化"""
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            num_ctx=4096  # コンテキストウィンドウのサイズを大きく設定
        )
        self.parser = JsonOutputParser(pydantic_object=QuestionGenerationOutput)
    
    def generate_questions(
        self, 
        document: str, 
        num_questions: int = 5, 
        question_types: List[str] = None
    ) -> List[TestExample]:
        """ドキュメントから質問を生成"""
        if question_types is None:
            question_types = ["single_hop"]
        
        # プロンプトテンプレートの作成
        system_prompt = """あなたはドキュメントから高品質な質問を生成するAIアシスタントです。
        与えられたドキュメントの内容に基づいて、明確で具体的な質問を生成してください。
        質問はドキュメントの内容を正確に反映している必要があります。
        """
        
        human_prompt = """
        以下のドキュメントから、{num_questions}つの質問を生成してください。
        各質問は、ドキュメントの内容を正確に反映している必要があります。
        質問の種類: {question_types}
        
        ドキュメント:
        {document}
        
        出力形式:
        ```json
        {{
            "questions": [
                {{
                    "question": "生成された質問",
                    "answer": "質問に対する回答",
                    "context": ["回答の根拠となるドキュメントの一部"],
                    "question_type": "質問のタイプ"
                }}
            ]
        }}
        ```
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        # チェーンを作成
        chain = prompt | self.llm
        
        # 質問を生成
        response = chain.invoke({
            "document": document,
            "num_questions": num_questions,
            "question_types": ", ".join(question_types)
        })
        
        # 応答をパース
        try:
            # JSON部分を抽出
            json_str = response.strip().split('```json')[1].split('```')[0].strip()
            result = json.loads(json_str)
            
            # TestExampleオブジェクトに変換
            examples = []
            for q in result.get("questions", []):
                example = TestExample(
                    question=q.get("question", ""),
                    ground_truth=q.get("answer", ""),
                    context=q.get("context", []),
                    question_type=q.get("question_type", "single_hop"),
                    metadata={
                        "generated_by": "ollama",
                        "model": self.llm.model,
                    }
                )
                examples.append(example)
            
            return examples
            
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            return []

class QuestionGenerationOutput(BaseModel):
    """質問生成の出力スキーマ"""
    questions: List[Dict[str, Any]] = Field(
        ...,
        description="生成された質問のリスト"
    )

def load_document(file_path: str) -> str:
    """ドキュメントを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading document: {e}")
        return ""

def save_testset(examples: List[TestExample], output_file: str):
    """テストセットをファイルに保存"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved {len(examples)} examples to {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RAGASテストセット生成ツール')
    parser.add_argument('--input', '-i', type=str, required=True, help='入力ドキュメントのパス')
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT_FILE, help='出力ファイルのパス')
    parser.add_argument('--num-questions', '-n', type=int, default=DEFAULT_NUM_QUESTIONS, help='生成する質問の数')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL_NAME, help='使用するOllamaモデル名')
    parser.add_argument('--temperature', '-t', type=float, default=DEFAULT_TEMPERATURE, help='生成のランダム性を制御するパラメータ')
    
    args = parser.parse_args()
    
    # ドキュメントを読み込み
    print(f"Loading document: {args.input}")
    document = load_document(args.input)
    if not document:
        print("Error: Failed to load document")
        return
    
    # 質問生成器を初期化
    print(f"Initializing question generator with model: {args.model}")
    generator = QuestionGenerator(
        model_name=args.model,
        temperature=args.temperature
    )
    
    # 質問を生成
    print(f"Generating {args.num_questions} questions...")
    examples = generator.generate_questions(
        document=document,
        num_questions=args.num_questions,
        question_types=["single_hop", "multi_hop", "reasoning"]
    )
    
    # 結果を保存
    if examples:
        save_testset(examples, args.output)
        print("Test set generation completed successfully!")
    else:
        print("No questions were generated. Please check the input and try again.")

if __name__ == "__main__":
    main()
