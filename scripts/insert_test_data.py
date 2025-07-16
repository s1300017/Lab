from main import SessionLocal, text, engine
import os
from datetime import datetime

def jst_now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')

def ensure_table_exists():
    print(f"[{jst_now_str()}] [INFO] テーブルの存在確認を開始します")
    with engine.connect() as conn:
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
            conn.commit()
            print(f"[{jst_now_str()}] [INFO] embeddingsテーブルを作成しました")
        else:
            print(f"[{jst_now_str()}] [INFO] embeddingsテーブルは既に存在します")

def insert_test_data():
    print(f"[{jst_now_str()}] [INFO] テストデータの挿入を開始します")
    
    # テーブルの存在を確認し、必要に応じて作成
    ensure_table_exists()
    
    # テスト用のデータを挿入
    with SessionLocal() as session:
        # 既存のテストデータを削除
        session.execute(text("DELETE FROM embeddings WHERE chunk_strategy = 'test'"))
        session.commit()
        
        # テスト用のテキストを挿入
        test_text = '''これはテスト用のドキュメントです。
        こんにちは、RAGシステムのテストをしています。
        このテキストはベクトル検索のテスト用です。'''
        
        # テキストをチャンクに分割（簡易的に改行で分割）
        chunks = [c for c in test_text.split('。') if c.strip()]
        
        # 各チャンクをベクトル化して保存（簡易版）
        for i, chunk in enumerate(chunks):
            # ここでは簡易的にテキストをそのまま保存
            session.execute(
                text('''
                    INSERT INTO embeddings 
                    (text, embedding_model, chunk_strategy, chunk_size, chunk_overlap, avg_chunk_len, num_chunks)
                    VALUES 
                    (:text, 'huggingface_bge_small', 'test', 1000, 0, :avg_len, :num_chunks)
                '''),
                {'text': chunk, 'avg_len': len(chunk), 'num_chunks': len(chunks)}
            )
        session.commit()
        print(f"[{jst_now_str()}] [INFO] テストデータを{len(chunks)}件挿入しました")

if __name__ == "__main__":
    try:
        insert_test_data()
        print(f"[{jst_now_str()}] [SUCCESS] テストデータの挿入が完了しました")
    except Exception as e:
        print(f"[{jst_now_str()}] [ERROR] エラーが発生しました: {str(e)}")
        import traceback
        print(traceback.format_exc())
        exit(1)
