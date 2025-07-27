# 評価履歴管理API
from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text
import os
import json
from datetime import datetime
from pytz import timezone

# データベース接続設定
POSTGRES_DB = os.environ.get("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "rag_password")
DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@db:5432/{POSTGRES_DB}"
engine = create_engine(DB_URL)

router = APIRouter()

def jst_now_str():
    return datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')

@router.get("/experiments/")
def get_experiments():
    """
    実験履歴一覧を取得するAPI
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    id, session_id, experiment_name, file_name, 
                    status, total_combinations, completed_combinations,
                    created_at, updated_at
                FROM experiments 
                ORDER BY created_at DESC
                LIMIT 100
            """))
            
            experiments = []
            for row in result:
                experiments.append({
                    "id": row[0],
                    "session_id": row[1],
                    "experiment_name": row[2],
                    "file_name": row[3],
                    "status": row[4],
                    "total_combinations": row[5],
                    "completed_combinations": row[6],
                    "created_at": row[7].isoformat() if row[7] else None,
                    "updated_at": row[8].isoformat() if row[8] else None
                })
            
            return {"experiments": experiments}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"実験履歴取得エラー: {str(e)}")

@router.get("/experiments/{experiment_id}/results/")
def get_experiment_results(experiment_id: int):
    """
    特定の実験の詳細結果を取得するAPI
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    e.id, e.embedding_model, e.chunk_strategy, e.chunk_size, e.chunk_overlap,
                    e.avg_chunk_len, e.num_chunks, e.overall_score, e.faithfulness, 
                    e.answer_relevancy, e.context_recall, e.context_precision, 
                    e.answer_correctness, e.created_at,
                    exp.experiment_name, exp.file_name, exp.parameters
                FROM embeddings e
                JOIN experiments exp ON e.experiment_id = exp.id
                WHERE e.experiment_id = :experiment_id
                ORDER BY e.overall_score DESC
            """), {"experiment_id": experiment_id})
            
            results = []
            for row in result:
                results.append({
                    "id": row[0],
                    "embedding_model": row[1],
                    "chunk_strategy": row[2],
                    "chunk_size": row[3],
                    "chunk_overlap": row[4],
                    "avg_chunk_len": row[5],
                    "num_chunks": row[6],
                    "overall_score": row[7],
                    "faithfulness": row[8],
                    "answer_relevancy": row[9],
                    "context_recall": row[10],
                    "context_precision": row[11],
                    "answer_correctness": row[12],
                    "created_at": row[13].isoformat() if row[13] else None,
                    "experiment_name": row[14],
                    "file_name": row[15],
                    "parameters": json.loads(row[16]) if row[16] else {}
                })
            
            return {"results": results}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"実験結果取得エラー: {str(e)}")

@router.delete("/experiments/{experiment_id}/")
def delete_experiment(experiment_id: int):
    """
    実験とその結果を削除するAPI
    """
    try:
        with engine.connect() as conn:
            with conn.begin():
                # 関連する評価結果を削除
                conn.execute(text("DELETE FROM embeddings WHERE experiment_id = :experiment_id"), 
                           {"experiment_id": experiment_id})
                
                # 実験を削除
                result = conn.execute(text("DELETE FROM experiments WHERE id = :experiment_id"), 
                                    {"experiment_id": experiment_id})
                
                if result.rowcount == 0:
                    raise HTTPException(status_code=404, detail="実験が見つかりません")
                
                return {"message": f"実験ID {experiment_id} を削除しました"}
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"実験削除エラー: {str(e)}")

@router.get("/experiments/statistics/")
def get_experiment_statistics():
    """
    実験統計情報を取得するAPI
    """
    try:
        with engine.connect() as conn:
            # 総実験数
            total_experiments = conn.execute(text("SELECT COUNT(*) FROM experiments")).scalar()
            
            # 総評価結果数
            total_results = conn.execute(text("SELECT COUNT(*) FROM embeddings")).scalar()
            
            # モデル別統計
            model_stats = conn.execute(text("""
                SELECT embedding_model, COUNT(*) as count, AVG(overall_score) as avg_score
                FROM embeddings 
                GROUP BY embedding_model 
                ORDER BY avg_score DESC
            """))
            
            model_statistics = []
            for row in model_stats:
                model_statistics.append({
                    "model": row[0],
                    "count": row[1],
                    "avg_score": float(row[2]) if row[2] else 0.0
                })
            
            # チャンク戦略別統計
            chunk_stats = conn.execute(text("""
                SELECT chunk_strategy, COUNT(*) as count, AVG(overall_score) as avg_score
                FROM embeddings 
                GROUP BY chunk_strategy 
                ORDER BY avg_score DESC
            """))
            
            chunk_statistics = []
            for row in chunk_stats:
                chunk_statistics.append({
                    "strategy": row[0],
                    "count": row[1],
                    "avg_score": float(row[2]) if row[2] else 0.0
                })
            
            return {
                "total_experiments": total_experiments,
                "total_results": total_results,
                "model_statistics": model_statistics,
                "chunk_statistics": chunk_statistics
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"統計情報取得エラー: {str(e)}")
