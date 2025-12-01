from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import json
import os
import logging

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pytz import timezone
from sqlalchemy import create_engine, text
try:
    import yaml
except ImportError:
    yaml = None  # PyYAML が未導入の場合

from .settings import DB_URL, DATA_DIR, PDF_DIR, EXTRACTED_DIR, IMAGES_DIR, DEFAULT_LLM_NAME, DEFAULT_EMBEDDING_MODEL


def jst_now_str() -> str:
    return datetime.now(timezone("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S JST")


logger = logging.getLogger(__name__)


# データベース接続設定（main.py と同等の環境変数を利用）
engine = create_engine(DB_URL)

# モデルパスなど（main.py と同一設定）
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LOCAL_MODEL_PATH = Path("/app/models/BAAI_bge-small-en-v1.5")

# models.yaml / strategies.yaml 用のパス
MODELS_YAML_PATH = Path("/app/models.yaml")
STRATEGIES_YAML_PATH = Path("/app/strategies.yaml")


def load_models_yaml():
    """models.yaml を読み込んで dict を返すヘルパ。"""

    if yaml is None:
        raise RuntimeError(
            "PyYAMLがインストールされていません。requirements.txtに 'pyyaml' を追加してください。",
        )
    if not MODELS_YAML_PATH.exists():
        raise FileNotFoundError(f"models.yamlが見つかりません: {MODELS_YAML_PATH}")
    with MODELS_YAML_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_strategies_yaml():
    """strategies.yaml を読み込んで dict を返すヘルパ。"""

    if yaml is None:
        raise RuntimeError(
            "PyYAMLがインストールされていません。requirements.txtに 'pyyaml' を追加してください。",
        )
    if not STRATEGIES_YAML_PATH.exists():
        raise FileNotFoundError(
            f"strategies.yamlが見つかりません: {STRATEGIES_YAML_PATH}",
        )
    with STRATEGIES_YAML_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# データディレクトリ（main.py と同じ構成）


router = APIRouter()


@router.get("/models/")
def get_available_models() -> dict[str, Any]:
    """利用可能なモデルと現在のモデル状態を返す管理用API。"""

    model_exists = LOCAL_MODEL_PATH.exists()
    model_info = {
        "model_name": str(MODEL_NAME),
        "local_path": str(LOCAL_MODEL_PATH),
        "exists": model_exists,
        "size_mb": (
            sum(f.stat().st_size for f in LOCAL_MODEL_PATH.glob("**/*") if f.is_file())
            / (1024 * 1024)
        )
        if model_exists
        else 0,
    }

    return {
        "llm_models": [DEFAULT_LLM_NAME, "gpt-4o"],
        "embedding_models": [DEFAULT_EMBEDDING_MODEL, "gpt-4o"],
        "current_embedding_model": {
            "name": DEFAULT_EMBEDDING_MODEL,
            "type": "local" if model_exists else "remote",
            "info": model_info,
        },
        "environment": {
            "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", "Not set"),
            "hf_home": os.environ.get("HF_HOME", "Not set"),
        },
    }


@router.get("/list_models")
def list_models() -> JSONResponse:
    """models.yaml の内容をカテゴリー別に返すAPI。"""

    import os

    try:
        # デバッグ用: カレントディレクトリとファイル一覧をログ出力
        logger.debug("[DEBUG] os.getcwd() = %s", os.getcwd())
        logger.debug("[DEBUG] os.listdir('.') = %s", os.listdir("."))
        abs_path = os.path.abspath("models.yaml")
        logger.debug("[DEBUG] models.yaml abs path = %s", abs_path)
        logger.debug("[DEBUG] models.yaml exists = %s", os.path.exists(abs_path))

        models_dict = load_models_yaml()
        logger.debug("[DEBUG] models_dict loaded: %s", models_dict)
        if not models_dict or "models" not in models_dict:
            logger.error("[list_models ERROR] models.yamlに'models'キーがありません")
            return JSONResponse(
                status_code=404,
                content={"error": "models.yamlに'models'キーがありません"},
            )

        categorized_models = {
            "LLM": [m for m in models_dict["models"] if m.get("category") == "LLM"],
            "Embedding": [
                m
                for m in models_dict["models"]
                if m.get("category") == "Embedding"
            ],
        }

        logger.debug("[DEBUG] categorized_models: %s", categorized_models)
        return JSONResponse(content=categorized_models)
    except Exception as e:  # noqa: BLE001
        import traceback

        logger.error("[list_models ERROR] %s", e)
        logger.debug(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/list_strategies")
def list_strategies() -> JSONResponse:
    """strategies.yaml の内容を返すAPI。"""

    try:
        strategies = load_strategies_yaml()
        return JSONResponse(content=strategies)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/admin/reset_vectors_and_chunks/")
def admin_reset_vectors_and_chunks() -> dict[str, Any]:
    try:
        logger.warning("[%s] [WARN] admin_reset_vectors_and_chunks called", jst_now_str())
        with engine.begin() as conn:
            try:
                conn.execute(text("DELETE FROM langchain_pg_embedding;"))
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[%s] [WARN] failed to delete from langchain_pg_embedding: %s",
                    jst_now_str(),
                    e,
                )
            try:
                conn.execute(text("DELETE FROM langchain_pg_collection;"))
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[%s] [WARN] failed to delete from langchain_pg_collection: %s",
                    jst_now_str(),
                    e,
                )
            try:
                conn.execute(text("DELETE FROM pdf_chunks;"))
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[%s] [WARN] failed to delete from pdf_chunks: %s",
                    jst_now_str(),
                    e,
                )

        return {
            "status": "success",
            "message": "langchain_pg_embedding, langchain_pg_collection, pdf_chunks の全レコードを削除しました（pdf_files とPDF本体は維持されます）。",
        }
    except Exception as e:  # noqa: BLE001
        logger.error("[%s] [ERROR] admin_reset_vectors_and_chunks failed: %s", jst_now_str(), e)
        return {
            "status": "error",
            "message": f"admin_reset_vectors_and_chunks でエラーが発生しました: {str(e)}",
        }


@router.post("/admin/reset_pdfs_and_vectors/")
def admin_reset_pdfs_and_vectors() -> dict[str, Any]:
    try:
        logger.warning("[%s] [WARN] admin_reset_pdfs_and_vectors called", jst_now_str())
        pdf_rows: list[Any] = []
        with engine.begin() as conn:
            try:
                pdf_rows = conn.execute(
                    text("SELECT id, storage_path FROM pdf_files;")
                ).fetchall()
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] [WARN] failed to select from pdf_files: %s", jst_now_str(), e)
                pdf_rows = []

            for table_name in [
                "pdf_chunks",
                "generated_questions",
                "experiment_results",
                "experiments",
            ]:
                try:
                    conn.execute(text(f"DELETE FROM {table_name};"))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[%s] [WARN] failed to delete from %s: %s",
                        jst_now_str(),
                        table_name,
                        e,
                    )

            for table_name in [
                "langchain_pg_embedding",
                "langchain_pg_collection",
            ]:
                try:
                    conn.execute(text(f"DELETE FROM {table_name};"))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[%s] [WARN] failed to delete from %s: %s",
                        jst_now_str(),
                        table_name,
                        e,
                    )

            try:
                conn.execute(text("DELETE FROM pdf_files;"))
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] [WARN] failed to delete from pdf_files: %s", jst_now_str(), e)

        deleted_files = 0
        deleted_extracted = 0
        for row in pdf_rows:
            try:
                file_id = row[0]
                storage_path = row[1]
            except Exception:  # noqa: BLE001
                try:
                    file_id = getattr(row, "id", None)
                    storage_path = getattr(row, "storage_path", None)
                except Exception:  # noqa: BLE001
                    file_id = None
                    storage_path = None

            if storage_path:
                try:
                    p = Path(storage_path)
                    if p.exists():
                        p.unlink()
                        deleted_files += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[%s] [WARN] failed to delete storage file %s: %s",
                        jst_now_str(),
                        storage_path,
                        e,
                    )

            if file_id:
                try:
                    extracted_path = EXTRACTED_DIR / f"{file_id}.json"
                    if extracted_path.exists():
                        extracted_path.unlink()
                        deleted_extracted += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[%s] [WARN] failed to delete extracted json %s: %s",
                        jst_now_str(),
                        file_id,
                        e,
                    )

        return {
            "status": "success",
            "message": "PDF関連レコード（pdf_files, pdf_chunks など）および PGVector ベクトルストアを削除し、PDFファイルと抽出JSONも削除しました。",
            "deleted_pdf_rows": len(pdf_rows),
            "deleted_storage_files": deleted_files,
            "deleted_extracted_json": deleted_extracted,
        }
    except Exception as e:  # noqa: BLE001
        logger.error("[%s] [ERROR] admin_reset_pdfs_and_vectors failed: %s", jst_now_str(), e)
        return {
            "status": "error",
            "message": f"admin_reset_pdfs_and_vectors でエラーが発生しました: {str(e)}",
        }


@router.post("/admin/wipe_all")
def admin_wipe_all(payload: dict) -> JSONResponse:
    """DBと /app/data 配下のPDF・抽出JSON・画像をバックアップ付きで全消去する危険操作API。"""
    try:
        confirm = bool(payload.get("confirm"))
        make_backup = bool(payload.get("backup", True))
        delete_files = bool(payload.get("delete_files", True))
        if not confirm:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "confirm=true が必須です（破壊的操作の安全装置）",
            })

        # --- バックアップの作成 ---
        backup_info: dict[str, Any] = {}
        if make_backup:
            import shutil
            import zipfile
            import datetime as _dt

            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backups_dir = DATA_DIR / "backups" / ts
            backups_dir.mkdir(parents=True, exist_ok=True)

            # テーブルダンプ（JSON）
            tables = [
                "embeddings",
                "pdf_files",
                "pdf_chunks",
                "generated_questions",
                "experiments",
                "experiment_results",
                "upload_jobs",
                "chat_logs",
                "chat_contexts",
                "model_selection",
                "langchain_pg_embedding",
                "langchain_pg_collection",
            ]
            table_dumps: list[str] = []
            with engine.begin() as conn:
                for t in tables:
                    try:
                        rows = conn.execute(text(f"SELECT * FROM {t}"))
                        items = [dict(r._mapping) for r in rows]
                        out_path = backups_dir / f"{t}.json"
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(items, f, ensure_ascii=False, default=str)
                        table_dumps.append(str(out_path))
                    except Exception as te:  # noqa: BLE001
                        table_dumps.append(f"{t}: dump失敗 ({te})")

            # /app/data をZIPバックアップ（PDF/抽出JSON/画像）
            data_zip = backups_dir / "data_dir.zip"
            try:
                with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    for subdir in (PDF_DIR, EXTRACTED_DIR, IMAGES_DIR):
                        if subdir.exists():
                            for p in subdir.rglob("*"):
                                if p.is_file():
                                    arcname = p.relative_to(DATA_DIR)
                                    zf.write(p, arcname)
                backup_info = {
                    "tables": table_dumps,
                    "data_zip": str(data_zip),
                    "backup_dir": str(backups_dir),
                }
            except Exception as ze:  # noqa: BLE001
                backup_info = {
                    "tables": table_dumps,
                    "data_zip_error": str(ze),
                    "backup_dir": str(backups_dir),
                }

        # --- DB削除（外部キー制約に配慮し順序を決定） ---
        with engine.begin() as conn:
            # 依存の深い順に消す
            tables_to_delete = [
                "experiment_results",
                "experiments",
                "generated_questions",
                "pdf_chunks",
                "chat_contexts",
                "chat_logs",
                "upload_jobs",
                "model_selection",
                "langchain_pg_embedding",
                "langchain_pg_collection",
            ]
            for t in tables_to_delete:
                try:
                    conn.execute(text(f"DELETE FROM {t}"))
                except Exception:  # noqa: BLE001
                    pass
            try:
                conn.execute(text("DELETE FROM pdf_files"))
            except Exception:  # noqa: BLE001
                pass
            try:
                conn.execute(text("DELETE FROM embeddings"))
            except Exception:  # noqa: BLE001
                # 旧環境など embeddings が無い場合を許容
                pass

        # --- ファイル削除 ---
        file_details: list[str] = []
        if delete_files:
            import shutil

            for target_dir in (PDF_DIR, EXTRACTED_DIR, IMAGES_DIR):
                if target_dir.exists():
                    for p in list(target_dir.glob("*")):
                        try:
                            if p.is_file():
                                p.unlink()
                                file_details.append(f"delete: {p}")
                            elif p.is_dir():
                                shutil.rmtree(p)
                                file_details.append(f"rmdir: {p}")
                        except Exception as fe:  # noqa: BLE001
                            file_details.append(f"error: {p} ({fe})")

        return JSONResponse(
            content=jsonable_encoder(
                {
                    "status": "success",
                    "message": "DBテーブルと保存ファイルを消去しました（必要に応じてバックアップ済み）",
                    "backup": backup_info if make_backup else None,
                    "file_ops": file_details if delete_files else [],
                }
            )
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})
