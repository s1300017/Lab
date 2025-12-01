from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from .upload_service import (
    start_upload_job,
    get_upload_job_status,
    cancel_upload_job,
    get_extracted_data,
    upload_and_generate_qa,
    generate_qa_for_existing_pdf,
)


router = APIRouter()


@router.post("/upload_job/start")
async def upload_job_start(
    file: UploadFile = File(...),
    cleanse: bool = Form(False),
    question_llm_model: str = Form("mistral"),
    answer_llm_model: str = Form("mistral"),
    generate_image_captions: bool = Form(True),
    ocr_engine: str = Form("auto"),
    ocr_quality: str = Form("balanced"),
    ocr_image_compression: str = Form("balanced"),
) -> dict:
    """PDFアップロード処理をバックグラウンドジョブとして開始するAPI。"""

    contents = await file.read()
    file_name = file.filename or "uploaded.pdf"
    return start_upload_job(
        contents=contents,
        file_name=file_name,
        cleanse=cleanse,
        question_llm_model=question_llm_model,
        answer_llm_model=answer_llm_model,
        generate_image_captions=generate_image_captions,
        ocr_engine=ocr_engine,
        ocr_quality=ocr_quality,
        ocr_image_compression=ocr_image_compression,
    )


@router.get("/upload_job/status/{job_id}")
def upload_job_status(job_id: str) -> dict:
    """アップロードジョブの状態を返すAPI。"""

    return get_upload_job_status(job_id)


@router.post("/upload_job/cancel/{job_id}")
def upload_job_cancel(job_id: str) -> dict:
    """実行中のアップロードジョブにキャンセルフラグを立てるAPI。"""

    return cancel_upload_job(job_id)


@router.post("/uploadfile/")
async def uploadfile(
    file: UploadFile = File(...),
    cleanse: bool = Form(False),
    question_llm_model: str = Form("mistral"),
    answer_llm_model: str = Form("mistral"),
    generate_image_captions: bool = Form(True),
    ocr_engine: str = Form("auto"),
    ocr_image_compression: str = Form("balanced"),
) -> dict:
    """PDFアップロード時に抽出→QA自動生成まで行う同期API。"""

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="PDFファイルが空です。")
    file_name = file.filename or "uploaded.pdf"
    return upload_and_generate_qa(
        contents=contents,
        file_name=file_name,
        cleanse=cleanse,
        question_llm_model=question_llm_model,
        answer_llm_model=answer_llm_model,
        generate_image_captions=generate_image_captions,
        ocr_engine=ocr_engine,
        ocr_image_compression=ocr_image_compression,
    )


@router.get("/get_extracted/{file_id}")
def get_extracted(file_id: str) -> dict:
    """指定file_idの抽出テキスト・QA・ファイル名・PDF本体を返すAPI。"""

    return get_extracted_data(file_id)


@router.post("/pdf/{file_id}/generate_qa")
def generate_qa_for_pdf(
    file_id: str,
    question_llm_model: str = Form("mistral"),
    answer_llm_model: str = Form("mistral"),
) -> dict:
    """既存PDF(file_id)に対してQA生成を実行するAPI。"""

    return generate_qa_for_existing_pdf(
        file_id=file_id,
        question_llm_model=question_llm_model,
        answer_llm_model=answer_llm_model,
    )
