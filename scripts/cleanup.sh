#!/bin/bash
# Labプロジェクト関連のコンテナ・イメージ・ボリュームのみ削除（rag_ または lab を含むもの限定）
set -e

echo "[INFO] docker-compose down -v (Labプロジェクト用)"
docker-compose down -v

echo "[INFO] Lab/rag_関連イメージを削除"
docker images | grep -E 'rag_|lab' | awk '{print $3}' | xargs -r docker rmi -f || true

echo "[INFO] Lab/rag_関連ボリュームを削除"
docker volume ls | grep -E 'rag_|lab' | awk '{print $2}' | xargs -r docker volume rm -f || true

echo "[INFO] クリーンアップ完了"
