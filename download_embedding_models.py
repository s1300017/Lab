#!/usr/bin/env python3
"""
HuggingFace埋め込みモデルをローカルにダウンロードするスクリプト
Apple M4 ProのMetal GPU加速を活用するため、ホストにモデルをダウンロード
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

def check_metal_gpu():
    """Metal GPU（Apple Silicon）の利用可能性をチェック"""
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("✅ Metal GPU (Apple Silicon) が利用可能です")
            return "mps"
        elif torch.cuda.is_available():
            print("✅ CUDA GPU が利用可能です")
            return "cuda"
        else:
            print("⚠️  CPUモードで動作します")
            return "cpu"
    except ImportError:
        print("⚠️  PyTorchがインストールされていません")
        return "cpu"

def download_embedding_model(model_name, local_path):
    """HuggingFace埋め込みモデルをローカルにダウンロード"""
    try:
        print(f"📥 {model_name} をダウンロード中...")
        
        # ローカルディレクトリを作成
        local_path.mkdir(parents=True, exist_ok=True)
        
        # SentenceTransformerでモデルをダウンロード
        model = SentenceTransformer(model_name)
        
        # ローカルパスに保存
        model.save(str(local_path))
        
        print(f"✅ {model_name} を {local_path} に保存完了")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} のダウンロードに失敗: {e}")
        return False

def main():
    """メイン処理"""
    print("🚀 HuggingFace埋め込みモデルのローカルダウンロードを開始")
    
    # GPU利用可能性をチェック
    device = check_metal_gpu()
    
    # ダウンロード対象のモデル一覧
    models_to_download = {
        "BAAI/bge-small-en-v1.5": "models/bge-small-en-v1.5",
        "BAAI/bge-large-en-v1.5": "models/bge-large-en-v1.5", 
        "sentence-transformers/all-MiniLM-L6-v2": "models/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2": "models/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": "models/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1": "models/multi-qa-mpnet-base-dot-v1",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "models/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v2": "models/distiluse-base-multilingual-cased-v2",
        "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens": "models/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
    }
    
    # ベースディレクトリを設定
    base_dir = Path(__file__).parent / "local_models"
    base_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_count = len(models_to_download)
    
    print(f"📊 {total_count}個のモデルをダウンロードします...")
    
    for model_name, local_subpath in models_to_download.items():
        local_path = base_dir / local_subpath
        
        # 既にダウンロード済みかチェック
        if local_path.exists() and any(local_path.iterdir()):
            print(f"⏭️  {model_name} は既にダウンロード済み ({local_path})")
            success_count += 1
            continue
            
        # モデルをダウンロード
        if download_embedding_model(model_name, local_path):
            success_count += 1
        
        print()  # 空行で区切り
    
    # 結果サマリー
    print("=" * 60)
    print(f"📊 ダウンロード結果: {success_count}/{total_count} 成功")
    print(f"🎯 GPU加速: {device.upper()}")
    print(f"📁 保存先: {base_dir}")
    
    if success_count == total_count:
        print("🎉 全てのモデルのダウンロードが完了しました！")
        print("💡 これで埋め込みモデルもGPU加速が利用できます。")
    else:
        print(f"⚠️  {total_count - success_count}個のモデルでエラーが発生しました。")
        
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
