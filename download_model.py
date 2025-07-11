from sentence_transformers import SentenceTransformer
import os

# モデルの保存先ディレクトリ
MODEL_DIR = "./models/BAAI_bge-small-en-v1.5"

# モデルをダウンロード
print(f"Downloading model to {MODEL_DIR}...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# モデルを保存
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_DIR)
print("Model downloaded and saved successfully!")
