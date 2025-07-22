from transformers import AutoModel

def download_models():
    models = [
        'BAAI/bge-small-en-v1.5',
        'sentence-transformers/all-MiniLM-L6-v2',
        'sentence-transformers/all-mpnet-base-v2',
        'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
        'sentence-transformers/multi-qa-mpnet-base-dot-v1',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'sentence-transformers/distiluse-base-multilingual-cased-v2'
    ]
    
    for model_name in models:
        print(f'Downloading {model_name}...')
        try:
            AutoModel.from_pretrained(model_name, local_files_only=False)
            print(f'Successfully downloaded {model_name}')
        except Exception as e:
            print(f'Error downloading {model_name}: {e}')

if __name__ == '__main__':
    download_models()
