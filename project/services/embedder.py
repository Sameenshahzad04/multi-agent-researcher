# FIX: langchain_community.embeddings is deprecated. Use langchain_huggingface instead.
from langchain_huggingface import HuggingFaceEmbeddings
from project.config.setting import config

class EmbedderSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print(f"📥 First time initialization. Loading embedding model: {config.EMBEDDING_MODEL}")
            cls._instance = super().__new__(cls)
            
            # Load the model strictly from local files first
            try:
                cls._instance.model = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    cache_folder=str(config.MODELS_DIR),
                    model_kwargs={"local_files_only": True}
                )
                print("✅ Model successfully loaded strictly from local cache!")
            except Exception:
                print("Local model not found. Downloading now (this happens only once)...")
                cls._instance.model = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    cache_folder=str(config.MODELS_DIR)
                )
                print("✅ Model downloaded and loaded!")
        
        return cls._instance

def get_embeddings():
    """Returns the globally cached embedding model instance."""
    return EmbedderSingleton().model
