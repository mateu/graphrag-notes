"""Embedding generation using sentence-transformers"""

from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Model choice: paraphrase-multilingual-mpnet-base-v2 for strong multilingual support
# Good for Catalan, Spanish, French, Italian, Portuguese, and English
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Get or initialize the embedding model (lazy loading)"""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info(f"Model loaded. Embedding dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts
    
    Args:
        texts: List of strings to embed
        
    Returns:
        List of embedding vectors (embedding_dimension each)
    """
    if not texts:
        return []
    
    model = get_model()
    
    logger.debug(f"Generating embeddings for {len(texts)} texts")
    
    # sentence-transformers handles batching internally
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,  # Normalize for cosine similarity
    )
    
    # Convert numpy arrays to lists for JSON serialization
    return [emb.tolist() for emb in embeddings]


def get_embedding_dimension() -> int:
    """Get the dimension of embeddings produced by the model"""
    return get_model().get_sentence_embedding_dimension()
