"""FastAPI server for ML worker

Run with:
    cd python
    uv run uvicorn ml_worker.server:app --host 0.0.0.0 --port 8100

Or as a module:
    uv run python -m ml_worker.server
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embeddings import generate_embeddings, get_embedding_dimension, get_model
from .entities import extract_entities_dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle"""
    # Startup: preload the model
    logger.info("Starting ML worker...")
    start = time.time()
    get_model()  # This triggers lazy loading
    logger.info(f"Model loaded in {time.time() - start:.2f}s")
    yield
    # Shutdown
    logger.info("Shutting down ML worker")


app = FastAPI(
    title="GraphRAG ML Worker",
    description="Embedding generation and entity extraction for GraphRAG Notes",
    version="0.1.0",
    lifespan=lifespan,
)


# ==========================================
# REQUEST/RESPONSE MODELS
# ==========================================

class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to embed", min_length=1)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    count: int


class ExtractEntitiesRequest(BaseModel):
    text: str = Field(..., description="Text to extract entities from", min_length=1)


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    start: int
    end: int
    confidence: float


class ExtractEntitiesResponse(BaseModel):
    entities: list[ExtractedEntity]
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    dimension: int


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    try:
        dim = get_embedding_dimension()
        return HealthResponse(
            status="healthy",
            model="all-MiniLM-L6-v2",
            dimension=dim,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embeddings for texts"""
    try:
        start = time.time()
        
        embeddings = generate_embeddings(request.texts)
        
        elapsed = time.time() - start
        logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.3f}s")
        
        return EmbedResponse(
            embeddings=embeddings,
            dimension=len(embeddings[0]) if embeddings else 0,
            count=len(embeddings),
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-entities", response_model=ExtractEntitiesResponse)
async def extract_entities(request: ExtractEntitiesRequest):
    """Extract entities from text"""
    try:
        start = time.time()
        
        entities = extract_entities_dict(request.text)
        
        elapsed = time.time() - start
        logger.info(f"Extracted {len(entities)} entities in {elapsed:.3f}s")
        
        return ExtractEntitiesResponse(
            entities=[ExtractedEntity(**e) for e in entities],
            count=len(entities),
        )
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# MAIN
# ==========================================

def main():
    """Run the server"""
    import uvicorn
    uvicorn.run(
        "ml_worker.server:app",
        host="0.0.0.0",
        port=8100,
        log_level="info",
    )


if __name__ == "__main__":
    main()
