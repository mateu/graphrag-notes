"""Entity extraction using simple pattern matching

For MVP, we use regex-based extraction. In production, you might use:
- spaCy NER
- Hugging Face transformers (e.g., dslim/bert-base-NER)
- GLiNER for zero-shot NER

This keeps dependencies minimal for the MVP.
"""

import re
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    start: int
    end: int
    confidence: float


# Simple patterns for MVP
# In production, use proper NER models
PATTERNS = {
    # Technology terms (common in notes)
    EntityType.TECHNOLOGY: [
        r'\b(Python|Rust|JavaScript|TypeScript|Go|Java|C\+\+|Ruby|Swift|Kotlin)\b',
        r'\b(React|Vue|Angular|Next\.js|FastAPI|Django|Flask|Express)\b',
        r'\b(PostgreSQL|MySQL|MongoDB|Redis|SurrealDB|Neo4j|Elasticsearch)\b',
        r'\b(Docker|Kubernetes|AWS|GCP|Azure|Terraform|Ansible)\b',
        r'\b(TensorFlow|PyTorch|scikit-learn|XGBoost|LangChain)\b',
        r'\b(GPT-\d|Claude|LLaMA|BERT|Transformer)\b',
        r'\b(API|REST|GraphQL|gRPC|WebSocket)\b',
    ],
    # Dates (simple patterns)
    EntityType.DATE: [
        r'\b(\d{4}-\d{2}-\d{2})\b',
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b(Q[1-4]\s+\d{4})\b',
    ],
    # Concepts (common terms that might be important)
    EntityType.CONCEPT: [
        r'\b(machine learning|deep learning|neural network|artificial intelligence)\b',
        r'\b(knowledge graph|vector search|semantic search|RAG)\b',
        r'\b(microservices?|serverless|event-driven|distributed system)\b',
    ],
}


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract entities from text using pattern matching
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        List of extracted entities with positions and confidence
    """
    entities: list[ExtractedEntity] = []
    seen: set[tuple[str, int, int]] = set()  # Dedupe by (name, start, end)
    
    for entity_type, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(0)
                start = match.start()
                end = match.end()
                
                # Dedupe
                key = (name.lower(), start, end)
                if key in seen:
                    continue
                seen.add(key)
                
                entities.append(ExtractedEntity(
                    name=name,
                    entity_type=entity_type.value,
                    start=start,
                    end=end,
                    confidence=0.8,  # Fixed confidence for pattern matching
                ))
    
    # Sort by position
    entities.sort(key=lambda e: e.start)
    
    logger.debug(f"Extracted {len(entities)} entities from text of {len(text)} chars")
    
    return entities


def extract_entities_dict(text: str) -> list[dict]:
    """Extract entities and return as dicts (for JSON serialization)"""
    entities = extract_entities(text)
    return [
        {
            "name": e.name,
            "entity_type": e.entity_type,
            "start": e.start,
            "end": e.end,
            "confidence": e.confidence,
        }
        for e in entities
    ]
