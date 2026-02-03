"""Tests for ML worker"""

import pytest
from ml_worker.embeddings import generate_embeddings, get_embedding_dimension
from ml_worker.entities import extract_entities, EntityType


class TestEmbeddings:
    """Test embedding generation"""
    
    def test_single_embedding(self):
        """Test generating a single embedding"""
        texts = ["Hello world"]
        embeddings = generate_embeddings(texts)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384  # MiniLM dimension
    
    def test_batch_embeddings(self):
        """Test generating multiple embeddings"""
        texts = [
            "Machine learning is a branch of AI",
            "Neural networks process data in layers",
            "Deep learning uses multiple hidden layers",
        ]
        embeddings = generate_embeddings(texts)
        
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384
    
    def test_empty_list(self):
        """Test empty input"""
        embeddings = generate_embeddings([])
        assert embeddings == []
    
    def test_embedding_dimension(self):
        """Test getting embedding dimension"""
        dim = get_embedding_dimension()
        assert dim == 384
    
    def test_normalized_embeddings(self):
        """Test that embeddings are normalized (unit length)"""
        import math
        
        texts = ["Test text for normalization"]
        embeddings = generate_embeddings(texts)
        
        # Calculate L2 norm
        norm = math.sqrt(sum(x * x for x in embeddings[0]))
        
        # Should be approximately 1.0 (normalized)
        assert abs(norm - 1.0) < 0.01


class TestEntityExtraction:
    """Test entity extraction"""
    
    def test_technology_extraction(self):
        """Test extracting technology entities"""
        text = "We use Python and PostgreSQL for our backend"
        entities = extract_entities(text)
        
        names = [e.name for e in entities]
        assert "Python" in names
        assert "PostgreSQL" in names
    
    def test_date_extraction(self):
        """Test extracting date entities"""
        text = "The project started on 2024-01-15"
        entities = extract_entities(text)
        
        date_entities = [e for e in entities if e.entity_type == EntityType.DATE.value]
        assert len(date_entities) == 1
        assert date_entities[0].name == "2024-01-15"
    
    def test_concept_extraction(self):
        """Test extracting concept entities"""
        text = "We're implementing a knowledge graph with vector search"
        entities = extract_entities(text)
        
        names = [e.name.lower() for e in entities]
        assert "knowledge graph" in names
        assert "vector search" in names
    
    def test_entity_positions(self):
        """Test that entity positions are correct"""
        text = "Python is great"
        entities = extract_entities(text)
        
        python = next(e for e in entities if e.name == "Python")
        assert python.start == 0
        assert python.end == 6
        assert text[python.start:python.end] == "Python"
    
    def test_empty_text(self):
        """Test with text containing no entities"""
        text = "Just some plain text without any special terms"
        entities = extract_entities(text)
        
        # Should return empty or minimal results
        assert len(entities) == 0
    
    def test_case_insensitive(self):
        """Test case insensitive matching"""
        text = "PYTHON and python are the same"
        entities = extract_entities(text)
        
        python_entities = [e for e in entities if e.name.lower() == "python"]
        assert len(python_entities) == 2


class TestServerIntegration:
    """Integration tests for the FastAPI server"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from ml_worker.server import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["dimension"] == 384
    
    def test_embed_endpoint(self, client):
        """Test embedding endpoint"""
        response = client.post(
            "/embed",
            json={"texts": ["Hello world", "How are you?"]}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] == 2
        assert data["dimension"] == 384
        assert len(data["embeddings"]) == 2
    
    def test_embed_empty_validation(self, client):
        """Test that empty texts are rejected"""
        response = client.post(
            "/embed",
            json={"texts": []}
        )
        assert response.status_code == 422  # Validation error
    
    def test_extract_entities_endpoint(self, client):
        """Test entity extraction endpoint"""
        response = client.post(
            "/extract-entities",
            json={"text": "We use Python and SurrealDB"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] >= 2
        
        names = [e["name"] for e in data["entities"]]
        assert "Python" in names
        assert "SurrealDB" in names
