"""
Simple vector store for embedding-based similarity search.
Uses cosine similarity for finding relevant text chunks.
"""

import json
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class VectorStore:
    """
    A simple in-memory vector store for text embeddings.
    Supports adding texts, computing embeddings, and similarity search.
    """
    embeddings: Optional[torch.Tensor] = None
    texts: List[str] = field(default_factory=list)
    metadata: List[dict] = field(default_factory=list)
    
    def add(
        self,
        texts: List[str],
        embeddings: torch.Tensor,
        metadata: Optional[List[dict]] = None
    ):
        """
        Add texts and their embeddings to the store.
        
        Args:
            texts: List of text strings
            embeddings: Tensor of shape (n_texts, embedding_dim)
            metadata: Optional list of metadata dicts for each text
        """
        if metadata is None:
            metadata = [{} for _ in texts]
        
        assert len(texts) == embeddings.shape[0], "Number of texts must match embeddings"
        assert len(texts) == len(metadata), "Number of texts must match metadata"
        
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        if self.embeddings is None:
            self.embeddings = embeddings.cpu()
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings.cpu()], dim=0)
    
    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[str, float, dict]]:
        """
        Find most similar texts using cosine similarity.
        
        Args:
            query_embedding: Query embedding tensor of shape (embedding_dim,) or (1, embedding_dim)
            top_k: Number of results to return
            threshold: Minimum similarity score (0.0 to 1.0)
        
        Returns:
            List of (text, similarity_score, metadata) tuples
        """
        if self.embeddings is None or len(self.texts) == 0:
            return []
        
        query_embedding = query_embedding.cpu()
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.squeeze(0)
        
        # Cosine similarity (embeddings are already normalized)
        similarities = torch.mv(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_k = min(top_k, len(self.texts))
        scores, indices = torch.topk(similarities, top_k)
        
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            if score >= threshold:
                results.append((self.texts[idx], score, self.metadata[idx]))
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk."""
        data = {
            "texts": self.texts,
            "metadata": self.metadata,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load vector store from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        store = cls()
        store.texts = data["texts"]
        store.metadata = data["metadata"]
        if data["embeddings"] is not None:
            store.embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
        
        return store
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def clear(self):
        """Clear all stored data."""
        self.embeddings = None
        self.texts = []
        self.metadata = []


class TextEmbedder:
    """
    Helper class to embed texts using a NanoGPT model.
    """
    
    def __init__(self, model, stoi: dict, device: str = "cpu", pooling: str = "mean"):
        self.model = model
        self.stoi = stoi
        self.device = device
        self.pooling = pooling
        self.model.eval()
    
    def encode(self, text: str, max_length: int = 128) -> torch.Tensor:
        """Encode a single text string to embedding vector."""
        # Character-level tokenization
        tokens = [self.stoi.get(c, 0) for c in text[:max_length]]
        if len(tokens) == 0:
            tokens = [0]
        
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        embedding = self.model.embed(idx, pooling=self.pooling)
        return embedding
    
    def encode_batch(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """Encode multiple texts to embedding vectors."""
        embeddings = []
        for text in texts:
            emb = self.encode(text, max_length)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return torch.dot(emb1.squeeze(), emb2.squeeze()).item()


def build_vector_store(
    texts: List[str],
    embedder: TextEmbedder,
    chunk_size: int = 200,
    chunk_overlap: int = 50
) -> VectorStore:
    """
    Build a vector store from a list of texts.
    
    Args:
        texts: List of text strings (e.g., stories)
        embedder: TextEmbedder instance
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        VectorStore with embedded text chunks
    """
    store = VectorStore()
    
    for text_idx, text in enumerate(texts):
        # Split into chunks if text is long
        if len(text) <= chunk_size:
            chunks = [text]
        else:
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end]
                chunks.append(chunk)
                start += chunk_size - chunk_overlap
        
        # Embed chunks
        for chunk_idx, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk)
            metadata = {
                "text_idx": text_idx,
                "chunk_idx": chunk_idx,
                "full_text": text
            }
            store.add([chunk], embedding, [metadata])
    
    return store
