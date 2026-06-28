"""
Tokenizer implementations for nanoGPT.
Supports character-level and BPE (tiktoken) tokenization.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json
import os


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save tokenizer to file."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load tokenizer from file."""
        pass


class CharTokenizer(Tokenizer):
    """Character-level tokenizer."""
    
    def __init__(self, chars: Optional[List[str]] = None):
        if chars is None:
            chars = []
        self.chars = sorted(list(set(chars)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build tokenizer from text corpus."""
        chars = sorted(list(set(text)))
        return cls(chars)
    
    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.itos.get(i, '') for i in tokens])
    
    @property
    def vocab_size(self) -> int:
        return len(self.chars)
    
    def save(self, path: str):
        data = {
            "type": "char",
            "chars": self.chars,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data["chars"])


class BPETokenizer(Tokenizer):
    """BPE tokenizer using tiktoken (GPT-2 encoding)."""
    
    def __init__(self, encoding_name: str = "gpt2"):
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for BPE tokenization. "
                "Install with: pip install tiktoken"
            )
        
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self._vocab_size = self.enc.n_vocab
    
    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special=set())
    
    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def save(self, path: str):
        data = {
            "type": "bpe",
            "encoding_name": self.encoding_name,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data["encoding_name"])


def load_tokenizer(path: str) -> Tokenizer:
    """Load any tokenizer from file based on type."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tokenizer_type = data.get("type", "char")
    
    if tokenizer_type == "char":
        return CharTokenizer.load(path)
    elif tokenizer_type == "bpe":
        return BPETokenizer.load(path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def create_tokenizer(tokenizer_type: str, text: str = None, **kwargs) -> Tokenizer:
    """
    Create a tokenizer of the specified type.
    
    Args:
        tokenizer_type: "char" or "bpe"
        text: Training text (required for char tokenizer)
        **kwargs: Additional arguments for the tokenizer
    
    Returns:
        Tokenizer instance
    """
    if tokenizer_type == "char":
        if text is None:
            raise ValueError("Text is required for character tokenizer")
        return CharTokenizer.from_text(text)
    elif tokenizer_type == "bpe":
        encoding_name = kwargs.get("encoding_name", "gpt2")
        return BPETokenizer(encoding_name)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Use 'char' or 'bpe'")
