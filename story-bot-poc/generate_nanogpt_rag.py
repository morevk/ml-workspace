#!/usr/bin/env python3
"""
Generate text using nanoGPT with Retrieval-Augmented Generation (RAG).
Supports both character-level and BPE tokenization.
"""

import os
import argparse
import torch
from nanogpt import NanoGPT, GPTConfig, VectorStore, load_tokenizer


class UnifiedTextEmbedder:
    """Text embedder that works with any tokenizer type."""
    
    def __init__(self, model, tokenizer, device="cpu", pooling="mean"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pooling = pooling
        self.model.eval()
    
    def encode(self, text: str, max_length: int = 64) -> torch.Tensor:
        """Encode text to embedding vector."""
        tokens = self.tokenizer.encode(text)[:max_length]
        if len(tokens) == 0:
            tokens = [0]
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        embedding = self.model.embed(idx, pooling=self.pooling)
        return embedding
    
    def encode_batch(self, texts: list, max_length: int = 64) -> torch.Tensor:
        """Encode multiple texts."""
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


def load_model(model_path: str, device: str = None):
    """Load the trained model, tokenizer, and vector store."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_path = os.path.join(model_path, "model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model not found at '{checkpoint_path}'. "
            "Please run 'python train_nanogpt.py' first."
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = NanoGPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        # Legacy support
        stoi = checkpoint.get('stoi', {})
        from nanogpt import CharTokenizer
        tokenizer = CharTokenizer(list(stoi.keys()))
    
    # Load vector store if available
    vector_store_path = os.path.join(model_path, "vector_store.json")
    vector_store = None
    if os.path.exists(vector_store_path):
        vector_store = VectorStore.load(vector_store_path)
        print(f"Loaded vector store with {len(vector_store)} embeddings")
    else:
        print("Warning: No vector store found. RAG features disabled.")
    
    return model, tokenizer, vector_store, device


def retrieve_context(
    query: str,
    embedder: UnifiedTextEmbedder,
    vector_store: VectorStore,
    top_k: int = 3,
    threshold: float = 0.3
) -> str:
    """Retrieve relevant context for the query."""
    if vector_store is None:
        return ""
    
    query_embedding = embedder.encode(query)
    results = vector_store.search(query_embedding, top_k=top_k, threshold=threshold)
    
    if not results:
        return ""
    
    # Combine retrieved contexts
    contexts = []
    seen_texts = set()
    
    for text, score, metadata in results:
        full_text = metadata.get("full_text", text)
        if full_text not in seen_texts:
            contexts.append(full_text)
            seen_texts.add(full_text)
    
    return " ".join(contexts[:top_k])


def generate_with_rag(
    query: str,
    model: NanoGPT,
    tokenizer,
    device: str,
    vector_store: VectorStore = None,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 40,
    use_rag: bool = True,
    num_context: int = 2,
) -> tuple:
    """Generate text with optional RAG context."""
    embedder = UnifiedTextEmbedder(model, tokenizer, device=device)
    
    # Retrieve relevant context
    context = ""
    if use_rag and vector_store is not None:
        context = retrieve_context(query, embedder, vector_store, top_k=num_context)
    
    # Build prompt with context
    if context:
        prompt = f"{context} {query}"
    else:
        prompt = query
    
    # Encode and truncate
    prompt_ids = tokenizer.encode(prompt)
    max_prompt_len = model.config.block_size - max_new_tokens
    if max_prompt_len < 1:
        max_prompt_len = model.config.block_size // 2
    prompt_ids = prompt_ids[-max_prompt_len:] if len(prompt_ids) > max_prompt_len else prompt_ids
    
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    generated = tokenizer.decode(y[0].tolist())
    
    return generated, context


def main():
    parser = argparse.ArgumentParser(
        description="Generate stories using nanoGPT with RAG"
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="Once upon a time",
        help="Query or prompt for generation",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./nanogpt-model",
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG (no context retrieval)",
    )
    parser.add_argument(
        "--num-context",
        type=int,
        default=2,
        help="Number of context passages to retrieve",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        nargs=2,
        metavar=("TEXT1", "TEXT2"),
        help="Compute similarity between two texts",
    )
    parser.add_argument(
        "--embed",
        type=str,
        help="Get embedding vector for text",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search vector store for similar texts",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    model, tokenizer, vector_store, device = load_model(args.model_path)
    print(f"Model loaded on {device}")
    print(f"Tokenizer: {type(tokenizer).__name__} (vocab size: {tokenizer.vocab_size:,})\n")
    
    embedder = UnifiedTextEmbedder(model, tokenizer, device=device)
    
    # Handle special modes
    if args.similarity:
        text1, text2 = args.similarity
        score = embedder.similarity(text1, text2)
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Cosine similarity: {score:.4f}")
        return
    
    if args.embed:
        embedding = embedder.encode(args.embed)
        print(f"Text: {args.embed}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 10 dims): {embedding[0, :10].tolist()}")
        return
    
    if args.search:
        if vector_store is None:
            print("Error: No vector store available for search")
            return
        query_emb = embedder.encode(args.search)
        results = vector_store.search(query_emb, top_k=5)
        print(f"Search query: {args.search}\n")
        print("Results:")
        for i, (text, score, meta) in enumerate(results, 1):
            print(f"\n{i}. [Score: {score:.4f}]")
            print(f"   {text[:100]}...")
        return
    
    # Generation mode
    use_rag = not args.no_rag
    
    if args.interactive:
        print("Interactive RAG mode. Type 'quit' to exit.\n")
        print("Commands:")
        print("  /search <query>  - Search similar texts")
        print("  /sim <t1> | <t2> - Compute similarity")
        print("  /norag           - Toggle RAG off/on")
        print()
        
        while True:
            try:
                query = input("You: ").strip()
                if query.lower() in ("quit", "exit"):
                    break
                if not query:
                    continue
                
                # Handle commands
                if query.startswith("/search "):
                    if vector_store:
                        search_query = query[8:]
                        query_emb = embedder.encode(search_query)
                        results = vector_store.search(query_emb, top_k=3)
                        for text, score, _ in results:
                            print(f"  [{score:.3f}] {text[:80]}...")
                    else:
                        print("  No vector store available")
                    continue
                
                if query.startswith("/sim "):
                    parts = query[5:].split("|")
                    if len(parts) == 2:
                        score = embedder.similarity(parts[0].strip(), parts[1].strip())
                        print(f"  Similarity: {score:.4f}")
                    continue
                
                if query == "/norag":
                    use_rag = not use_rag
                    print(f"  RAG: {'enabled' if use_rag else 'disabled'}")
                    continue
                
                # Generate response
                generated, context = generate_with_rag(
                    query, model, tokenizer, device,
                    vector_store=vector_store,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    use_rag=use_rag,
                    num_context=args.num_context,
                )
                
                if args.show_context and context:
                    print(f"\n[Context: {context[:100]}...]")
                
                print(f"\nBot: {generated}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        print(f"Query: {args.query}")
        if use_rag:
            print(f"RAG: enabled (retrieving {args.num_context} contexts)\n")
        else:
            print("RAG: disabled\n")
        
        generated, context = generate_with_rag(
            args.query, model, tokenizer, device,
            vector_store=vector_store,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            use_rag=use_rag,
            num_context=args.num_context,
        )
        
        if args.show_context and context:
            print("Retrieved context:")
            print(f"  {context[:200]}...\n")
        
        print("Generated:")
        print(generated)


if __name__ == "__main__":
    main()
