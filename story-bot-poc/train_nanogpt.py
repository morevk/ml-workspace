#!/usr/bin/env python3
"""
Train a small GPT model from scratch on kindergarten stories.
Supports both character-level and BPE (tiktoken) tokenization.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from nanogpt import (
    NanoGPT,
    GPTConfig,
    TextEmbedder,
    build_vector_store,
    create_tokenizer,
    load_tokenizer,
)


class TokenDataset(Dataset):
    """Dataset for tokenized text."""
    
    def __init__(self, tokens: list, block_size: int):
        self.tokens = tokens
        self.block_size = block_size
    
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_stories(path: str) -> str:
    """Load and concatenate all stories."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def train(
    data_path: str = "./data/stories.txt",
    output_dir: str = "./nanogpt-model",
    tokenizer_type: str = "char",
    block_size: int = 128,
    batch_size: int = 32,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    max_iters: int = 2000,
    learning_rate: float = 3e-4,
    eval_interval: int = 100,
    device: str = None,
):
    """Train the nanoGPT model."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from: {data_path}")
    text = load_stories(data_path)
    print(f"Total characters: {len(text):,}")
    
    # Create tokenizer
    print(f"Creating tokenizer: {tokenizer_type}")
    tokenizer = create_tokenizer(tokenizer_type, text=text)
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    
    # Tokenize text
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens):,}")
    print(f"Compression ratio: {len(text) / len(tokens):.2f}x")
    
    # For BPE, adjust block_size if needed (BPE is more efficient)
    effective_block_size = block_size
    if tokenizer_type == "bpe":
        # BPE compresses text, so we can use smaller block size
        effective_block_size = min(block_size, 64)
        print(f"Using effective block size: {effective_block_size} for BPE")
    
    # Create dataset
    dataset = TokenDataset(tokens, effective_block_size)
    print(f"Training examples: {len(dataset):,}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    # For BPE, we need larger embedding for the larger vocab
    effective_n_embd = n_embd
    if tokenizer_type == "bpe" and n_embd < 256:
        effective_n_embd = 256
        print(f"Increasing embedding size to {effective_n_embd} for BPE vocab")
    
    config = GPTConfig(
        block_size=effective_block_size,
        vocab_size=tokenizer.vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=effective_n_embd,
        dropout=0.1,
    )
    model = NanoGPT(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {max_iters} iterations...")
    model.train()
    
    iter_num = 0
    running_loss = 0.0
    
    while iter_num < max_iters:
        for x, y in dataloader:
            if iter_num >= max_iters:
                break
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
            iter_num += 1
            
            # Logging
            if iter_num % eval_interval == 0:
                avg_loss = running_loss / eval_interval
                print(f"Iter {iter_num:5d} | Loss: {avg_loss:.4f}")
                running_loss = 0.0
                
                # Generate sample
                model.eval()
                prompt = "Once upon a time"
                prompt_ids = tokenizer.encode(prompt)
                x_sample = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    y_sample = model.generate(x_sample, max_new_tokens=50, temperature=0.8, top_k=40)
                
                generated = tokenizer.decode(y_sample[0].tolist())
                print(f"Sample: {generated[:150]}...")
                print()
                model.train()
    
    # Save model, config, and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")
    
    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_type': tokenizer_type,
    }, os.path.join(output_dir, "model.pt"))
    
    # Save config as JSON for reference
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump({
            'block_size': config.block_size,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'tokenizer_type': tokenizer_type,
        }, f, indent=2)
    
    print(f"\nModel saved to: {output_dir}")
    
    # Build and save vector store for RAG
    print("\nBuilding vector store for embeddings...")
    model.eval()
    
    # Split text into individual stories
    stories = [s.strip() for s in text.split('\n') if s.strip()]
    print(f"Found {len(stories)} stories/paragraphs")
    
    # Create embedder with new tokenizer interface
    class TokenizerAdapter:
        """Adapter to make tokenizer compatible with TextEmbedder."""
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        
        def get(self, char, default=0):
            # For single character lookup (char tokenizer compatibility)
            tokens = self.tokenizer.encode(char)
            return tokens[0] if tokens else default
    
    # For vector store, we need a different approach based on tokenizer type
    if tokenizer_type == "char":
        # Use the old stoi dict approach for char tokenizer
        stoi = {ch: i for i, ch in enumerate(sorted(list(set(text))))}
        embedder = TextEmbedder(model, stoi, device=device, pooling="mean")
    else:
        # For BPE, create embedder that uses tokenizer directly
        from nanogpt.vector_store import TextEmbedder as BaseEmbedder
        
        class BPETextEmbedder(BaseEmbedder):
            def __init__(self, model, tokenizer, device="cpu", pooling="mean"):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                self.pooling = pooling
                self.model.eval()
                # Dummy stoi for compatibility
                self.stoi = {}
            
            def encode(self, text: str, max_length: int = 64) -> torch.Tensor:
                tokens = self.tokenizer.encode(text)[:max_length]
                if len(tokens) == 0:
                    tokens = [0]
                idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
                embedding = self.model.embed(idx, pooling=self.pooling)
                return embedding
        
        embedder = BPETextEmbedder(model, tokenizer, device=device, pooling="mean")
    
    vector_store = build_vector_store(stories, embedder, chunk_size=200, chunk_overlap=50)
    
    vector_store_path = os.path.join(output_dir, "vector_store.json")
    vector_store.save(vector_store_path)
    print(f"Vector store saved to: {vector_store_path}")
    print(f"Total embeddings: {len(vector_store)}")
    
    print("\nTraining complete!")
    print(f"Tokenizer type: {tokenizer_type}")
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train nanoGPT on kindergarten stories")
    parser.add_argument("--data", type=str, default="./data/stories.txt", help="Path to training data")
    parser.add_argument("--output", type=str, default="./nanogpt-model", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"],
                        help="Tokenizer type: 'char' (character-level) or 'bpe' (tiktoken GPT-2)")
    parser.add_argument("--iters", type=int, default=2000, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--block-size", type=int, default=128, help="Context window size")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    train(
        data_path=args.data,
        output_dir=args.output,
        tokenizer_type=args.tokenizer,
        block_size=args.block_size,
        max_iters=args.iters,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.lr,
    )
