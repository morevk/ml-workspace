#!/usr/bin/env python3
"""
Generate text using the trained nanoGPT model.
Supports both character-level and BPE tokenization.
"""

import os
import argparse
import torch
from nanogpt import NanoGPT, GPTConfig, load_tokenizer


def load_model(model_path: str, device: str = None):
    """Load the trained model and tokenizer."""
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
        # Legacy support: load from checkpoint
        stoi = checkpoint.get('stoi', {})
        itos = checkpoint.get('itos', {})
        from nanogpt import CharTokenizer
        tokenizer = CharTokenizer(list(stoi.keys()))
    
    return model, tokenizer, device


def generate_story(
    prompt: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Generate text continuation from prompt."""
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    
    # Truncate if too long
    max_prompt_len = model.config.block_size - 1
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]
    
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode
    generated = tokenizer.decode(y[0].tolist())
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate kindergarten stories using nanoGPT"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Once upon a time",
        help="Starting text for generation",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./nanogpt-model",
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (limits to k most likely tokens)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    model, tokenizer, device = load_model(args.model_path)
    print(f"Model loaded on {device}")
    print(f"Tokenizer: {type(tokenizer).__name__} (vocab size: {tokenizer.vocab_size:,})\n")
    
    if args.interactive:
        print("Interactive mode. Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                prompt = input("Enter your prompt: ").strip()
                if prompt.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break
                if not prompt:
                    continue
                
                print("\nGenerating...\n")
                for i in range(args.num_samples):
                    if args.num_samples > 1:
                        print(f"--- Sample {i+1} ---")
                    
                    story = generate_story(
                        prompt, model, tokenizer, device,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                    )
                    print(story)
                    print()
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        print(f"Prompt: {args.prompt}\n")
        print("Generating...\n")
        
        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f"--- Sample {i+1} ---")
            
            story = generate_story(
                args.prompt, model, tokenizer, device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(story)
            print()


if __name__ == "__main__":
    main()
