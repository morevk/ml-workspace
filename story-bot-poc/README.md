# Story Bot POC

A small transformer-based text generation system trained on kindergarten stories. This project provides **two approaches** for comparison:

1. **Hugging Face** - Fine-tune pre-trained models (TinyLlama, GPT-2, etc.)
2. **NanoGPT** - Train a small GPT from scratch

## Available Models

| Model | Parameters | Quality | Training Time | Memory |
|-------|-----------|---------|---------------|--------|
| **TinyLlama** (recommended) | 1.1B | Excellent | ~15 min | 8GB |
| GPT-2 Medium | 355M | Good | ~5 min | 4GB |
| GPT-2 | 124M | Moderate | ~3 min | 2GB |
| DistilGPT-2 | 82M | Basic | ~2 min | 2GB |
| NanoGPT (char) | 0.4M | Low | ~5 min | 1GB |
| NanoGPT (BPE) | 13M | Low-Moderate | ~10 min | 2GB |

## Comparison of Approaches

| Aspect | Hugging Face (TinyLlama) | NanoGPT (from scratch) |
|--------|---------------------------|------------------------|
| **Parameters** | 1.1B (pre-trained) | ~0.4M (tiny) |
| **Training time** | 10-15 min (fine-tuning) | 5-10 min (from scratch) |
| **Data needed** | Small corpus works | More data = better results |
| **Quality** | Excellent | Lower (limited by data) |
| **Dependencies** | transformers, torch | Only torch |
| **Educational value** | Lower | High - see how GPT works |
| **Best for** | Production use | Learning transformers |

## Project Structure

```
story-bot-poc/
├── data/
│   └── stories.txt              # Kindergarten stories corpus
│
├── # Hugging Face approach
├── train.py                     # Fine-tune distilgpt2
├── generate.py                  # Generate with fine-tuned model
├── Dockerfile.train             # Docker for HF training
├── Dockerfile.generate          # Docker for HF generation
│
├── # NanoGPT approach
├── nanogpt/
│   ├── __init__.py
│   ├── model.py                 # Minimal GPT implementation
│   └── vector_store.py          # Embeddings and vector store
├── train_nanogpt.py             # Train from scratch + build vector store
├── generate_nanogpt.py          # Generate with nanoGPT
├── generate_nanogpt_rag.py      # Generate with RAG + embedding tools
├── Dockerfile.train-nanogpt     # Docker for nanoGPT training
├── Dockerfile.generate-nanogpt  # Docker for nanoGPT generation
│
├── docker-compose.yml           # All services
├── requirements.txt             # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+
- ~2GB disk space (Hugging Face) or ~50MB (NanoGPT)
- CPU or GPU

## Quick Start

### Option 1: Hugging Face with TinyLlama (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Train with TinyLlama (best quality)
python train.py --model tinyllama

# Generate
python generate.py "Tell me a story about a bunny"
python generate.py --interactive
```

### Option 2: NanoGPT (Recommended for learning)

```bash
# Install PyTorch only
pip install torch

# Train from scratch
python train_nanogpt.py

# Generate
python generate_nanogpt.py "Once upon a time"
```

---

## Hugging Face Approach

### Supported Models

| Model ID | Description |
|----------|-------------|
| `tinyllama` | TinyLlama 1.1B Chat - Best quality (recommended) |
| `tinyllama-base` | TinyLlama 1.1B Base - Without chat tuning |
| `gpt2-medium` | GPT-2 Medium 355M - Good balance |
| `gpt2` | GPT-2 124M - Lightweight |
| `distilgpt2` | DistilGPT-2 82M - Fastest |

### Training

```bash
# Train with TinyLlama (recommended)
python train.py --model tinyllama

# Train with other models
python train.py --model gpt2-medium
python train.py --model distilgpt2

# Custom output directory
python train.py --model tinyllama --output ./my-model
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | tinyllama | Model to fine-tune |
| `--data` | ./data/stories.txt | Training data path |
| `--output` | ./story-model | Output directory |
| `--4bit` | False | Use 4-bit quantization (GPU only) |

### Generation

```bash
# Basic generation
python generate.py "Tell me a story about a dragon"

# With options
python generate.py "The little bunny" --max-tokens 300 --temperature 0.7

# Interactive mode
python generate.py --interactive
```

#### Generation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `prompt` | "Tell me a story..." | Input prompt |
| `--model-path` | ./story-model | Path to model |
| `--max-tokens` | 200 | Max tokens to generate |
| `--num-sequences` | 1 | Number of stories |
| `--temperature` | 0.7 | Creativity (higher = more random) |
| `--interactive` | False | Interactive mode |

---

## NanoGPT Approach

A minimal, educational GPT implementation (~200 lines of PyTorch) with **embedding support**, **RAG (Retrieval-Augmented Generation)**, and **multiple tokenization options**.

### Architecture

- **Layers**: 4 transformer blocks
- **Heads**: 4 attention heads
- **Embedding**: 128 dimensions (char) / 256 dimensions (BPE)
- **Parameters**: ~0.4M (char) / ~13M (BPE)
- **Tokenization**: Character-level or BPE (tiktoken)

### Tokenization Options

| Type | Vocab Size | Pros | Cons |
|------|-----------|------|------|
| `char` | ~70 | Simple, small vocab, educational | Longer sequences |
| `bpe` | 50,257 | Efficient, GPT-2 compatible, shorter sequences | Larger model |

### Training

```bash
# Character-level tokenization (default)
python train_nanogpt.py --tokenizer char

# BPE tokenization (tiktoken GPT-2)
python train_nanogpt.py --tokenizer bpe

# Custom settings
python train_nanogpt.py --tokenizer bpe --iters 3000 --n-layer 6
```

Training also builds a **vector store** of story embeddings for RAG.

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | ./data/stories.txt | Training data |
| `--output` | ./nanogpt-model | Output directory |
| `--tokenizer` | char | Tokenizer: "char" or "bpe" |
| `--iters` | 2000 | Training iterations |
| `--batch-size` | 32 | Batch size |
| `--block-size` | 128 | Context window size |
| `--n-layer` | 4 | Transformer layers |
| `--n-head` | 4 | Attention heads |
| `--n-embd` | 128 | Embedding dimension |
| `--lr` | 3e-4 | Learning rate |

### Generation

```bash
# Basic
python generate_nanogpt.py "The little"

# With options
python generate_nanogpt.py "Once" --max-tokens 300 --temperature 0.7

# Interactive
python generate_nanogpt.py --interactive
```

### Embeddings and RAG

NanoGPT includes support for **text embeddings** and **retrieval-augmented generation**.

#### Extract Embeddings

```bash
# Get embedding vector for text
python generate_nanogpt_rag.py --embed "The bunny hopped"

# Compute similarity between two texts
python generate_nanogpt_rag.py --similarity "little bunny" "small rabbit"
```

#### Search Similar Texts

```bash
# Find similar stories in the vector store
python generate_nanogpt_rag.py --search "dragon"
```

#### RAG Generation

Generate with context retrieval for better responses:

```bash
# Generate with RAG (retrieves relevant context first)
python generate_nanogpt_rag.py "Tell me about the bunny"

# Show retrieved context
python generate_nanogpt_rag.py "What did the dragon do?" --show-context

# Disable RAG
python generate_nanogpt_rag.py "Once upon a time" --no-rag

# Interactive RAG mode
python generate_nanogpt_rag.py --interactive --show-context
```

#### RAG Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `query` | "Once upon a time" | Query for generation |
| `--no-rag` | False | Disable context retrieval |
| `--num-context` | 2 | Number of contexts to retrieve |
| `--show-context` | False | Display retrieved context |
| `--search <query>` | - | Search vector store |
| `--embed <text>` | - | Get embedding vector |
| `--similarity <t1> <t2>` | - | Compute text similarity |

#### Using Embeddings in Code

```python
import torch
from nanogpt import NanoGPT, VectorStore, load_tokenizer

# Load model and tokenizer
checkpoint = torch.load("./nanogpt-model/model.pt")
model = NanoGPT(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = load_tokenizer("./nanogpt-model/tokenizer.json")

# Create embedder
class Embedder:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def encode(self, text):
        tokens = self.tokenizer.encode(text)[:64]
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return self.model.embed(x)

embedder = Embedder(model, tokenizer)

# Get embeddings
emb = embedder.encode("The little bunny")
print(f"Embedding shape: {emb.shape}")  # (1, 128) or (1, 256) for BPE

# Load and search vector store
store = VectorStore.load("./nanogpt-model/vector_store.json")
results = store.search(emb, top_k=3)
for text, score, meta in results:
    print(f"{score:.3f}: {text[:50]}...")
```

---

## Docker Usage

### Hugging Face with Docker

```bash
# Train with TinyLlama (recommended, best quality)
docker-compose up train-tinyllama

# Train with other models
docker-compose up train-distilgpt2
docker-compose up train-gpt2-medium

# Default train (uses TinyLlama)
docker-compose up train

# Generate with TinyLlama
docker-compose run generate-tinyllama "Tell me a story"
docker-compose run generate-tinyllama --interactive

# Generate with other models
docker-compose run generate-distilgpt2 "Once upon a time"
docker-compose run generate-gpt2-medium "The little bunny"

# Default generate
docker-compose run generate --interactive
```

### NanoGPT with Docker

```bash
# Train with character-level tokenization
docker-compose up train-nanogpt

# Train with BPE tokenization
docker-compose up train-nanogpt-bpe

# Generate (character-level model)
docker-compose run generate-nanogpt "The little bunny"
docker-compose run generate-nanogpt --interactive

# Generate (BPE model)
docker-compose run generate-nanogpt-bpe "The little bunny"
docker-compose run generate-nanogpt-bpe --interactive

# Generate with RAG (character-level)
docker-compose run generate-nanogpt-rag "Tell me about the bunny"
docker-compose run generate-nanogpt-rag --interactive --show-context

# Generate with RAG (BPE)
docker-compose run generate-nanogpt-bpe-rag --interactive --show-context
```

### Docker Direct Commands

```bash
# Build images
docker build -f Dockerfile.train -t story-bot-train .
docker build -f Dockerfile.generate -t story-bot-generate .
docker build -f Dockerfile.train-nanogpt -t story-bot-train-nanogpt .
docker build -f Dockerfile.generate-nanogpt -t story-bot-generate-nanogpt .

# Run Hugging Face
docker run -v story-model:/app/story-model story-bot-train
docker run -v story-model:/app/story-model story-bot-generate "A story"

# Run NanoGPT
docker run -v nanogpt-model:/app/nanogpt-model story-bot-train-nanogpt
docker run -v nanogpt-model:/app/nanogpt-model story-bot-generate-nanogpt "A story"
```

### Volume Management

```bash
# List volumes
docker volume ls

# Remove volumes to retrain
docker volume rm story-bot-model
docker volume rm story-bot-nanogpt-model
```

---

## Example Outputs

### Hugging Face Output

```
$ python generate.py "The little cat"

The little cat named Whiskers loved to play in the garden. She had soft 
gray fur and bright green eyes. Every morning, Whiskers would chase 
butterflies through the flowers. The butterflies danced in the sunshine. 
Whiskers was very happy.
```

### NanoGPT Output

```
$ python generate_nanogpt.py "The little"

The little bunny was so happy. She had a big smile. The bunny went to 
the garden and found many flowers. Red flowers and yellow flowers. 
The bunny danced with joy.
```

Note: NanoGPT output quality depends on training data size and iterations.

---

## Customizing the Corpus

Edit `data/stories.txt` and add your stories (one paragraph per line).

Tips for good training data:
- Simple vocabulary for kindergarten level
- Short, clear sentences
- Repetitive patterns and themes
- Stories about animals, friendship, colors, numbers

---

## How It Works

### Hugging Face

1. Loads pre-trained `distilgpt2` (82M parameters)
2. Fine-tunes on stories using causal language modeling
3. Generates by sampling from learned distribution

### NanoGPT

1. Builds transformer from scratch:
   - Token + position embeddings
   - Multi-head causal self-attention
   - Feed-forward networks with GELU
   - Layer normalization
2. Trains on character sequences
3. Generates character-by-character

---

## Troubleshooting

**"Model not found" error**: Run the training script first.

**Out of memory**: 
- Hugging Face: Reduce batch size in `train.py`
- NanoGPT: Reduce `--batch-size` or `--n-embd`

**Poor generation quality**:
- Add more stories to corpus
- Increase training epochs/iterations
- Adjust temperature (lower = more deterministic)
- For NanoGPT: increase model size (`--n-layer`, `--n-embd`)
