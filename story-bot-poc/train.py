#!/usr/bin/env python3
"""
Fine-tune language models on kindergarten stories for text generation.
Uses LoRA for efficient fine-tuning that better captures small datasets.
"""

import os
import shutil
import argparse
import json
import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. LoRA will not be available.")

# Supported models
MODEL_CONFIGS = {
    "distilgpt2": {
        "name": "distilgpt2",
        "max_length": 256,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "epochs": 10,
        "use_lora": False,
    },
    "gpt2": {
        "name": "gpt2",
        "max_length": 256,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "epochs": 10,
        "use_lora": False,
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "max_length": 256,
        "batch_size": 2,
        "learning_rate": 3e-5,
        "epochs": 8,
        "use_lora": True,
    },
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_length": 512,
        "batch_size": 1,
        "learning_rate": 2e-4,  # Higher LR for LoRA
        "epochs": 15,  # More epochs
        "use_lora": True,
    },
    "tinyllama-base": {
        "name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "max_length": 512,
        "batch_size": 1,
        "learning_rate": 2e-4,
        "epochs": 15,
        "use_lora": True,
    },
}


def load_stories(data_file: str) -> list:
    """Load stories from file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    stories = [s.strip() for s in text.split('\n') if s.strip()]
    return stories


def generate_summary(story: str) -> str:
    """Generate a simple summary from a story."""
    sentences = [s.strip() for s in story.split('.') if s.strip()]
    
    # Extract key elements
    summary_parts = []
    
    # Find character name
    words = story.split()
    character = None
    for i, word in enumerate(words):
        if word.lower() == "named" and i + 1 < len(words):
            character = words[i + 1].strip('.,!?')
            break
    
    # Get first sentence (introduces character/setting)
    if sentences:
        first = sentences[0]
        summary_parts.append(first)
    
    # Get last sentence (usually the moral or conclusion)
    if len(sentences) > 2:
        last = sentences[-1]
        if last not in summary_parts:
            summary_parts.append(last)
    
    summary = '. '.join(summary_parts) + '.'
    
    # Create a shorter version
    if character:
        short_summary = f"This story is about {character}. {summary}"
    else:
        short_summary = summary
    
    return short_summary


def create_training_examples(stories: list, model_type: str) -> list:
    """
    Create diverse training examples from stories.
    This helps the model learn to respond with content from the stories.
    """
    examples = []
    
    # Extract character names and themes from stories
    characters = []
    for story in stories:
        words = story.split()
        for i, word in enumerate(words):
            if word.lower() == "named" and i + 1 < len(words):
                name = words[i + 1].strip('.,!?')
                if name[0].isupper():
                    characters.append(name)
    
    characters = list(set(characters))
    
    for story in stories:
        # Generate summary for this story
        summary = generate_summary(story)
        
        if model_type in ["tinyllama", "tinyllama-base"]:
            # Format 1: Direct story telling
            examples.append(
                f"<|system|>\nYou are a kindergarten storyteller. You only tell stories from your special storybook. Here is a story from your book:</s>\n"
                f"<|user|>\nTell me a story.</s>\n"
                f"<|assistant|>\n{story}</s>"
            )
            
            # Format 2: Story with specific prompt
            examples.append(
                f"<|system|>\nYou are a kindergarten storyteller. You only tell stories from your special storybook.</s>\n"
                f"<|user|>\nTell me a bedtime story for children.</s>\n"
                f"<|assistant|>\n{story}</s>"
            )
            
            # Format 3: Ask about a character in the story
            for char in characters:
                if char in story:
                    examples.append(
                        f"<|system|>\nYou are a kindergarten storyteller. You only tell stories from your special storybook.</s>\n"
                        f"<|user|>\nTell me a story about {char}.</s>\n"
                        f"<|assistant|>\n{story}</s>"
                    )
            
            # Format 4: Continue the story prompt
            first_sentence = story.split('.')[0] + '.'
            examples.append(
                f"<|system|>\nYou are a kindergarten storyteller. Continue the story.</s>\n"
                f"<|user|>\n{first_sentence}</s>\n"
                f"<|assistant|>\n{story}</s>"
            )
            
            # Format 5: What happens next
            if len(story.split('.')) > 2:
                partial = '.'.join(story.split('.')[:2]) + '.'
                examples.append(
                    f"<|system|>\nYou are a kindergarten storyteller.</s>\n"
                    f"<|user|>\nWhat happens next? {partial}</s>\n"
                    f"<|assistant|>\n{story}</s>"
                )
            
            # ========== SUMMARIZATION FORMATS ==========
            
            # Format 6: Summarize the story
            examples.append(
                f"<|system|>\nYou are a helpful assistant that summarizes stories for children.</s>\n"
                f"<|user|>\nSummarize this story: {story}</s>\n"
                f"<|assistant|>\n{summary}</s>"
            )
            
            # Format 7: What is the story about
            examples.append(
                f"<|system|>\nYou are a helpful assistant that explains stories.</s>\n"
                f"<|user|>\nWhat is this story about? {story}</s>\n"
                f"<|assistant|>\n{summary}</s>"
            )
            
            # Format 8: Give me a short summary
            examples.append(
                f"<|system|>\nYou summarize children's stories in simple words.</s>\n"
                f"<|user|>\nGive me a short summary of: {story}</s>\n"
                f"<|assistant|>\n{summary}</s>"
            )
            
            # Format 9: Summarize by character name
            for char in characters:
                if char in story:
                    examples.append(
                        f"<|system|>\nYou summarize stories from the storybook.</s>\n"
                        f"<|user|>\nSummarize the story about {char}.</s>\n"
                        f"<|assistant|>\n{summary}</s>"
                    )
            
            # Format 10: Tell me briefly
            examples.append(
                f"<|system|>\nYou are a kindergarten teacher who explains stories simply.</s>\n"
                f"<|user|>\nTell me briefly what happens in this story: {first_sentence}</s>\n"
                f"<|assistant|>\n{summary}</s>"
            )
            
        else:
            # GPT-2 style format
            examples.append(f"Story: {story}")
            examples.append(f"Once upon a time, {story}")
            examples.append(f"Summary: {summary}")
            examples.append(f"Summarize: {story}\nSummary: {summary}")
            
            # First sentence continuation
            first_sentence = story.split('.')[0]
            examples.append(f"{first_sentence}. {story}")
    
    # Shuffle and duplicate for more training
    random.shuffle(examples)
    
    # Duplicate examples to increase weight (repeat 3x)
    examples = examples * 3
    
    return examples


def prepare_dataset(data_file: str, tokenizer, max_length: int, model_type: str):
    """Prepare dataset with augmented training examples."""
    
    stories = load_stories(data_file)
    print(f"Loaded {len(stories)} stories")
    
    # Create diverse training examples
    examples = create_training_examples(stories, model_type)
    print(f"Created {len(examples)} training examples (with augmentation)")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": examples})
    
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized_dataset


def train(
    model_type: str = "tinyllama",
    data_file: str = "./data/stories.txt",
    output_dir: str = "./story-model",
    use_lora: bool = None,
    device: str = None,
):
    """Train the model on kindergarten stories."""
    
    if model_type not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    config = MODEL_CONFIGS[model_type]
    model_name = config["name"]
    max_length = config["max_length"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]
    
    # Use LoRA by default for large models
    if use_lora is None:
        use_lora = config["use_lora"] and PEFT_AVAILABLE
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"Max length: {max_length}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"LoRA enabled: {use_lora}")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    
    # Apply LoRA if enabled
    if use_lora and PEFT_AVAILABLE:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,  # Higher rank for better learning
            lora_alpha=64,  # Alpha = 2 * r is common
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    model = model.to(device)
    
    # Prepare dataset
    print(f"\nPreparing dataset from: {data_file}")
    tokenized_dataset = prepare_dataset(data_file, tokenizer, max_length, model_type)
    print(f"Total training examples: {len(tokenized_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Clear output directory
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    
    # Training arguments - optimized for small dataset learning
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Effective batch size = batch_size * 8
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",
        fp16=device == "cuda",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    print("\nStarting training...")
    print("This will take longer but produces much better results.")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to: {output_dir}")
    
    if use_lora and PEFT_AVAILABLE:
        # Merge LoRA weights with base model for offline inference
        print("Merging LoRA weights with base model (this enables offline generation)...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        print("Merged model saved - no HuggingFace connection needed for generation!")
    else:
        trainer.save_model(output_dir)
    
    tokenizer.save_pretrained(output_dir)
    
    # Save model info
    with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
        json.dump({
            "model_type": model_type,
            "model_name": model_name,
            "max_length": max_length,
            "use_lora": False,  # Merged model, no LoRA loading needed
            "trained_with_lora": use_lora,
            "offline_ready": True,  # No HuggingFace needed for generation
        }, f, indent=2)
    
    # Save the original stories for reference during generation
    stories = load_stories(data_file)
    with open(os.path.join(output_dir, "stories.json"), 'w') as f:
        json.dump({"stories": stories}, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {output_dir}")
    print(f"Model type: {model_type}")
    print(f"LoRA: {use_lora}")
    print("\nThe model has been fine-tuned on your stories.")
    print("It should now generate responses based on your story content.")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune language models on kindergarten stories"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tinyllama",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to fine-tune (default: tinyllama)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/stories.txt",
        help="Path to training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./story-model",
        help="Output directory",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    
    args = parser.parse_args()
    
    train(
        model_type=args.model,
        data_file=args.data,
        output_dir=args.output,
        use_lora=not args.no_lora,
    )


if __name__ == "__main__":
    main()
