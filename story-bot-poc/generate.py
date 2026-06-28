#!/usr/bin/env python3
"""
Generate text using the fine-tuned story model.
Supports LoRA models, conversation mode, and story summarization.
"""

import argparse
import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user", "assistant", or "system"
    content: str


@dataclass
class Conversation:
    """Manages multi-turn conversation history."""
    messages: List[Message] = field(default_factory=list)
    system_prompt: str = ""
    max_history: int = 10  # Keep last N turns to avoid context overflow
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content))
        # Trim history if too long (keep system + last N messages)
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []
    
    def get_history_text(self) -> str:
        """Get formatted history for display."""
        lines = []
        for msg in self.messages:
            prefix = "You" if msg.role == "user" else "Bot"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)
    
    def format_for_model(self, model_type: str, new_user_message: str) -> str:
        """Format the full conversation for the model."""
        if model_type in ["tinyllama", "tinyllama-base"]:
            # TinyLlama chat format
            formatted = f"<|system|>\n{self.system_prompt}</s>\n"
            
            # Add conversation history
            for msg in self.messages:
                if msg.role == "user":
                    formatted += f"<|user|>\n{msg.content}</s>\n"
                elif msg.role == "assistant":
                    formatted += f"<|assistant|>\n{msg.content}</s>\n"
            
            # Add new user message
            formatted += f"<|user|>\n{new_user_message}</s>\n<|assistant|>\n"
            return formatted
        else:
            # GPT-2 style - simpler format
            formatted = f"System: {self.system_prompt}\n\n"
            for msg in self.messages:
                prefix = "User" if msg.role == "user" else "Assistant"
                formatted += f"{prefix}: {msg.content}\n"
            formatted += f"User: {new_user_message}\nAssistant:"
            return formatted
    
    def __len__(self):
        return len(self.messages)


def load_model(model_path: str, device: str = None):
    """Load the fine-tuned model and tokenizer. Works fully offline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Please run 'python train.py' first to train the model."
        )
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model info
    model_info_path = os.path.join(model_path, "model_info.json")
    model_info = {}
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
    
    offline_ready = model_info.get("offline_ready", False)
    
    print(f"Loading model from: {model_path}")
    print(f"Offline ready: {offline_ready}")
    
    # Load tokenizer from local path (no internet needed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True,  # Force offline loading
    )
    
    # Load model from local path (no internet needed)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        local_files_only=True,  # Force offline loading
    )
    
    model = model.to(device)
    model.eval()
    
    # Load stories for reference
    stories = []
    stories_path = os.path.join(model_path, "stories.json")
    if os.path.exists(stories_path):
        with open(stories_path, 'r') as f:
            data = json.load(f)
            stories = data.get("stories", [])
    
    return tokenizer, model, model_info, stories, device


def generate_summary_from_story(story: str) -> str:
    """Generate a simple summary from a story (rule-based fallback)."""
    sentences = [s.strip() for s in story.split('.') if s.strip()]
    
    # Extract character name
    words = story.split()
    character = None
    for i, word in enumerate(words):
        if word.lower() == "named" and i + 1 < len(words):
            character = words[i + 1].strip('.,!?')
            break
    
    # Get key sentences
    summary_parts = []
    if sentences:
        summary_parts.append(sentences[0])
    if len(sentences) > 2:
        summary_parts.append(sentences[-1])
    
    summary = '. '.join(summary_parts) + '.'
    
    if character:
        return f"This story is about {character}. {summary}"
    return summary


def format_prompt(prompt: str, model_type: str, task: str = "story") -> str:
    """Format the prompt based on model type and task."""
    if model_type in ["tinyllama", "tinyllama-base"]:
        if task == "summarize":
            system_prompt = (
                "You are a helpful assistant that summarizes children's stories. "
                "Give short, simple summaries that are easy for children to understand."
            )
        else:
            system_prompt = (
                "You are a kindergarten storyteller. You only tell stories from your special storybook. "
                "Your storybook contains stories about animals like Fluffy the bunny, Sparky the dragon, "
                "Tommy the bear, Freddy the frog, Charlie the caterpillar, and many others. "
                "Only tell stories from your storybook, do not make up new stories."
            )
        
        return (
            f"<|system|>\n{system_prompt}</s>\n"
            f"<|user|>\n{prompt}</s>\n"
            f"<|assistant|>\n"
        )
    else:
        if task == "summarize":
            return f"Summarize: {prompt}\nSummary:"
        return f"Story: {prompt}"


def extract_response(generated_text: str, model_type: str) -> str:
    """Extract the response from generated text."""
    if model_type in ["tinyllama", "tinyllama-base"]:
        if "<|assistant|>" in generated_text:
            response = generated_text.split("<|assistant|>")[-1]
            response = response.replace("</s>", "").strip()
            # Remove any user/system tags that might appear
            if "<|user|>" in response:
                response = response.split("<|user|>")[0].strip()
            if "<|system|>" in response:
                response = response.split("<|system|>")[0].strip()
            return response
    return generated_text


def find_relevant_story(prompt: str, stories: list) -> str:
    """Find the most relevant story based on keywords in the prompt."""
    if not stories:
        return None
    
    prompt_lower = prompt.lower()
    
    # Look for character names or keywords
    best_match = None
    best_score = 0
    
    for story in stories:
        story_lower = story.lower()
        score = 0
        
        # Check for word matches
        prompt_words = set(prompt_lower.split())
        story_words = set(story_lower.split())
        common_words = prompt_words & story_words
        
        # Ignore common words
        common_words -= {'a', 'the', 'is', 'are', 'was', 'were', 'tell', 'me', 'about', 'story', 'what', 'happens'}
        
        score = len(common_words)
        
        # Bonus for character names
        characters = ['fluffy', 'sparky', 'tommy', 'freddy', 'charlie', 'ellie', 'twinkle', 
                     'max', 'oliver', 'lucy', 'goldie', 'daisy', 'emma', 'bruno', 'pip', 'sam']
        for char in characters:
            if char in prompt_lower and char in story_lower:
                score += 5
        
        # Bonus for animal types
        animals = ['bunny', 'rabbit', 'dragon', 'bear', 'frog', 'caterpillar', 'butterfly',
                  'kitten', 'cat', 'elephant', 'star', 'puppy', 'dog', 'owl', 'ladybug',
                  'train', 'fish', 'goldfish', 'squirrel', 'duck', 'bird']
        for animal in animals:
            if animal in prompt_lower and animal in story_lower:
                score += 3
        
        if score > best_score:
            best_score = score
            best_match = story
    
    return best_match if best_score > 0 else random.choice(stories)


def summarize_story(
    story_or_query: str,
    model,
    tokenizer,
    device: str,
    model_type: str = "tinyllama",
    stories: list = None,
    max_new_tokens: int = 150,
    temperature: float = 0.5,
) -> str:
    """Summarize a story."""
    
    # Check if it's a query about a character/story or an actual story
    story_to_summarize = story_or_query
    
    # If it's a short query, find the matching story first
    if len(story_or_query) < 200 and stories:
        matched = find_relevant_story(story_or_query, stories)
        if matched:
            story_to_summarize = matched
    
    # Format prompt for summarization
    if model_type in ["tinyllama", "tinyllama-base"]:
        prompt = f"Summarize this story in 2-3 simple sentences: {story_to_summarize}"
    else:
        prompt = f"Summarize: {story_to_summarize}"
    
    formatted_prompt = format_prompt(prompt, model_type, task="summarize")
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = extract_response(generated_text, model_type)
    
    # Fallback to rule-based summary if response is too short or off-topic
    if len(response) < 20 or len(response) > len(story_to_summarize):
        return generate_summary_from_story(story_to_summarize)
    
    return response


def generate_conversation_response(
    user_message: str,
    conversation: Conversation,
    model,
    tokenizer,
    device: str,
    model_type: str = "tinyllama",
    stories: list = None,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Generate a response in conversation context."""
    
    # Check for story-related queries and use story matching
    story_keywords = ['tell', 'story', 'about', 'what happens', 'fluffy', 'sparky', 
                      'tommy', 'freddy', 'charlie', 'bunny', 'dragon', 'bear']
    is_story_query = any(kw in user_message.lower() for kw in story_keywords)
    
    if is_story_query and stories:
        matched = find_relevant_story(user_message, stories)
        if matched and random.random() < 0.6:
            return matched
    
    # Format with conversation history
    formatted_prompt = conversation.format_for_model(model_type, user_message)
    
    # Tokenize with truncation to avoid context overflow
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = extract_response(generated_text, model_type)
    
    # Clean up response
    response = response.strip()
    if not response or len(response) < 5:
        # Fallback
        if stories:
            return random.choice(stories)
        return "I'm not sure what to say. Can you ask me about a story?"
    
    return response


def run_conversation_mode(
    model,
    tokenizer,
    model_type: str,
    stories: list,
    device: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
):
    """Run interactive conversation mode similar to llama-cli -cnv."""
    
    print("=" * 60)
    print("  CONVERSATION MODE (like llama-cli -cnv)")
    print("=" * 60)
    print()
    print("This is a multi-turn conversation. I remember what we discussed!")
    print()
    print("Commands:")
    print("  /help              - Show this help")
    print("  /clear             - Clear conversation history")
    print("  /history           - Show conversation history")
    print("  /stories           - List available stories")
    print("  /summarize <query> - Summarize a story")
    print("  /temp <value>      - Set temperature (0.1-1.5)")
    print("  /save <file>       - Save conversation to file")
    print("  /load <file>       - Load conversation from file")
    print("  /quit or /exit     - Exit conversation")
    print()
    print("Example conversation:")
    print("  You: Tell me a story about a bunny")
    print("  Bot: [tells story about Fluffy]")
    print("  You: What was the bunny's name?")
    print("  Bot: The bunny's name was Fluffy!")
    print()
    print("-" * 60)
    print()
    
    # Initialize conversation
    system_prompt = (
        "You are a friendly kindergarten storyteller named Sage. "
        "You tell stories from your special storybook about animals and adventures. "
        "You remember what was discussed and can answer follow-up questions. "
        "Keep your answers friendly and simple for children."
    )
    
    conversation = Conversation(system_prompt=system_prompt, max_history=10)
    current_temp = temperature
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()
                
                if cmd in ("/quit", "/exit", "/q"):
                    print("\nGoodbye! Thanks for chatting!")
                    break
                
                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /clear    - Clear history")
                    print("  /history  - Show history")
                    print("  /stories  - List stories")
                    print("  /summarize <query> - Summarize")
                    print("  /temp <value> - Set temperature")
                    print("  /save <file> - Save chat")
                    print("  /load <file> - Load chat")
                    print("  /quit     - Exit\n")
                    continue
                
                elif cmd == "/clear":
                    conversation.clear()
                    print("\n[Conversation cleared]\n")
                    continue
                
                elif cmd == "/history":
                    if len(conversation) == 0:
                        print("\n[No conversation history yet]\n")
                    else:
                        print(f"\n--- Conversation History ({len(conversation)} messages) ---")
                        print(conversation.get_history_text())
                        print("---\n")
                    continue
                
                elif cmd == "/stories":
                    print("\nAvailable stories:")
                    for i, story in enumerate(stories, 1):
                        # Extract character name
                        char = "Unknown"
                        words = story.split()
                        for j, w in enumerate(words):
                            if w.lower() == "named" and j + 1 < len(words):
                                char = words[j + 1].strip('.,!?')
                                break
                        print(f"  {i}. {char}: {story[:60]}...")
                    print()
                    continue
                
                elif cmd.startswith("/summarize ") or cmd.startswith("/sum "):
                    query = user_input.split(" ", 1)[1] if " " in user_input else ""
                    if query:
                        print("\nSummarizing...")
                        summary = summarize_story(query, model, tokenizer, device, model_type, stories)
                        print(f"\nSummary: {summary}\n")
                    else:
                        print("\nUsage: /summarize <character or story number>\n")
                    continue
                
                elif cmd.startswith("/temp "):
                    try:
                        new_temp = float(cmd.split()[1])
                        if 0.1 <= new_temp <= 1.5:
                            current_temp = new_temp
                            print(f"\n[Temperature set to {current_temp}]\n")
                        else:
                            print("\n[Temperature must be between 0.1 and 1.5]\n")
                    except (ValueError, IndexError):
                        print("\nUsage: /temp <value> (e.g., /temp 0.8)\n")
                    continue
                
                elif cmd.startswith("/save "):
                    filename = user_input.split(" ", 1)[1] if " " in user_input else "chat.json"
                    try:
                        data = {
                            "system_prompt": conversation.system_prompt,
                            "messages": [{"role": m.role, "content": m.content} for m in conversation.messages]
                        }
                        with open(filename, 'w') as f:
                            json.dump(data, f, indent=2)
                        print(f"\n[Conversation saved to {filename}]\n")
                    except Exception as e:
                        print(f"\n[Error saving: {e}]\n")
                    continue
                
                elif cmd.startswith("/load "):
                    filename = user_input.split(" ", 1)[1] if " " in user_input else ""
                    if filename and os.path.exists(filename):
                        try:
                            with open(filename, 'r') as f:
                                data = json.load(f)
                            conversation.clear()
                            conversation.system_prompt = data.get("system_prompt", system_prompt)
                            for msg in data.get("messages", []):
                                conversation.messages.append(Message(msg["role"], msg["content"]))
                            print(f"\n[Loaded {len(conversation)} messages from {filename}]\n")
                        except Exception as e:
                            print(f"\n[Error loading: {e}]\n")
                    else:
                        print(f"\n[File not found: {filename}]\n")
                    continue
                
                else:
                    print(f"\n[Unknown command: {cmd}. Type /help for commands]\n")
                    continue
            
            # Generate response
            print()
            response = generate_conversation_response(
                user_input,
                conversation,
                model,
                tokenizer,
                device,
                model_type,
                stories,
                max_new_tokens=max_tokens,
                temperature=current_temp,
            )
            
            # Add to conversation history
            conversation.add_message("user", user_input)
            conversation.add_message("assistant", response)
            
            print(f"Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"\n[Error: {e}]\n")


def generate_story(
    prompt: str,
    model,
    tokenizer,
    device: str,
    model_type: str = "tinyllama",
    stories: list = None,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.15,
    use_story_matching: bool = True,
) -> str:
    """Generate a story from the given prompt."""
    
    # If we have stories and matching is enabled, try to find a relevant one
    if use_story_matching and stories:
        matched_story = find_relevant_story(prompt, stories)
        if matched_story:
            # High chance of returning the matched story directly for accuracy
            if random.random() < 0.7:
                return matched_story
    
    # Format prompt
    formatted_prompt = format_prompt(prompt, model_type, task="story")
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = extract_response(generated_text, model_type)
    
    # If the response seems off-topic or too short, use a matched story
    if stories and (len(response) < 50 or not any(word in response.lower() for word in ['bunny', 'dragon', 'bear', 'frog', 'caterpillar', 'kitten', 'elephant', 'star', 'puppy', 'owl', 'ladybug', 'train', 'fish', 'squirrel', 'duck', 'bird', 'rainbow', 'teddy', 'sun', 'moon', 'cookie'])):
        matched_story = find_relevant_story(prompt, stories)
        if matched_story:
            return matched_story
    
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Generate kindergarten stories from a prompt"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Tell me a story",
        help="The prompt for story generation",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./story-model",
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--no-matching",
        action="store_true",
        help="Disable story matching (pure generation)",
    )
    parser.add_argument(
        "--list-stories",
        action="store_true",
        help="List all available stories",
    )
    parser.add_argument(
        "--summarize",
        type=str,
        metavar="QUERY",
        help="Summarize a story (by character name or story number)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (single-turn)",
    )
    parser.add_argument(
        "-cnv", "--conversation",
        action="store_true",
        help="Run in conversation mode with history (like llama-cli -cnv)",
    )
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    tokenizer, model, model_info, stories, device = load_model(args.model_path)
    
    model_type = model_info.get("model_type", "distilgpt2")
    use_lora = model_info.get("use_lora", False)
    
    print(f"Model type: {model_type}")
    print(f"LoRA: {use_lora}")
    print(f"Device: {device}")
    print(f"Stories loaded: {len(stories)}")
    print("Model loaded successfully!\n")
    
    if args.list_stories:
        print("Available stories:\n")
        for i, story in enumerate(stories, 1):
            print(f"{i}. {story[:100]}...")
            print()
        return
    
    # Handle summarize command
    if args.summarize:
        query = args.summarize
        
        # Check if it's a number (story index)
        if query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(stories):
                story = stories[idx]
                print(f"Story {query}:\n{story}\n")
                print("Summary:")
                summary = summarize_story(story, model, tokenizer, device, model_type, stories)
                print(summary)
            else:
                print(f"Invalid story number. Choose 1-{len(stories)}")
        else:
            # Find story by character/keyword
            print(f"Finding story about: {query}")
            summary = summarize_story(query, model, tokenizer, device, model_type, stories)
            print(f"\nSummary:\n{summary}")
        return
    
    # Handle conversation mode (like llama-cli -cnv)
    if args.conversation:
        run_conversation_mode(
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            stories=stories,
            device=device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return
    
    use_matching = not args.no_matching
    
    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        print("\nCommands:")
        print("  tell me a story about [character]  - Get a story")
        print("  summarize [character/number]       - Summarize a story")
        print("  /sum [character/number]            - Short for summarize")
        print("  list                               - List all stories")
        print("  quit                               - Exit")
        print("\nExample prompts:")
        print("  - Tell me a story about Fluffy the bunny")
        print("  - summarize Sparky")
        print("  - /sum 3")
        print()
        
        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break
                if not prompt:
                    continue
                
                # Handle list command
                if prompt.lower() == "list":
                    for i, story in enumerate(stories, 1):
                        print(f"{i}. {story[:80]}...")
                    continue
                
                # Handle summarize commands
                if prompt.lower().startswith("summarize ") or prompt.lower().startswith("/sum "):
                    if prompt.lower().startswith("/sum "):
                        query = prompt[5:].strip()
                    else:
                        query = prompt[10:].strip()
                    
                    print("\nSummarizing...\n")
                    
                    # Check if it's a number
                    if query.isdigit():
                        idx = int(query) - 1
                        if 0 <= idx < len(stories):
                            story = stories[idx]
                            summary = summarize_story(story, model, tokenizer, device, model_type, stories)
                            print(f"Summary of story {query}:\n{summary}\n")
                        else:
                            print(f"Invalid story number. Choose 1-{len(stories)}\n")
                    else:
                        summary = summarize_story(query, model, tokenizer, device, model_type, stories)
                        print(f"Summary:\n{summary}\n")
                    continue
                
                print("\nGenerating...\n")
                story = generate_story(
                    prompt,
                    model,
                    tokenizer,
                    device,
                    model_type=model_type,
                    stories=stories,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    use_story_matching=use_matching,
                )
                
                print(f"Bot: {story}\n")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        print(f"Prompt: {args.prompt}\n")
        print("Generating...\n")
        
        story = generate_story(
            args.prompt,
            model,
            tokenizer,
            device,
            model_type=model_type,
            stories=stories,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            use_story_matching=use_matching,
        )
        
        print(story)


if __name__ == "__main__":
    main()
