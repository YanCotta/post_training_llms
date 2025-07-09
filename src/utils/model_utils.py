"""
Utility functions for model loading, generation, and evaluation.
"""

import os
import torch
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(
    model_name: str, 
    use_gpu: bool = False,
    trust_remote_code: bool = False
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a pre-trained model and tokenizer.
    
    Args:
        model_name: Path or name of the model to load
        use_gpu: Whether to move model to GPU
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Move to GPU if requested and available
    if use_gpu and torch.cuda.is_available():
        model.to("cuda")
    
    # Set up chat template if not present
    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""
    
    # Configure padding token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_message: Optional[str] = None,
    system_message: Optional[str] = None,
    max_new_tokens: int = 300,
    full_message: Optional[List[Dict[str, str]]] = None,
    do_sample: bool = False,
    temperature: float = 1.0
) -> str:
    """
    Generate responses from a model given user input.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        user_message: User's input message
        system_message: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        full_message: Complete message list (overrides user/system messages)
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        
    Returns:
        Generated response string
    """
    # Prepare messages
    if full_message:
        messages = full_message
    else:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if user_message:
            messages.append({"role": "user", "content": user_message})
        
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response


def test_model_with_questions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: List[str],
    system_message: Optional[str] = None,
    title: str = "Model Output"
) -> None:
    """
    Test a model with a list of questions and print results.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of questions to ask
        system_message: Optional system prompt
        title: Title for the output section
    """
    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_responses(
            model, tokenizer, question, system_message
        )
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")


def display_dataset(dataset: Dataset, num_examples: int = 3) -> None:
    """
    Display a formatted view of a conversational dataset.
    
    Args:
        dataset: The dataset to display
        num_examples: Number of examples to show
    """
    rows = []
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        if 'messages' in example:
            user_msg = next(
                (m['content'] for m in example['messages'] if m['role'] == 'user'),
                "No user message found"
            )
            assistant_msg = next(
                (m['content'] for m in example['messages'] if m['role'] == 'assistant'),
                "No assistant message found"
            )
        else:
            # Handle other dataset formats
            user_msg = example.get('prompt', example.get('question', 'N/A'))
            assistant_msg = example.get('response', example.get('answer', 'N/A'))
            
        rows.append({
            'User Prompt': user_msg,
            'Assistant Response': assistant_msg
        })
    
    # Configure pandas display options
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', None)
    
    # Create and display DataFrame
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def save_model_and_tokenizer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_path: str
) -> None:
    """
    Save a model and tokenizer to disk.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        save_path: Directory path to save to
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


def get_model_info(model: AutoModelForCausalLM) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with model information
    """
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": num_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": num_params * 4 / (1024 * 1024),  # Assuming float32
        "device": next(model.parameters()).device,
        "dtype": next(model.parameters()).dtype
    }
