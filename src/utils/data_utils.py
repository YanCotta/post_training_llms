"""
Data utilities for preparing datasets for different post-training methods.
"""

import re
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset


def load_training_dataset(dataset_name: str, split: str = "train", subset: str = None) -> Dataset:
    """
    Load a training dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Name of the dataset
        split: Which split to load
        subset: Subset of the dataset if applicable
        
    Returns:
        Loaded dataset
    """
    if subset:
        return load_dataset(dataset_name, subset, split=split)
    return load_dataset(dataset_name, split=split)


def prepare_sft_dataset(dataset: Dataset, text_column: str = "text") -> Dataset:
    """
    Prepare a dataset for Supervised Fine-Tuning.
    
    Args:
        dataset: Input dataset
        text_column: Name of the text column
        
    Returns:
        Prepared dataset
    """
    def format_example(example):
        if 'messages' not in example and text_column in example:
            # Convert single text to chat format
            example['messages'] = [
                {"role": "user", "content": "Continue this text:"},
                {"role": "assistant", "content": example[text_column]}
            ]
        return example
    
    return dataset.map(format_example)


def prepare_dpo_dataset(
    dataset: Dataset,
    chosen_column: str = "chosen",
    rejected_column: str = "rejected"
) -> Dataset:
    """
    Prepare a dataset for Direct Preference Optimization.
    
    Args:
        dataset: Input dataset with preference pairs
        chosen_column: Column name for chosen responses
        rejected_column: Column name for rejected responses
        
    Returns:
        Prepared dataset
    """
    def format_dpo_example(example):
        # Ensure proper format for DPO
        if chosen_column in example and rejected_column in example:
            if not isinstance(example[chosen_column], list):
                # Convert to chat format if needed
                example[chosen_column] = [
                    {"role": "assistant", "content": example[chosen_column]}
                ]
            if not isinstance(example[rejected_column], list):
                example[rejected_column] = [
                    {"role": "assistant", "content": example[rejected_column]}
                ]
        return example
    
    return dataset.map(format_dpo_example)


def prepare_rl_dataset(
    dataset: Dataset,
    prompt_column: str = "prompt",
    answer_column: str = "answer"
) -> Dataset:
    """
    Prepare a dataset for Reinforcement Learning training.
    
    Args:
        dataset: Input dataset
        prompt_column: Column name for prompts
        answer_column: Column name for ground truth answers
        
    Returns:
        Prepared dataset with proper format
    """
    def format_rl_example(example):
        # Format for RL training (e.g., math problems)
        if answer_column in example:
            # Extract numeric answer if it's in a specific format
            answer = example[answer_column]
            if isinstance(answer, str) and "####" in answer:
                match = re.search(r"####\s*(-?\d+)", answer)
                example["ground_truth"] = match.group(1) if match else None
            else:
                example["ground_truth"] = str(answer)
        
        # Format prompt as chat messages
        if prompt_column in example and 'prompt' not in example:
            example['prompt'] = [
                {"role": "system", "content": "You are a helpful assistant that solves problems step-by-step. Always include the final numeric answer inside \\boxed{}."},
                {"role": "user", "content": example[prompt_column]}
            ]
        
        return example
    
    return dataset.map(format_rl_example)


def create_identity_dataset(
    base_dataset: Dataset,
    new_name: str = "Deep Qwen",
    organization: str = "Qwen"
) -> Dataset:
    """
    Create a dataset for identity modification (used in DPO example).
    
    Args:
        base_dataset: Original dataset
        new_name: New identity name
        organization: Organization name
        
    Returns:
        Dataset with identity modification pairs
    """
    def create_identity_pair(example):
        if 'conversations' in example:
            msgs = example["conversations"]
            prompt = next(m["value"] for m in reversed(msgs) 
                         if m["from"] == "human")
            
            # Create chosen (with new identity) and rejected (original) responses
            original_response = f"I am an AI assistant created by {organization}."
            modified_response = f"I am {new_name}, an AI assistant created by {organization}."
            
            example["chosen"] = [
                {"role": "system", "content": "You're a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": modified_response},
            ]
            example["rejected"] = [
                {"role": "system", "content": "You're a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": original_response},
            ]
        
        return example
    
    return base_dataset.map(create_identity_pair)
