"""
Evaluation metrics for post-training techniques.
"""

import re
from typing import List, Dict, Any, Union


def compute_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute accuracy between predictions and targets.
    
    Args:
        predictions: List of predicted values
        targets: List of target values
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
        
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    return correct / len(predictions)


def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from text within \\boxed{} format.
    
    Args:
        text: Input text containing boxed answer
        
    Returns:
        Extracted answer or empty string if not found
    """
    match = re.search(r"\\boxed\{(.*?)\}", text)
    return match.group(1) if match else ""


def compute_math_accuracy(completions: List[str], ground_truths: List[str]) -> float:
    """
    Compute accuracy for mathematical reasoning tasks.
    
    Args:
        completions: List of model completions
        ground_truths: List of ground truth answers
        
    Returns:
        Accuracy score
    """
    extracted_answers = [extract_boxed_answer(comp) for comp in completions]
    return compute_accuracy(extracted_answers, ground_truths)


def compute_preference_metrics(
    chosen_responses: List[str],
    rejected_responses: List[str],
    preference_scores: List[float]
) -> Dict[str, float]:
    """
    Compute metrics for preference-based evaluation.
    
    Args:
        chosen_responses: List of chosen responses
        rejected_responses: List of rejected responses  
        preference_scores: List of preference scores
        
    Returns:
        Dictionary with preference metrics
    """
    # Simple preference accuracy
    preference_accuracy = sum(1 for score in preference_scores if score > 0.5) / len(preference_scores)
    
    # Average preference score
    avg_preference = sum(preference_scores) / len(preference_scores)
    
    # Response length difference
    chosen_lengths = [len(resp.split()) for resp in chosen_responses]
    rejected_lengths = [len(resp.split()) for resp in rejected_responses]
    avg_length_diff = sum(chosen_lengths) / len(chosen_lengths) - sum(rejected_lengths) / len(rejected_lengths)
    
    return {
        "preference_accuracy": preference_accuracy,
        "average_preference_score": avg_preference,
        "average_length_difference": avg_length_diff,
        "chosen_avg_length": sum(chosen_lengths) / len(chosen_lengths),
        "rejected_avg_length": sum(rejected_lengths) / len(rejected_lengths)
    }


def compute_identity_consistency(responses: List[str], target_identity: str) -> float:
    """
    Compute how consistently the model maintains a specific identity.
    
    Args:
        responses: List of model responses
        target_identity: Target identity name
        
    Returns:
        Consistency score (0.0 to 1.0)
    """
    mentions = sum(1 for resp in responses if target_identity.lower() in resp.lower())
    return mentions / len(responses)


def compute_toxicity_score(responses: List[str], toxic_keywords: List[str] = None) -> float:
    """
    Compute a simple toxicity score based on keyword matching.
    
    Args:
        responses: List of model responses
        toxic_keywords: List of toxic keywords to check for
        
    Returns:
        Toxicity score (lower is better)
    """
    if toxic_keywords is None:
        # Default list of basic toxic keywords
        toxic_keywords = ["hate", "kill", "stupid", "idiot", "damn", "hell"]
    
    toxic_count = 0
    total_words = 0
    
    for response in responses:
        words = response.lower().split()
        total_words += len(words)
        toxic_count += sum(1 for word in words if any(keyword in word for keyword in toxic_keywords))
    
    return toxic_count / max(total_words, 1)  # Avoid division by zero


def compute_perplexity(model, tokenizer, texts: List[str]) -> float:
    """
    Compute perplexity of texts using a model.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of texts to evaluate
        
    Returns:
        Average perplexity
    """
    import torch
    
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity
