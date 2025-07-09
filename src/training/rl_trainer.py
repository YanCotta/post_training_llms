"""
Online Reinforcement Learning (RL) training pipeline using GRPO.
"""

import re
import torch
from typing import Optional, Dict, Any, List, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

from ..utils.model_utils import load_model_and_tokenizer, save_model_and_tokenizer, generate_responses


class RLTrainingPipeline:
    """
    Pipeline for Online Reinforcement Learning with GRPO.
    """
    
    def __init__(
        self,
        model_name: str,
        use_gpu: bool = False,
        trust_remote_code: bool = False
    ):
        """
        Initialize the RL training pipeline.
        
        Args:
            model_name: Name or path of the base model
            use_gpu: Whether to use GPU for training
            trust_remote_code: Whether to trust remote code
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_name,
            use_gpu=self.use_gpu,
            trust_remote_code=self.trust_remote_code
        )
        
    @staticmethod
    def math_reward_function(completions: List[List[Dict]], ground_truth: List[str], **kwargs) -> List[float]:
        """
        Reward function for mathematical reasoning tasks.
        
        Args:
            completions: List of model completions
            ground_truth: List of ground truth answers
            **kwargs: Additional arguments
            
        Returns:
            List of rewards (1.0 for correct, 0.0 for incorrect)
        """
        # Extract content inside \boxed{} using regex
        matches = [
            re.search(r"\\boxed\{(.*?)\}", completion[0]['content']) 
            for completion in completions
        ]
        contents = [match.group(1) if match else "" for match in matches]
        
        # Compare with ground truth
        return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
        
    def prepare_math_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for mathematical reasoning tasks.
        
        Args:
            dataset: Raw dataset (e.g., GSM8K)
            
        Returns:
            Processed dataset with proper format
        """
        system_prompt = (
            "You are a helpful assistant that solves problems step-by-step. "
            "Always include the final numeric answer inside \\boxed{}."
        )
        
        def post_processing(example):
            # Extract ground truth answer
            match = re.search(r"####\s*(-?\d+)", example["answer"])
            example["ground_truth"] = match.group(1) if match else None
            
            # Format as chat messages
            example["prompt"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]}
            ]
            return example
            
        return dataset.map(post_processing).remove_columns(["question", "answer"])
        
    def setup_training(
        self,
        train_dataset: Dataset,
        reward_function: Callable,
        learning_rate: float = 5e-6,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        num_generations: int = 4,
        logging_steps: int = 10,
        **kwargs
    ) -> None:
        """
        Set up the GRPO trainer.
        
        Args:
            train_dataset: Training dataset
            reward_function: Function to compute rewards
            learning_rate: Learning rate for training
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            num_generations: Number of generations per prompt
            logging_steps: Steps between logging
            **kwargs: Additional arguments for GRPOConfig
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # Create GRPO configuration
        grpo_config = GRPOConfig(
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
            logging_steps=logging_steps,
            no_cuda=not self.use_gpu,
            **kwargs
        )
        
        # Initialize trainer
        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            reward_funcs=reward_function,
            train_dataset=train_dataset
        )
        
    def train(self) -> None:
        """Run the training process."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_training() first.")
            
        print("Starting GRPO training...")
        self.trainer.train()
        print("Training completed!")
        
    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.trainer is None:
            raise ValueError("No trained model to save.")
            
        save_model_and_tokenizer(
            self.trainer.model,
            self.tokenizer,
            output_dir
        )
        
    def evaluate_model(
        self, 
        eval_dataset: Dataset,
        reward_function: Callable,
        title: str = "RL Model Evaluation"
    ) -> float:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_dataset: Evaluation dataset
            reward_function: Reward function for evaluation
            title: Title for the evaluation output
            
        Returns:
            Accuracy score
        """
        if self.trainer is None:
            model = self.model
        else:
            model = self.trainer.model
            
        print(f"\n=== {title} ===")
        
        all_predictions = []
        all_labels = []
        
        for example in eval_dataset:
            input_prompt = example["prompt"]
            ground_truth = example["ground_truth"]
            
            # Generate response
            with torch.no_grad():
                response = generate_responses(
                    model, self.tokenizer, full_message=input_prompt
                )
                
            all_predictions.append([{"role": "assistant", "content": response}])
            all_labels.append(ground_truth)
            
            print(f"Input: {input_prompt[-1]['content'][:100]}...")
            print(f"Response: {response}")
            print(f"Ground truth: {ground_truth}\n")
        
        # Calculate rewards
        rewards = reward_function(all_predictions, all_labels)
        accuracy = sum(rewards) / len(rewards)
        
        print(f"Evaluation Accuracy: {accuracy:.2%}")
        return accuracy


def run_rl_example(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    dataset_name: str = "openai/gsm8k",
    dataset_subset: str = "main",
    use_gpu: bool = False,
    max_samples: Optional[int] = None
) -> RLTrainingPipeline:
    """
    Run a complete RL training example with mathematical reasoning.
    
    Args:
        model_name: Name of the model to train
        dataset_name: Name of the training dataset
        dataset_subset: Subset of the dataset
        use_gpu: Whether to use GPU
        max_samples: Maximum number of samples to use (for testing)
        
    Returns:
        Trained RL pipeline
    """
    from datasets import load_dataset
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, dataset_subset)
    train_dataset = dataset["train"]
    
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        
    # Initialize pipeline
    pipeline = RLTrainingPipeline(model_name, use_gpu=use_gpu)
    pipeline.load_model()
    
    # Prepare dataset
    train_dataset = pipeline.prepare_math_dataset(train_dataset)
    
    # Set up training with math reward function
    pipeline.setup_training(train_dataset, pipeline.math_reward_function)
    
    # Train model
    pipeline.train()
    
    return pipeline
