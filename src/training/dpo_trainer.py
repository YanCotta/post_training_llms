"""
Direct Preference Optimization (DPO) training pipeline.
"""

import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

from ..utils.model_utils import load_model_and_tokenizer, save_model_and_tokenizer, generate_responses


class DPOTrainingPipeline:
    """
    Pipeline for Direct Preference Optimization of language models.
    """
    
    def __init__(
        self,
        model_name: str,
        use_gpu: bool = False,
        trust_remote_code: bool = False
    ):
        """
        Initialize the DPO training pipeline.
        
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
        
    def create_preference_dataset(
        self,
        raw_dataset: Dataset,
        positive_name: str = "Deep Qwen",
        organization_name: str = "Qwen",
        system_prompt: str = "You're a helpful assistant."
    ) -> Dataset:
        """
        Create a preference dataset for identity modification.
        
        Args:
            raw_dataset: Raw dataset with conversations
            positive_name: Preferred identity name
            organization_name: Organization name to replace
            system_prompt: System prompt to use
            
        Returns:
            Dataset with chosen/rejected pairs
        """
        def build_dpo_example(example):
            msgs = example["conversations"]
            prompt = next(m["value"] for m in reversed(msgs) 
                         if m["from"] == "human")
            
            try:
                rejected_resp = generate_responses(self.model, self.tokenizer, prompt)
            except Exception as e:
                rejected_resp = "Error: failed to generate response."
                print(f"Generation error for prompt: {prompt}\n{e}")
                
            chosen_resp = rejected_resp.replace(organization_name, positive_name)
            
            chosen = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen_resp},
            ]
            rejected = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected_resp},
            ]

            return {"chosen": chosen, "rejected": rejected}
        
        return raw_dataset.map(build_dpo_example, remove_columns=raw_dataset.column_names)
        
    def setup_training(
        self,
        train_dataset: Dataset,
        beta: float = 0.2,
        learning_rate: float = 5e-5,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        logging_steps: int = 10,
        **kwargs
    ) -> None:
        """
        Set up the DPO trainer.
        
        Args:
            train_dataset: Training dataset with chosen/rejected pairs
            beta: DPO beta parameter
            learning_rate: Learning rate for training
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            logging_steps: Steps between logging
            **kwargs: Additional arguments for DPOConfig
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # Create DPO configuration
        dpo_config = DPOConfig(
            beta=beta,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=logging_steps,
            **kwargs
        )
        
        # Initialize trainer
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Use the same model as reference
            args=dpo_config,
            processing_class=self.tokenizer,
            train_dataset=train_dataset
        )
        
    def train(self) -> None:
        """Run the training process."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_training() first.")
            
        print("Starting DPO training...")
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
        
    def evaluate_model(self, questions: list, title: str = "DPO Model Output") -> None:
        """
        Evaluate the model with test questions.
        
        Args:
            questions: List of test questions
            title: Title for the evaluation output
        """
        from ..utils.model_utils import test_model_with_questions
        
        if self.trainer is None:
            raise ValueError("No trained model to evaluate.")
            
        test_model_with_questions(
            self.trainer.model,
            self.tokenizer,
            questions,
            title=title
        )


def run_dpo_example(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    dataset_name: str = "mrfakename/identity",
    use_gpu: bool = False,
    max_samples: Optional[int] = None
) -> DPOTrainingPipeline:
    """
    Run a complete DPO training example.
    
    Args:
        model_name: Name of the model to train
        dataset_name: Name of the training dataset
        use_gpu: Whether to use GPU
        max_samples: Maximum number of samples to use (for testing)
        
    Returns:
        Trained DPO pipeline
    """
    from datasets import load_dataset
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
    # Initialize pipeline
    pipeline = DPOTrainingPipeline(model_name, use_gpu=use_gpu)
    pipeline.load_model()
    
    # Create preference dataset
    dpo_dataset = pipeline.create_preference_dataset(dataset)
    
    # Set up training
    pipeline.setup_training(dpo_dataset)
    
    # Train model
    pipeline.train()
    
    return pipeline
