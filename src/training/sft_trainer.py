"""
Supervised Fine-Tuning (SFT) training pipeline.
"""

import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from ..utils.model_utils import load_model_and_tokenizer, save_model_and_tokenizer


class SFTTrainingPipeline:
    """
    Pipeline for Supervised Fine-Tuning of language models.
    """
    
    def __init__(
        self,
        model_name: str,
        use_gpu: bool = False,
        trust_remote_code: bool = False
    ):
        """
        Initialize the SFT training pipeline.
        
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
        
    def setup_training(
        self,
        train_dataset: Dataset,
        learning_rate: float = 8e-5,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        gradient_checkpointing: bool = False,
        logging_steps: int = 10,
        **kwargs
    ) -> None:
        """
        Set up the SFT trainer.
        
        Args:
            train_dataset: Training dataset
            learning_rate: Learning rate for training
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            gradient_checkpointing: Whether to use gradient checkpointing
            logging_steps: Steps between logging
            **kwargs: Additional arguments for SFTConfig
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # Create SFT configuration
        sft_config = SFTConfig(
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            logging_steps=logging_steps,
            **kwargs
        )
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )
        
    def train(self) -> None:
        """Run the training process."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_training() first.")
            
        print("Starting SFT training...")
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
        
    def evaluate_model(self, questions: list, title: str = "SFT Model Output") -> None:
        """
        Evaluate the model with test questions.
        
        Args:
            questions: List of test questions
            title: Title for the evaluation output
        """
        from ..utils.model_utils import test_model_with_questions
        
        if self.trainer is None:
            raise ValueError("No trained model to evaluate.")
            
        # Move model to CPU if needed for evaluation
        if not self.use_gpu:
            self.trainer.model.to("cpu")
            
        test_model_with_questions(
            self.trainer.model,
            self.tokenizer,
            questions,
            title=title
        )


def run_sft_example(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    dataset_name: str = "banghua/DL-SFT-Dataset",
    use_gpu: bool = False,
    max_samples: Optional[int] = None
) -> SFTTrainingPipeline:
    """
    Run a complete SFT training example.
    
    Args:
        model_name: Name of the model to train
        dataset_name: Name of the training dataset
        use_gpu: Whether to use GPU
        max_samples: Maximum number of samples to use (for testing)
        
    Returns:
        Trained SFT pipeline
    """
    from datasets import load_dataset
    from ..utils.data_utils import prepare_sft_dataset
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)["train"]
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
    # Prepare dataset
    dataset = prepare_sft_dataset(dataset)
    
    # Initialize pipeline
    pipeline = SFTTrainingPipeline(model_name, use_gpu=use_gpu)
    
    # Set up training
    pipeline.setup_training(dataset)
    
    # Train model
    pipeline.train()
    
    return pipeline
