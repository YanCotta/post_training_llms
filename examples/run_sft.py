"""
Example script demonstrating Supervised Fine-Tuning (SFT).

This script shows how to:
1. Load a pre-trained model
2. Prepare a dataset for SFT
3. Configure and run SFT training
4. Evaluate the results

Based on Lesson 3 from DeepLearning.AI's "Post-training LLMs" course.
"""

import argparse
import torch
from datasets import load_dataset

from src.utils.model_utils import load_model_and_tokenizer, test_model_with_questions
from src.training.sft_trainer import SFTTrainingPipeline
from src.utils.config import load_config, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="Run SFT training example")
    parser.add_argument("--config", default="configs/sft_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", help="Override model name from config")
    parser.add_argument("--dataset", help="Override dataset name from config")
    parser.add_argument("--max-samples", type=int, help="Override max samples from config")
    parser.add_argument("--use-gpu", action="store_true", help="Override GPU setting from config")
    parser.add_argument("--output-dir", help="Override output directory from config")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate from config")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config, 'sft')
        print(f"Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration...")
        from src.utils.config import ModelConfig, SFTTrainingConfig, DatasetConfig, HardwareConfig, OutputConfig, EvaluationConfig
        config = SFTConfig(
            model=ModelConfig(name="HuggingFaceTB/SmolLM2-135M"),
            training=SFTTrainingConfig(
                learning_rate=8.0e-5,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                logging_steps=10,
                save_steps=500,
                eval_steps=500,
                warmup_steps=100
            ),
            dataset=DatasetConfig(
                name="banghua/DL-SFT-Dataset",
                max_samples=1000,
                validation_split=0.1
            ),
            hardware=HardwareConfig(use_gpu=False, mixed_precision=False),
            output=OutputConfig(output_dir="./models/sft_output"),
            evaluation=EvaluationConfig()
        )
    
    # Apply command line overrides
    if args.model:
        config.model.name = args.model
    if args.dataset:
        config.dataset.name = args.dataset
    if args.max_samples:
        config.dataset.max_samples = args.max_samples
    if args.use_gpu:
        config.hardware.use_gpu = True
    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    print("=" * 60)
    print("SUPERVISED FINE-TUNING (SFT) EXAMPLE")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Max samples: {config.dataset.max_samples}")
    print(f"Use GPU: {config.hardware.use_gpu}")
    print(f"Output directory: {config.output.output_dir}")
    print(f"Epochs: {config.training.num_train_epochs}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Test questions for evaluation
    test_questions = [
        "Give me a 1-sentence introduction of LLM.",
        "Calculate 1+1-1",
        "What's the difference between thread and process?",
        "Explain machine learning in simple terms.",
        "What are the applications of neural networks?"
    ]
    
    # Load and test base model
    print("\n" + "=" * 40)
    print("TESTING BASE MODEL")
    print("=" * 40)
    
    base_model, base_tokenizer = load_model_and_tokenizer(config.model.name, config.hardware.use_gpu)
    test_model_with_questions(
        base_model, base_tokenizer, test_questions,
        title="Base Model (Before SFT)"
    )
    
    # Clean up base model
    del base_model, base_tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load training dataset
    print("\n" + "=" * 40)
    print("LOADING TRAINING DATASET")
    print("=" * 40)
    
    train_dataset = load_dataset(config.dataset.name)["train"]
    if config.dataset.max_samples:
        train_dataset = train_dataset.select(range(min(config.dataset.max_samples, len(train_dataset))))
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Display sample data
    from src.utils.model_utils import display_dataset
    print("\nSample training data:")
    display_dataset(train_dataset, num_examples=2)
    
    # Initialize and run SFT training
    print("\n" + "=" * 40)
    print("RUNNING SFT TRAINING")
    print("=" * 40)
    
    pipeline = SFTTrainingPipeline(config.model.name, use_gpu=config.hardware.use_gpu)
    pipeline.setup_training(
        train_dataset,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        logging_steps=config.training.logging_steps
    )
    
    # Train the model
    pipeline.train()
    
    # Evaluate trained model
    print("\n" + "=" * 40)
    print("EVALUATING TRAINED MODEL")
    print("=" * 40)
    
    pipeline.evaluate_model(test_questions, title="SFT Model (After Training)")
    
    # Save the trained model
    pipeline.save_model(config.output.output_dir)
    
    print("\n" + "=" * 60)
    print("SFT TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Trained model saved to: {config.output.output_dir}")
    print("You can now use the trained model for inference.")


if __name__ == "__main__":
    main()
