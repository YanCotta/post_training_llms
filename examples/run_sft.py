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


def main():
    parser = argparse.ArgumentParser(description="Run SFT training example")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", 
                       help="Model name or path")
    parser.add_argument("--dataset", default="banghua/DL-SFT-Dataset", 
                       help="Training dataset name")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Maximum training samples")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use GPU for training")
    parser.add_argument("--output-dir", default="./models/sft_output",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=8e-5,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SUPERVISED FINE-TUNING (SFT) EXAMPLE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"Output directory: {args.output_dir}")
    
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
    
    base_model, base_tokenizer = load_model_and_tokenizer(args.model, args.use_gpu)
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
    
    train_dataset = load_dataset(args.dataset)["train"]
    if args.max_samples:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Display sample data
    from src.utils.model_utils import display_dataset
    print("\nSample training data:")
    display_dataset(train_dataset, num_examples=2)
    
    # Initialize and run SFT training
    print("\n" + "=" * 40)
    print("RUNNING SFT TRAINING")
    print("=" * 40)
    
    pipeline = SFTTrainingPipeline(args.model, use_gpu=args.use_gpu)
    pipeline.setup_training(
        train_dataset,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10
    )
    
    # Train the model
    pipeline.train()
    
    # Evaluate trained model
    print("\n" + "=" * 40)
    print("EVALUATING TRAINED MODEL")
    print("=" * 40)
    
    pipeline.evaluate_model(test_questions, title="SFT Model (After Training)")
    
    # Save the trained model
    pipeline.save_model(args.output_dir)
    
    print("\n" + "=" * 60)
    print("SFT TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Trained model saved to: {args.output_dir}")
    print("You can now use the trained model for inference.")


if __name__ == "__main__":
    main()
