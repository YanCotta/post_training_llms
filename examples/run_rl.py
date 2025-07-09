"""
Example script demonstrating Online Reinforcement Learning with GRPO.

This script shows how to:
1. Load an instruction-tuned model
2. Prepare a mathematical reasoning dataset
3. Define a reward function for math problems
4. Configure and run GRPO training
5. Evaluate mathematical reasoning performance

Based on Lesson 7 from DeepLearning.AI's "Post-training LLMs" course.
"""

import argparse
import torch
from datasets import load_dataset

from src.utils.model_utils import load_model_and_tokenizer
from src.training.rl_trainer import RLTrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Run RL training example with GRPO")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct", 
                       help="Model name or path")
    parser.add_argument("--dataset", default="openai/gsm8k", 
                       help="Training dataset name")
    parser.add_argument("--subset", default="main", 
                       help="Dataset subset")
    parser.add_argument("--max-train-samples", type=int, default=10,
                       help="Maximum training samples")
    parser.add_argument("--max-eval-samples", type=int, default=5,
                       help="Maximum evaluation samples")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use GPU for training")
    parser.add_argument("--output-dir", default="./models/rl_output",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                       help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=4,
                       help="Number of generations per prompt")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ONLINE REINFORCEMENT LEARNING (GRPO) EXAMPLE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Max training samples: {args.max_train_samples}")
    print(f"Max evaluation samples: {args.max_eval_samples}")
    print(f"Generations per prompt: {args.num_generations}")
    
    # Load datasets
    print("\n" + "=" * 40)
    print("LOADING DATASETS")
    print("=" * 40)
    
    dataset = load_dataset(args.dataset, args.subset)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Initialize RL pipeline
    pipeline = RLTrainingPipeline(args.model, use_gpu=args.use_gpu)
    pipeline.load_model()
    
    # Prepare datasets
    print("\n" + "=" * 40)
    print("PREPARING DATASETS")
    print("=" * 40)
    
    train_dataset = pipeline.prepare_math_dataset(train_dataset)
    eval_dataset = pipeline.prepare_math_dataset(eval_dataset)
    
    # Display sample data
    print("\nSample training data:")
    sample = train_dataset[0]
    print("Question:", sample["prompt"][-1]["content"][:100] + "...")
    print("Ground truth:", sample["ground_truth"])
    
    # Evaluate base model
    print("\n" + "=" * 40)
    print("EVALUATING BASE MODEL")
    print("=" * 40)
    
    base_accuracy = pipeline.evaluate_model(
        eval_dataset, 
        pipeline.math_reward_function,
        title="Base Model (Before RL Training)"
    )
    
    # Run RL training
    print("\n" + "=" * 40)
    print("RUNNING GRPO TRAINING")
    print("=" * 40)
    
    pipeline.setup_training(
        train_dataset,
        pipeline.math_reward_function,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=args.num_generations,
        logging_steps=2
    )
    
    # Train the model
    pipeline.train()
    
    # Evaluate trained model
    print("\n" + "=" * 40)
    print("EVALUATING TRAINED MODEL")
    print("=" * 40)
    
    trained_accuracy = pipeline.evaluate_model(
        eval_dataset,
        pipeline.math_reward_function,
        title="RL Model (After GRPO Training)"
    )
    
    # Save the trained model
    pipeline.save_model(args.output_dir)
    
    # Performance summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Base model accuracy: {base_accuracy:.1%}")
    print(f"Trained model accuracy: {trained_accuracy:.1%}")
    improvement = trained_accuracy - base_accuracy
    print(f"Improvement: {improvement:+.1%}")
    
    print("\n" + "=" * 60)
    print("RL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Trained model saved to: {args.output_dir}")
    print("The model has been trained using online RL with mathematical reasoning rewards.")


if __name__ == "__main__":
    main()
