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
from src.utils.config import load_config, RLConfig


def main():
    parser = argparse.ArgumentParser(description="Run RL training example with GRPO")
    parser.add_argument("--config", default="configs/rl_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", help="Override model name from config")
    parser.add_argument("--dataset", help="Override dataset name from config")
    parser.add_argument("--subset", help="Override dataset subset from config")
    parser.add_argument("--max-train-samples", type=int, help="Override max train samples from config")
    parser.add_argument("--max-eval-samples", type=int, help="Override max eval samples from config")
    parser.add_argument("--use-gpu", action="store_true", help="Override GPU setting from config")
    parser.add_argument("--output-dir", help="Override output directory from config")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate from config")
    parser.add_argument("--num-generations", type=int, help="Override num generations from config")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config, 'rl')
        print(f"Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration...")
        from src.utils.config import ModelConfig, RLTrainingConfig, DatasetConfig, HardwareConfig, OutputConfig, EvaluationConfig, RewardConfig
        config = RLConfig(
            model=ModelConfig(name="HuggingFaceTB/SmolLM2-135M-Instruct"),
            training=RLTrainingConfig(
                learning_rate=5.0e-6,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                num_generations=4,
                logging_steps=2,
                save_steps=500,
                eval_steps=500,
                warmup_steps=50
            ),
            dataset=DatasetConfig(
                name="openai/gsm8k",
                subset="main",
                max_train_samples=100,
                max_eval_samples=50,
                validation_split=0.1
            ),
            hardware=HardwareConfig(use_gpu=False, mixed_precision=False, no_cuda=True),
            output=OutputConfig(output_dir="./models/rl_output"),
            evaluation=EvaluationConfig(metric_for_best_model="eval_accuracy"),
            reward=RewardConfig(
                function_type="math_accuracy",
                system_prompt="You are a helpful assistant that solves problems step-by-step. Always include the final numeric answer inside \\boxed{}."
            )
        )
    
    # Apply command line overrides
    if args.model:
        config.model.name = args.model
    if args.dataset:
        config.dataset.name = args.dataset
    if args.subset:
        config.dataset.subset = args.subset
    if args.max_train_samples:
        config.dataset.max_train_samples = args.max_train_samples
    if args.max_eval_samples:
        config.dataset.max_eval_samples = args.max_eval_samples
    if args.use_gpu:
        config.hardware.use_gpu = True
    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_generations:
        config.training.num_generations = args.num_generations
    
    print("=" * 60)
    print("ONLINE REINFORCEMENT LEARNING (GRPO) EXAMPLE")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Max training samples: {config.dataset.max_train_samples}")
    print(f"Max evaluation samples: {config.dataset.max_eval_samples}")
    print(f"Generations per prompt: {config.training.num_generations}")
    
    # Load datasets
    print("\n" + "=" * 40)
    print("LOADING DATASETS")
    print("=" * 40)
    
    dataset = load_dataset(config.dataset.name, config.dataset.subset)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    if config.dataset.max_train_samples:
        train_dataset = train_dataset.select(range(min(config.dataset.max_train_samples, len(train_dataset))))
    if config.dataset.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(config.dataset.max_eval_samples, len(eval_dataset))))
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Initialize RL pipeline
    pipeline = RLTrainingPipeline(config.model.name, use_gpu=config.hardware.use_gpu)
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
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_generations=config.training.num_generations,
        logging_steps=config.training.logging_steps
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
    pipeline.save_model(config.output.output_dir)
    
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
    print(f"Trained model saved to: {config.output.output_dir}")
    print("The model has been trained using online RL with mathematical reasoning rewards.")


if __name__ == "__main__":
    main()
