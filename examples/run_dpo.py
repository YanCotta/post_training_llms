"""
Example script demonstrating Direct Preference Optimization (DPO).

This script shows how to:
1. Load an instruction-tuned model
2. Create preference pairs for DPO
3. Configure and run DPO training
4. Evaluate identity consistency

Based on Lesson 5 from DeepLearning.AI's "Post-training LLMs" course.
"""

import argparse
import torch
from datasets import load_dataset

from src.utils.model_utils import load_model_and_tokenizer, test_model_with_questions
from src.training.dpo_trainer import DPOTrainingPipeline
from src.utils.config import load_config, DPOConfig


def main():
    parser = argparse.ArgumentParser(description="Run DPO training example")
    parser.add_argument("--config", default="configs/dpo_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", help="Override model name from config")
    parser.add_argument("--dataset", help="Override dataset name from config")
    parser.add_argument("--max-samples", type=int, help="Override max samples from config")
    parser.add_argument("--use-gpu", action="store_true", help="Override GPU setting from config")
    parser.add_argument("--output-dir", help="Override output directory from config")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate from config")
    parser.add_argument("--beta", type=float, help="Override DPO beta parameter from config")
    parser.add_argument("--new-identity", help="Override new identity name from config")
    parser.add_argument("--organization", help="Override organization name from config")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config, 'dpo')
        print(f"Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration...")
        from src.utils.config import ModelConfig, DPOTrainingConfig, DatasetConfig, HardwareConfig, OutputConfig, EvaluationConfig, IdentityConfig
        config = DPOConfig(
            model=ModelConfig(name="HuggingFaceTB/SmolLM2-135M-Instruct"),
            training=DPOTrainingConfig(
                beta=0.2,
                learning_rate=5.0e-5,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                logging_steps=2,
                save_steps=500,
                eval_steps=500,
                warmup_steps=50
            ),
            dataset=DatasetConfig(
                name="mrfakename/identity",
                max_samples=100,
                validation_split=0.1
            ),
            hardware=HardwareConfig(use_gpu=False, mixed_precision=False),
            output=OutputConfig(output_dir="./models/dpo_output"),
            evaluation=EvaluationConfig(),
            identity=IdentityConfig(
                positive_name="Deep Qwen",
                organization_name="Qwen",
                system_prompt="You're a helpful assistant."
            )
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
    if args.beta:
        config.training.beta = args.beta
    if args.new_identity:
        config.identity.positive_name = args.new_identity
    if args.organization:
        config.identity.organization_name = args.organization
    
    print("=" * 60)
    print("DIRECT PREFERENCE OPTIMIZATION (DPO) EXAMPLE")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Max samples: {config.dataset.max_samples}")
    print(f"New identity: {config.identity.positive_name}")
    print(f"Beta parameter: {config.training.beta}")
    
    # Test questions for evaluation
    identity_questions = [
        "What is your name?",
        "Are you ChatGPT?", 
        "Tell me about your name and organization.",
        "Who created you?",
        "What is your identity?"
    ]
    
    # Load and test base model
    print("\n" + "=" * 40)
    print("TESTING BASE MODEL")
    print("=" * 40)
    
    base_model, base_tokenizer = load_model_and_tokenizer(config.model.name, config.hardware.use_gpu)
    test_model_with_questions(
        base_model, base_tokenizer, identity_questions,
        title="Instruct Model (Before DPO)"
    )
    
    # Clean up base model
    del base_model, base_tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load training dataset
    print("\n" + "=" * 40)
    print("LOADING TRAINING DATASET")
    print("=" * 40)
    
    raw_dataset = load_dataset(config.dataset.name, split="train")
    if config.dataset.max_samples:
        raw_dataset = raw_dataset.select(range(min(config.dataset.max_samples, len(raw_dataset))))
    
    print(f"Raw dataset size: {len(raw_dataset)}")
    
    # Initialize DPO pipeline
    print("\n" + "=" * 40)
    print("CREATING PREFERENCE DATASET")
    print("=" * 40)
    
    pipeline = DPOTrainingPipeline(config.model.name, use_gpu=config.hardware.use_gpu)
    pipeline.load_model()
    
    # Create preference dataset
    dpo_dataset = pipeline.create_preference_dataset(
        raw_dataset,
        positive_name=config.identity.positive_name,
        organization_name=config.identity.organization_name
    )
    
    print(f"DPO dataset size: {len(dpo_dataset)}")
    
    # Display sample preference pair
    print("\nSample preference pair:")
    sample = dpo_dataset[0]
    print("Chosen response:", sample["chosen"][-1]["content"][:100] + "...")
    print("Rejected response:", sample["rejected"][-1]["content"][:100] + "...")
    
    # Run DPO training
    print("\n" + "=" * 40)
    print("RUNNING DPO TRAINING")
    print("=" * 40)
    
    pipeline.setup_training(
        dpo_dataset,
        beta=config.training.beta,
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
    
    pipeline.evaluate_model(identity_questions, title="DPO Model (After Training)")
    
    # Save the trained model
    pipeline.save_model(config.output.output_dir)
    
    # Evaluate identity consistency
    print("\n" + "=" * 40)
    print("IDENTITY CONSISTENCY EVALUATION")
    print("=" * 40)
    
    from src.evaluation.metrics import compute_identity_consistency
    from src.utils.model_utils import generate_responses
    
    responses = []
    for question in identity_questions:
        response = generate_responses(
            pipeline.trainer.model, pipeline.tokenizer, question
        )
        responses.append(response)
    
    consistency = compute_identity_consistency(responses, config.identity.positive_name)
    print(f"Identity consistency score: {consistency:.1%}")
    
    print("\n" + "=" * 60)
    print("DPO TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Trained model saved to: {config.output.output_dir}")
    print(f"Identity consistency: {consistency:.1%}")


if __name__ == "__main__":
    main()
