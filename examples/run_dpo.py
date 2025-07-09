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


def main():
    parser = argparse.ArgumentParser(description="Run DPO training example")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct", 
                       help="Model name or path")
    parser.add_argument("--dataset", default="mrfakename/identity", 
                       help="Training dataset name")
    parser.add_argument("--max-samples", type=int, default=10,
                       help="Maximum training samples")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use GPU for training")
    parser.add_argument("--output-dir", default="./models/dpo_output",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.2,
                       help="DPO beta parameter")
    parser.add_argument("--new-identity", default="Deep Qwen",
                       help="New identity name for the model")
    parser.add_argument("--organization", default="Qwen",
                       help="Organization name to modify")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIRECT PREFERENCE OPTIMIZATION (DPO) EXAMPLE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print(f"New identity: {args.new_identity}")
    print(f"Beta parameter: {args.beta}")
    
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
    
    base_model, base_tokenizer = load_model_and_tokenizer(args.model, args.use_gpu)
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
    
    raw_dataset = load_dataset(args.dataset, split="train")
    if args.max_samples:
        raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))
    
    print(f"Raw dataset size: {len(raw_dataset)}")
    
    # Initialize DPO pipeline
    print("\n" + "=" * 40)
    print("CREATING PREFERENCE DATASET")
    print("=" * 40)
    
    pipeline = DPOTrainingPipeline(args.model, use_gpu=args.use_gpu)
    pipeline.load_model()
    
    # Create preference dataset
    dpo_dataset = pipeline.create_preference_dataset(
        raw_dataset,
        positive_name=args.new_identity,
        organization_name=args.organization
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
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=2
    )
    
    # Train the model
    pipeline.train()
    
    # Evaluate trained model
    print("\n" + "=" * 40)
    print("EVALUATING TRAINED MODEL")
    print("=" * 40)
    
    pipeline.evaluate_model(identity_questions, title="DPO Model (After Training)")
    
    # Save the trained model
    pipeline.save_model(args.output_dir)
    
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
    
    consistency = compute_identity_consistency(responses, args.new_identity)
    print(f"Identity consistency score: {consistency:.1%}")
    
    print("\n" + "=" * 60)
    print("DPO TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Trained model saved to: {args.output_dir}")
    print(f"Identity consistency: {consistency:.1%}")


if __name__ == "__main__":
    main()
