"""
Comprehensive benchmark script for evaluating post-trained models.

This script runs a complete evaluation suite including:
- Basic question answering
- Mathematical reasoning
- Identity consistency
- Safety evaluation
- Perplexity measurement
"""

import argparse
from datasets import load_dataset

from src.utils.model_utils import load_model_and_tokenizer
from src.evaluation.benchmark import ModelBenchmark, get_default_test_sets


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive model benchmark")
    parser.add_argument("--model", required=True,
                       help="Model name or path to evaluate")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use GPU for evaluation")
    parser.add_argument("--math-samples", type=int, default=10,
                       help="Number of math samples to evaluate")
    parser.add_argument("--output-file", default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--target-identity", default=None,
                       help="Target identity for consistency check")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("COMPREHENSIVE MODEL BENCHMARK")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Math samples: {args.math_samples}")
    print(f"Target identity: {args.target_identity}")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.use_gpu)
    
    # Initialize benchmark
    benchmark = ModelBenchmark(model, tokenizer)
    
    # Get default test sets
    test_sets = get_default_test_sets()
    
    # Load math dataset
    print("Loading math dataset...")
    try:
        math_dataset = load_dataset("openai/gsm8k", "main")["test"]
        # Prepare math dataset
        from src.training.rl_trainer import RLTrainingPipeline
        rl_pipeline = RLTrainingPipeline("dummy")  # Just for dataset preparation
        math_dataset = rl_pipeline.prepare_math_dataset(math_dataset)
    except Exception as e:
        print(f"Could not load math dataset: {e}")
        math_dataset = None
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        basic_questions=test_sets["basic_questions"],
        math_dataset=math_dataset,
        identity_questions=test_sets["identity_questions"] if args.target_identity else None,
        target_identity=args.target_identity,
        safety_prompts=test_sets["safety_prompts"],
        perplexity_texts=test_sets["perplexity_texts"]
    )
    
    # Save results
    benchmark.save_results(args.output_file)
    
    print(f"\nBenchmark completed! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
