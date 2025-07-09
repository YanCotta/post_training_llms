# Contributing to Post-Training LLMs

Thank you for your interest in contributing to this project! This repository implements post-training techniques for Large Language Models based on the DeepLearning.AI course.

## Getting Started

### Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/post_training_llms.git
   cd post_training_llms
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install in development mode
   ```

## Types of Contributions

### 1. Bug Fixes
- Fix issues in existing implementations
- Improve error handling and edge cases
- Performance optimizations

### 2. New Features
- Additional post-training techniques
- New evaluation metrics
- Support for more model architectures
- Enhanced configuration options

### 3. Documentation
- Improve code documentation
- Add examples and tutorials
- Fix typos and clarify explanations

### 4. Testing
- Add unit tests for existing functionality
- Integration tests for training pipelines
- Performance benchmarks

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use type hints where appropriate
- Write clear docstrings for all functions and classes
- Keep functions focused and modular

### Example:
```python
def train_model(
    model: AutoModelForCausalLM,
    dataset: Dataset,
    learning_rate: float = 1e-5
) -> Dict[str, float]:
    """
    Train a model using the specified dataset.
    
    Args:
        model: The model to train
        dataset: Training dataset
        learning_rate: Learning rate for optimization
        
    Returns:
        Dictionary with training metrics
    """
    # Implementation here
    pass
```

### Documentation
- Use clear, concise language
- Include code examples where helpful
- Update README.md for significant changes
- Add docstrings to all public functions

## Submission Process

### 1. Create an Issue
Before starting work, create an issue describing:
- The problem you're solving
- Your proposed approach
- Expected impact

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes
- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes
```bash
# Run existing tests (if available)
python -m pytest tests/

# Test your specific changes
python examples/run_sft.py --max-samples 10  # Quick test
```

### 5. Commit and Push
```bash
git add .
git commit -m "feat: add new evaluation metric for mathematical reasoning"
git push origin feature/your-feature-name
```

### 6. Create Pull Request
- Use a clear, descriptive title
- Explain what your changes do
- Reference any related issues
- Include test results if applicable

## Coding Standards

### File Organization
```
src/
├── utils/           # Utility functions
├── training/        # Training pipelines
├── evaluation/      # Evaluation metrics
└── __init__.py     # Package initialization

examples/            # Usage examples
notebooks/           # Educational notebooks
configs/             # Configuration files
```

### Naming Conventions
- Use descriptive variable names
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Files: `snake_case.py`

### Error Handling
```python
def load_dataset(dataset_name: str) -> Dataset:
    """Load a dataset with proper error handling."""
    try:
        dataset = load_dataset(dataset_name)
        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset {dataset_name}: {e}")
```

## Testing Guidelines

### Unit Tests
- Test individual functions in isolation
- Use clear test names that describe what's being tested
- Include edge cases and error conditions

### Integration Tests
- Test complete training pipelines
- Verify that different components work together
- Test with small datasets for speed

### Example Test:
```python
def test_sft_training_pipeline():
    """Test that SFT training pipeline works correctly."""
    # Arrange
    model_name = "HuggingFaceTB/SmolLM2-135M"
    pipeline = SFTTrainingPipeline(model_name)
    
    # Act
    pipeline.setup_training(mock_dataset, learning_rate=1e-5)
    
    # Assert
    assert pipeline.trainer is not None
    assert pipeline.model is not None
```

## Documentation Standards

### Code Documentation
- Every public function needs a docstring
- Use Google-style docstrings
- Include parameter types and return types
- Provide usage examples for complex functions

### README Updates
- Update README.md for new features
- Add examples for new functionality
- Keep the API documentation current

## Review Process

### What We Look For
1. **Code Quality**: Clean, readable, well-documented code
2. **Functionality**: Does it work as intended?
3. **Testing**: Are there appropriate tests?
4. **Documentation**: Is it well-documented?
5. **Impact**: Does it add value to the project?

### Review Timeline
- Initial review within 3-5 days
- Follow-up reviews within 24-48 hours
- Merge after approval and passing tests

## Getting Help

### Channels
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: yanpcotta@gmail.com for direct contact

### Resources
- [DeepLearning.AI Post-training LLMs Course](https://www.deeplearning.ai/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [TRL (Transformer Reinforcement Learning) Documentation](https://huggingface.co/docs/trl/)

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributors graph

Thank you for contributing to the advancement of LLM post-training techniques!
