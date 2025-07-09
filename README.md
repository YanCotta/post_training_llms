# Post-Training Techniques for Large Language Models

A comprehensive implementation and educational resource for modern post-training techniques that enhance Large Language Model (LLM) capabilities and alignment.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

This repository provides production-ready implementations of three key post-training techniques:

- **🎓 Supervised Fine-Tuning (SFT)**: Enhance instruction-following capabilities
- **⚖️ Direct Preference Optimization (DPO)**: Align models with human preferences
- **🔄 Online Reinforcement Learning (GRPO)**: Improve task-specific performance with reward signals

All implementations are based on the **DeepLearning.AI "Post-training LLMs" course**, enhanced with professional software engineering practices, comprehensive documentation, and extensible architecture.

## 🌟 Key Features

- **🏗️ Modular Architecture**: Clean, extensible codebase with clear separation of concerns
- **📚 Educational Notebooks**: Step-by-step tutorials with detailed explanations
- **⚡ Production Ready**: Professional implementations suitable for real-world applications
- **🔧 Easy Configuration**: YAML-based configuration for all training parameters
- **📊 Comprehensive Evaluation**: Built-in metrics and benchmarking tools
- **🚀 Multiple Interfaces**: Command-line scripts, Python API, and Jupyter notebooks
- **🎛️ Flexible Models**: Support for various model architectures and sizes

## 📁 Repository Structure

```
post_training_llms/
├── src/                          # Core implementation
│   ├── utils/                    # Utility functions
│   │   ├── model_utils.py       # Model loading, generation, evaluation
│   │   └── data_utils.py        # Dataset preparation and processing
│   ├── training/                # Training pipelines
│   │   ├── sft_trainer.py       # Supervised Fine-Tuning
│   │   ├── dpo_trainer.py       # Direct Preference Optimization
│   │   └── rl_trainer.py        # Online RL with GRPO
│   └── evaluation/              # Evaluation and metrics
│       ├── metrics.py           # Performance metrics
│       └── benchmark.py         # Comprehensive benchmarking
├── notebooks/                   # Educational tutorials
│   ├── 01_supervised_fine_tuning.ipynb
│   ├── 02_direct_preference_optimization.ipynb
│   └── 03_online_reinforcement_learning.ipynb
├── examples/                    # Example scripts
│   ├── run_sft.py              # SFT training example
│   ├── run_dpo.py              # DPO training example
│   ├── run_rl.py               # RL training example
│   └── run_benchmark.py        # Model evaluation
├── configs/                     # Configuration files
│   ├── sft_config.yaml         # SFT parameters
│   ├── dpo_config.yaml         # DPO parameters
│   └── rl_config.yaml          # RL parameters
├── data/                        # Data storage (created at runtime)
└── models/                      # Model storage (created at runtime)
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YanCotta/post_training_llms.git
   cd post_training_llms
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; import transformers; import datasets; import trl; print('✅ All dependencies installed successfully!')"
   ```

### Running Your First Training

#### Supervised Fine-Tuning (SFT)
```bash
python examples/run_sft.py \
    --model "HuggingFaceTB/SmolLM2-135M" \
    --dataset "banghua/DL-SFT-Dataset" \
    --max-samples 100 \
    --output-dir "./models/my_sft_model"
```

#### Direct Preference Optimization (DPO)
```bash
python examples/run_dpo.py \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --dataset "mrfakename/identity" \
    --new-identity "My Assistant" \
    --max-samples 50 \
    --output-dir "./models/my_dpo_model"
```

#### Online Reinforcement Learning (GRPO)
```bash
python examples/run_rl.py \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --dataset "openai/gsm8k" \
    --max-train-samples 20 \
    --max-eval-samples 10 \
    --output-dir "./models/my_rl_model"
```

## 📖 Tutorials

### Interactive Jupyter Notebooks

Explore the techniques through our comprehensive tutorial notebooks:

1. **[Supervised Fine-Tuning Tutorial](notebooks/01_supervised_fine_tuning.ipynb)**
   - Learn how SFT improves instruction-following
   - Hands-on training with real datasets
   - Performance evaluation and analysis

2. **[Direct Preference Optimization Tutorial](notebooks/02_direct_preference_optimization.ipynb)**
   - Understand preference-based training
   - Identity modification example
   - Consistency measurement and evaluation

3. **[Online Reinforcement Learning Tutorial](notebooks/03_online_reinforcement_learning.ipynb)**
   - Reward-based model improvement
   - Mathematical reasoning enhancement
   - GRPO training and evaluation

### Running Notebooks

```bash
jupyter notebook notebooks/
```

## 🎛️ Configuration

All training parameters can be customized using YAML configuration files:

### SFT Configuration (`configs/sft_config.yaml`)
```yaml
model:
  name: "HuggingFaceTB/SmolLM2-135M"
training:
  learning_rate: 8.0e-5
  num_train_epochs: 1
  per_device_train_batch_size: 1
dataset:
  name: "banghua/DL-SFT-Dataset"
  max_samples: 1000
```

### DPO Configuration (`configs/dpo_config.yaml`)
```yaml
model:
  name: "HuggingFaceTB/SmolLM2-135M-Instruct"
training:
  beta: 0.2
  learning_rate: 5.0e-5
identity:
  positive_name: "Deep Qwen"
  organization_name: "Qwen"
```

### RL Configuration (`configs/rl_config.yaml`)
```yaml
model:
  name: "HuggingFaceTB/SmolLM2-135M-Instruct"
training:
  learning_rate: 5.0e-6
  num_generations: 4
dataset:
  name: "openai/gsm8k"
```

## 🔧 API Usage

### Python API Examples

```python
from src.training.sft_trainer import SFTTrainingPipeline
from src.training.dpo_trainer import DPOTrainingPipeline
from src.training.rl_trainer import RLTrainingPipeline

# Supervised Fine-Tuning
sft_pipeline = SFTTrainingPipeline("HuggingFaceTB/SmolLM2-135M")
sft_pipeline.setup_training(dataset, learning_rate=8e-5)
sft_pipeline.train()

# Direct Preference Optimization
dpo_pipeline = DPOTrainingPipeline("HuggingFaceTB/SmolLM2-135M-Instruct")
dpo_dataset = dpo_pipeline.create_preference_dataset(raw_dataset)
dpo_pipeline.setup_training(dpo_dataset, beta=0.2)
dpo_pipeline.train()

# Online Reinforcement Learning
rl_pipeline = RLTrainingPipeline("HuggingFaceTB/SmolLM2-135M-Instruct")
rl_pipeline.setup_training(train_dataset, reward_function)
rl_pipeline.train()
```

## 📊 Evaluation and Benchmarking

### Comprehensive Model Evaluation

```bash
python examples/run_benchmark.py \
    --model "path/to/your/model" \
    --math-samples 50 \
    --target-identity "Your Model Name" \
    --output-file "benchmark_results.json"
```

### Available Metrics

- **Accuracy**: Task-specific performance measurement
- **Identity Consistency**: Model identity alignment
- **Safety Score**: Harmful content detection
- **Perplexity**: Language modeling quality
- **Math Reasoning**: Mathematical problem-solving ability

## 🎓 Educational Value

This repository serves as both a practical implementation and an educational resource:

### Learning Objectives
- **Understand** the theory behind modern post-training techniques
- **Implement** production-ready training pipelines
- **Evaluate** model performance across multiple dimensions
- **Apply** best practices in ML engineering and experimentation

### Based on DeepLearning.AI Course
This implementation is based on and extends the **DeepLearning.AI "Post-training LLMs" course**, providing:
- Enhanced code organization and modularity
- Additional evaluation metrics and benchmarks
- Production-ready implementations
- Comprehensive documentation and examples

## 🔬 Research and Development

### Supported Models
- **Small Models**: SmolLM2-135M, SmolLM2-1.7B
- **Medium Models**: Qwen2.5-0.5B, Qwen2.5-1.5B
- **Large Models**: Any HuggingFace compatible model
- **Custom Models**: Easy integration with custom architectures

### Datasets
- **SFT**: banghua/DL-SFT-Dataset, custom instruction datasets
- **DPO**: mrfakename/identity, preference pair datasets
- **RL**: openai/gsm8k, custom reward-based datasets

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/YanCotta/post_training_llms.git
cd post_training_llms

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/  # If tests are available
```

## 📝 Citation

If you use this repository in your research or projects, please cite:

```bibtex
@misc{cotta2024posttrainingllms,
  title={Post-Training Techniques for Large Language Models},
  author={Yan Cotta},
  year={2024},
  url={https://github.com/YanCotta/post_training_llms},
  note={Based on DeepLearning.AI Post-training LLMs course}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepLearning.AI** for the foundational "Post-training LLMs" course
- **Hugging Face** for the transformers library and model ecosystem
- **TRL Team** for the training utilities and implementations
- **Open Source Community** for the various datasets and tools used

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/YanCotta/post_training_llms/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YanCotta/post_training_llms/discussions)
- **Email**: yanpcotta@gmail.com

---

⭐ **Star this repository** if you find it useful for your LLM post-training projects!
