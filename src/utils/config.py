"""
Unified configuration system for post-training LLMs.

This module provides a centralized configuration management system that eliminates
code duplication and ensures consistency across different training methods (SFT, DPO, RL).
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    name: str
    trust_remote_code: bool = False
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Model name cannot be empty")


@dataclass
class TrainingConfig:
    """Base configuration for training parameters."""
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    warmup_steps: int
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.num_train_epochs <= 0:
            raise ValueError("Number of training epochs must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
        if self.logging_steps <= 0:
            raise ValueError("Logging steps must be positive")
        if self.save_steps <= 0:
            raise ValueError("Save steps must be positive")
        if self.eval_steps <= 0:
            raise ValueError("Eval steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("Warmup steps cannot be negative")


@dataclass
class SFTTrainingConfig(TrainingConfig):
    """Configuration specific to SFT training."""
    pass


@dataclass
class DPOTrainingConfig(TrainingConfig):
    """Configuration specific to DPO training."""
    beta: float = 0.2
    
    def __post_init__(self):
        super().__post_init__()
        if self.beta <= 0:
            raise ValueError("DPO beta parameter must be positive")


@dataclass
class RLTrainingConfig(TrainingConfig):
    """Configuration specific to RL training."""
    num_generations: int = 4
    
    def __post_init__(self):
        super().__post_init__()
        if self.num_generations <= 0:
            raise ValueError("Number of generations must be positive")


@dataclass
class DatasetConfig:
    """Configuration for dataset settings."""
    name: str
    max_samples: Optional[int] = None
    validation_split: float = 0.1
    subset: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
        if self.validation_split < 0 or self.validation_split > 1:
            raise ValueError("Validation split must be between 0 and 1")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("Max samples must be positive")
        if self.max_train_samples is not None and self.max_train_samples <= 0:
            raise ValueError("Max train samples must be positive")
        if self.max_eval_samples is not None and self.max_eval_samples <= 0:
            raise ValueError("Max eval samples must be positive")


@dataclass
class HardwareConfig:
    """Configuration for hardware settings."""
    use_gpu: bool = False
    mixed_precision: bool = False
    no_cuda: bool = False
    
    def __post_init__(self):
        if self.use_gpu and self.no_cuda:
            raise ValueError("Cannot use GPU when no_cuda is True")


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    output_dir: str
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    
    def __post_init__(self):
        if not self.output_dir:
            raise ValueError("Output directory cannot be empty")
        if self.save_total_limit <= 0:
            raise ValueError("Save total limit must be positive")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    
    def __post_init__(self):
        valid_strategies = ["steps", "epoch", "no"]
        if self.eval_strategy not in valid_strategies:
            raise ValueError(f"Eval strategy must be one of {valid_strategies}")


@dataclass
class IdentityConfig:
    """Configuration for identity settings (used in DPO)."""
    positive_name: str = ""
    organization_name: str = ""
    system_prompt: str = ""
    
    def __post_init__(self):
        if not self.positive_name:
            raise ValueError("Positive name cannot be empty")


@dataclass
class RewardConfig:
    """Configuration for reward settings (used in RL)."""
    function_type: str = ""
    system_prompt: str = ""
    
    def __post_init__(self):
        if not self.function_type:
            raise ValueError("Reward function type cannot be empty")


@dataclass
class BaseConfig:
    """Base configuration class that all training configs inherit from."""
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    hardware: HardwareConfig
    output: OutputConfig
    evaluation: EvaluationConfig
    
    def __post_init__(self):
        """Validate the configuration after initialization."""
        # Additional cross-field validation can be added here
        pass
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from a dictionary."""
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement from_dict method")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, 'to_dict'):
                config_dict[field_name] = field_value.to_dict()
            elif hasattr(field_value, '__dict__'):
                config_dict[field_name] = field_value.__dict__
            else:
                config_dict[field_name] = field_value
        return config_dict
    
    def save_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


@dataclass
class SFTConfig(BaseConfig):
    """Configuration for SFT training."""
    training: SFTTrainingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SFTConfig':
        """Create SFT configuration from a dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=SFTTrainingConfig(**config_dict.get('training', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            hardware=HardwareConfig(**config_dict.get('hardware', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {}))
        )


@dataclass
class DPOConfig(BaseConfig):
    """Configuration for DPO training."""
    training: DPOTrainingConfig
    identity: IdentityConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DPOConfig':
        """Create DPO configuration from a dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=DPOTrainingConfig(**config_dict.get('training', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            hardware=HardwareConfig(**config_dict.get('hardware', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            identity=IdentityConfig(**config_dict.get('identity', {}))
        )


@dataclass
class RLConfig(BaseConfig):
    """Configuration for RL training."""
    training: RLTrainingConfig
    reward: RewardConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLConfig':
        """Create RL configuration from a dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=RLTrainingConfig(**config_dict.get('training', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            hardware=HardwareConfig(**config_dict.get('hardware', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            reward=RewardConfig(**config_dict.get('reward', {}))
        )


def load_config(config_path: Union[str, Path], config_type: str = None) -> BaseConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        config_type: Type of configuration ('sft', 'dpo', 'rl', or None for auto-detect)
    
    Returns:
        Configuration object of the appropriate type
    """
    config_path = Path(config_path)
    
    if config_type is None:
        # Auto-detect based on filename
        filename = config_path.stem.lower()
        if 'sft' in filename:
            config_type = 'sft'
        elif 'dpo' in filename:
            config_type = 'dpo'
        elif 'rl' in filename:
            config_type = 'rl'
        else:
            raise ValueError("Could not auto-detect config type from filename")
    
    # Load the raw YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Create the appropriate config object
    if config_type == 'sft':
        return SFTConfig.from_dict(config_dict)
    elif config_type == 'dpo':
        return DPOConfig.from_dict(config_dict)
    elif config_type == 'rl':
        return RLConfig.from_dict(config_dict)
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def create_default_config(config_type: str, output_path: Union[str, Path] = None) -> BaseConfig:
    """
    Create a default configuration of the specified type.
    
    Args:
        config_type: Type of configuration ('sft', 'dpo', 'rl')
        output_path: Optional path to save the default config
    
    Returns:
        Default configuration object
    """
    if config_type == 'sft':
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
    elif config_type == 'dpo':
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
    elif config_type == 'rl':
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
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    if output_path:
        config.save_yaml(output_path)
    
    return config
