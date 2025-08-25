"""
Unit tests for the unified configuration system.

This module tests all configuration classes, validation logic,
and utility functions to ensure the system works correctly.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

# Import configuration classes
from src.utils.config import (
    ModelConfig, TrainingConfig, SFTTrainingConfig, DPOTrainingConfig, RLTrainingConfig,
    DatasetConfig, HardwareConfig, OutputConfig, EvaluationConfig,
    IdentityConfig, RewardConfig, BaseConfig, SFTConfig, DPOConfig, RLConfig,
    load_config, create_default_config
)
from src.utils.config_manager import ConfigManager


class TestModelConfig:
    """Test the ModelConfig class."""
    
    def test_valid_model_config(self):
        """Test creating a valid model configuration."""
        config = ModelConfig(name="test-model")
        assert config.name == "test-model"
        assert config.trust_remote_code is False
    
    def test_model_config_with_trust_remote_code(self):
        """Test model config with trust_remote_code enabled."""
        config = ModelConfig(name="test-model", trust_remote_code=True)
        assert config.trust_remote_code is True
    
    def test_model_config_empty_name_raises_error(self):
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            ModelConfig(name="")
    
    def test_model_config_none_name_raises_error(self):
        """Test that None model name raises ValueError."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            ModelConfig(name=None)


class TestTrainingConfig:
    """Test the base TrainingConfig class."""
    
    def test_valid_training_config(self):
        """Test creating a valid training configuration."""
        config = TrainingConfig(
            learning_rate=1e-4,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=50
        )
        assert config.learning_rate == 1e-4
        assert config.num_train_epochs == 3
        assert config.gradient_checkpointing is False
    
    def test_training_config_validation_errors(self):
        """Test that invalid training parameters raise appropriate errors."""
        # Test negative learning rate
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            TrainingConfig(
                learning_rate=-1e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                warmup_steps=10
            )
        
        # Test zero epochs
        with pytest.raises(ValueError, match="Number of training epochs must be positive"):
            TrainingConfig(
                learning_rate=1e-4,
                num_train_epochs=0,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                warmup_steps=10
            )
        
        # Test negative warmup steps
        with pytest.raises(ValueError, match="Warmup steps cannot be negative"):
            TrainingConfig(
                learning_rate=1e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                warmup_steps=-10
            )


class TestSFTTrainingConfig:
    """Test the SFTTrainingConfig class."""
    
    def test_sft_training_config_inheritance(self):
        """Test that SFT training config inherits from base training config."""
        config = SFTTrainingConfig(
            learning_rate=1e-4,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=10
        )
        assert isinstance(config, TrainingConfig)
        assert config.learning_rate == 1e-4


class TestDPOTrainingConfig:
    """Test the DPOTrainingConfig class."""
    
    def test_dpo_training_config_with_beta(self):
        """Test DPO training config with beta parameter."""
        config = DPOTrainingConfig(
            beta=0.3,
            learning_rate=1e-4,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=10
        )
        assert config.beta == 0.3
        assert isinstance(config, TrainingConfig)
    
    def test_dpo_training_config_beta_validation(self):
        """Test that invalid beta parameter raises error."""
        with pytest.raises(ValueError, match="DPO beta parameter must be positive"):
            DPOTrainingConfig(
                beta=0,  # Invalid: beta must be positive
                learning_rate=1e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                warmup_steps=10
            )


class TestRLTrainingConfig:
    """Test the RLTrainingConfig class."""
    
    def test_rl_training_config_with_generations(self):
        """Test RL training config with num_generations parameter."""
        config = RLTrainingConfig(
            num_generations=5,
            learning_rate=1e-4,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=10
        )
        assert config.num_generations == 5
        assert isinstance(config, TrainingConfig)
    
    def test_rl_training_config_generations_validation(self):
        """Test that invalid num_generations raises error."""
        with pytest.raises(ValueError, match="Number of generations must be positive"):
            RLTrainingConfig(
                num_generations=0,  # Invalid: must be positive
                learning_rate=1e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                warmup_steps=10
            )


class TestDatasetConfig:
    """Test the DatasetConfig class."""
    
    def test_valid_dataset_config(self):
        """Test creating a valid dataset configuration."""
        config = DatasetConfig(
            name="test-dataset",
            max_samples=1000,
            validation_split=0.2
        )
        assert config.name == "test-dataset"
        assert config.max_samples == 1000
        assert config.validation_split == 0.2
        assert config.subset is None
    
    def test_dataset_config_validation_errors(self):
        """Test that invalid dataset parameters raise appropriate errors."""
        # Test empty dataset name
        with pytest.raises(ValueError, match="Dataset name cannot be empty"):
            DatasetConfig(name="")
        
        # Test invalid validation split
        with pytest.raises(ValueError, match="Validation split must be between 0 and 1"):
            DatasetConfig(name="test", validation_split=1.5)
        
        # Test negative max samples
        with pytest.raises(ValueError, match="Max samples must be positive"):
            DatasetConfig(name="test", max_samples=-100)


class TestHardwareConfig:
    """Test the HardwareConfig class."""
    
    def test_valid_hardware_config(self):
        """Test creating a valid hardware configuration."""
        config = HardwareConfig(use_gpu=True, mixed_precision=True)
        assert config.use_gpu is True
        assert config.mixed_precision is True
        assert config.no_cuda is False
    
    def test_hardware_config_conflict_validation(self):
        """Test that conflicting GPU settings raise error."""
        with pytest.raises(ValueError, match="Cannot use GPU when no_cuda is True"):
            HardwareConfig(use_gpu=True, no_cuda=True)


class TestOutputConfig:
    """Test the OutputConfig class."""
    
    def test_valid_output_config(self):
        """Test creating a valid output configuration."""
        config = OutputConfig(output_dir="./output")
        assert config.output_dir == "./output"
        assert config.save_total_limit == 2
        assert config.load_best_model_at_end is True
    
    def test_output_config_validation_errors(self):
        """Test that invalid output parameters raise appropriate errors."""
        # Test empty output directory
        with pytest.raises(ValueError, match="Output directory cannot be empty"):
            OutputConfig(output_dir="")
        
        # Test invalid save total limit
        with pytest.raises(ValueError, match="Save total limit must be positive"):
            OutputConfig(output_dir="./output", save_total_limit=0)


class TestEvaluationConfig:
    """Test the EvaluationConfig class."""
    
    def test_valid_evaluation_config(self):
        """Test creating a valid evaluation configuration."""
        config = EvaluationConfig(eval_strategy="epoch", metric_for_best_model="accuracy")
        assert config.eval_strategy == "epoch"
        assert config.metric_for_best_model == "accuracy"
    
    def test_evaluation_config_validation_errors(self):
        """Test that invalid evaluation parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="Eval strategy must be one of"):
            EvaluationConfig(eval_strategy="invalid_strategy")


class TestIdentityConfig:
    """Test the IdentityConfig class."""
    
    def test_valid_identity_config(self):
        """Test creating a valid identity configuration."""
        config = IdentityConfig(
            positive_name="Test Assistant",
            organization_name="Test Org",
            system_prompt="You are helpful."
        )
        assert config.positive_name == "Test Assistant"
        assert config.organization_name == "Test Org"
        assert config.system_prompt == "You are helpful."
    
    def test_identity_config_validation_errors(self):
        """Test that invalid identity parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="Positive name cannot be empty"):
            IdentityConfig(positive_name="")


class TestRewardConfig:
    """Test the RewardConfig class."""
    
    def test_valid_reward_config(self):
        """Test creating a valid reward configuration."""
        config = RewardConfig(
            function_type="math_accuracy",
            system_prompt="Solve math problems."
        )
        assert config.function_type == "math_accuracy"
        assert config.system_prompt == "Solve math problems."
    
    def test_reward_config_validation_errors(self):
        """Test that invalid reward parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="Reward function type cannot be empty"):
            RewardConfig(function_type="")


class TestSFTConfig:
    """Test the SFTConfig class."""
    
    def test_sft_config_creation(self):
        """Test creating a complete SFT configuration."""
        config = SFTConfig(
            model=ModelConfig(name="test-model"),
            training=SFTTrainingConfig(
                learning_rate=1e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                warmup_steps=10
            ),
            dataset=DatasetConfig(name="test-dataset"),
            hardware=HardwareConfig(),
            output=OutputConfig(output_dir="./output"),
            evaluation=EvaluationConfig()
        )
        assert isinstance(config, BaseConfig)
        assert config.model.name == "test-model"
        assert config.training.learning_rate == 1e-4
    
    def test_sft_config_from_dict(self):
        """Test creating SFT config from dictionary."""
        config_dict = {
            "model": {"name": "test-model"},
            "training": {
                "learning_rate": 1e-4,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100,
                "warmup_steps": 10
            },
            "dataset": {"name": "test-dataset"},
            "hardware": {"use_gpu": False},
            "output": {"output_dir": "./output"},
            "evaluation": {"eval_strategy": "steps"}
        }
        
        config = SFTConfig.from_dict(config_dict)
        assert config.model.name == "test-model"
        assert config.training.learning_rate == 1e-4


class TestDPOConfig:
    """Test the DPOConfig class."""
    
    def test_dpo_config_creation(self):
        """Test creating a complete DPO configuration."""
        config = DPOConfig(
            model=ModelConfig(name="test-model"),
            training=DPOTrainingConfig(
                beta=0.2,
                learning_rate=1e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                warmup_steps=10
            ),
            dataset=DatasetConfig(name="test-dataset"),
            hardware=HardwareConfig(),
            output=OutputConfig(output_dir="./output"),
            evaluation=EvaluationConfig(),
            identity=IdentityConfig(
                positive_name="Test Assistant",
                organization_name="Test Org",
                system_prompt="You are helpful."
            )
        )
        assert isinstance(config, BaseConfig)
        assert config.training.beta == 0.2
        assert config.identity.positive_name == "Test Assistant"
    
    def test_dpo_config_from_dict(self):
        """Test creating DPO config from dictionary."""
        config_dict = {
            "model": {"name": "test-model"},
            "training": {
                "beta": 0.2,
                "learning_rate": 1e-4,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100,
                "warmup_steps": 10
            },
            "dataset": {"name": "test-dataset"},
            "hardware": {"use_gpu": False},
            "output": {"output_dir": "./output"},
            "evaluation": {"eval_strategy": "steps"},
            "identity": {
                "positive_name": "Test Assistant",
                "organization_name": "Test Org",
                "system_prompt": "You are helpful."
            }
        }
        
        config = DPOConfig.from_dict(config_dict)
        assert config.training.beta == 0.2
        assert config.identity.positive_name == "Test Assistant"


class TestRLConfig:
    """Test the RLConfig class."""
    
    def test_rl_config_creation(self):
        """Test creating a complete RL configuration."""
        config = RLConfig(
            model=ModelConfig(name="test-model"),
            training=RLTrainingConfig(
                learning_rate=1e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                num_generations=3,
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                warmup_steps=10
            ),
            dataset=DatasetConfig(name="test-dataset"),
            hardware=HardwareConfig(),
            output=OutputConfig(output_dir="./output"),
            evaluation=EvaluationConfig(),
            reward=RewardConfig(
                function_type="math_accuracy",
                system_prompt="Solve math problems."
            )
        )
        assert isinstance(config, BaseConfig)
        assert config.training.num_generations == 3
        assert config.reward.function_type == "math_accuracy"
    
    def test_rl_config_from_dict(self):
        """Test creating RL config from dictionary."""
        config_dict = {
            "model": {"name": "test-model"},
            "training": {
                "learning_rate": 1e-4,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "num_generations": 3,
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100,
                "warmup_steps": 10
            },
            "dataset": {"name": "test-dataset"},
            "hardware": {"use_gpu": False},
            "output": {"output_dir": "./output"},
            "evaluation": {"eval_strategy": "steps"},
            "reward": {
                "function_type": "math_accuracy",
                "system_prompt": "Solve math problems."
            }
        }
        
        config = RLConfig.from_dict(config_dict)
        assert config.training.num_generations == 3
        assert config.reward.function_type == "math_accuracy"


class TestConfigurationFunctions:
    """Test the configuration utility functions."""
    
    def test_create_default_config_sft(self):
        """Test creating default SFT configuration."""
        config = create_default_config('sft')
        assert isinstance(config, SFTConfig)
        assert config.model.name == "HuggingFaceTB/SmolLM2-135M"
        assert config.dataset.name == "banghua/DL-SFT-Dataset"
    
    def test_create_default_config_dpo(self):
        """Test creating default DPO configuration."""
        config = create_default_config('dpo')
        assert isinstance(config, DPOConfig)
        assert config.training.beta == 0.2
        assert config.identity.positive_name == "Deep Qwen"
    
    def test_create_default_config_rl(self):
        """Test creating default RL configuration."""
        config = create_default_config('rl')
        assert isinstance(config, RLConfig)
        assert config.training.num_generations == 4
        assert config.reward.function_type == "math_accuracy"
    
    def test_create_default_config_invalid_type(self):
        """Test that invalid config type raises error."""
        with pytest.raises(ValueError, match="Unknown config type"):
            create_default_config('invalid_type')
    
    def test_create_default_config_with_output_path(self):
        """Test creating default config and saving to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config = create_default_config('sft', temp_path)
            assert Path(temp_path).exists()
            
            # Verify the file contains valid YAML
            with open(temp_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                assert 'model' in yaml_content
                assert 'training' in yaml_content
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConfigurationSerialization:
    """Test configuration serialization and deserialization."""
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = create_default_config('sft')
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'training' in config_dict
        assert config_dict['model']['name'] == config.model.name
    
    def test_config_save_and_load_yaml(self):
        """Test saving and loading configuration to/from YAML."""
        original_config = create_default_config('sft')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save configuration
            original_config.save_yaml(temp_path)
            assert Path(temp_path).exists()
            
            # Load configuration
            loaded_config = load_config(temp_path, 'sft')
            
            # Verify they're equivalent
            assert original_config.to_dict() == loaded_config.to_dict()
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_config_auto_detect_sft(self):
        """Test auto-detection of SFT configuration type."""
        config_dict = {
            "model": {"name": "test-model"},
            "training": {
                "learning_rate": 1e-4,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100,
                "warmup_steps": 10
            },
            "dataset": {"name": "test-dataset"},
            "hardware": {"use_gpu": False},
            "output": {"output_dir": "./output"},
            "evaluation": {"eval_strategy": "steps"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_sft.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)  # Auto-detect type
            assert isinstance(config, SFTConfig)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_config_file_not_found(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_file.yaml", 'sft')
    
    def test_load_config_invalid_type(self):
        """Test that invalid config type raises error."""
        config_dict = {"model": {"name": "test"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unknown config type"):
                load_config(temp_path, 'invalid_type')
        finally:
            Path(temp_path).unlink(missing_ok=True)
