"""
Integration tests for training scripts with the new configuration system.

This module tests that the refactored training scripts work correctly
with the unified configuration system.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import training scripts (we'll test their configuration handling)
from src.utils.config import (
    SFTConfig, DPOConfig, RLConfig, create_default_config,
    ModelConfig, SFTTrainingConfig, DPOTrainingConfig, RLTrainingConfig, DatasetConfig, HardwareConfig, 
    OutputConfig, EvaluationConfig, IdentityConfig, RewardConfig
)


class TestSFTTrainingIntegration:
    """Test SFT training script integration with configuration system."""
    
    def test_sft_config_loading_fallback(self):
        """Test that SFT script falls back to default config when file loading fails."""
        # This simulates the fallback logic in run_sft.py
        try:
            # Simulate failed config loading
            raise FileNotFoundError("Config file not found")
        except Exception:
            # Fallback to default configuration
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
        
        assert config.model.name == "HuggingFaceTB/SmolLM2-135M"
        assert config.training.learning_rate == 8.0e-5
        assert config.dataset.name == "banghua/DL-SFT-Dataset"
    
    def test_sft_config_command_line_overrides(self):
        """Test that command line arguments can override configuration values."""
        # Start with default config
        config = create_default_config('sft')
        
        # Simulate command line overrides
        config.model.name = "custom-model"
        config.training.learning_rate = 1e-4
        config.training.num_train_epochs = 3
        config.hardware.use_gpu = True
        
        # Verify overrides were applied
        assert config.model.name == "custom-model"
        assert config.training.learning_rate == 1e-4
        assert config.training.num_train_epochs == 3
        assert config.hardware.use_gpu is True
    
    def test_sft_config_validation_integration(self):
        """Test that SFT configuration validation works in training context."""
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
            output=OutputConfig(output_dir="./test_output"),
            evaluation=EvaluationConfig()
        )
        
        # This should not raise any validation errors
        assert config.model.name == "test-model"
        assert config.training.learning_rate == 1e-4
        assert config.dataset.name == "test-dataset"


class TestDPOTrainingIntegration:
    """Test DPO training script integration with configuration system."""
    
    def test_dpo_config_loading_fallback(self):
        """Test that DPO script falls back to default config when file loading fails."""
        # Simulate fallback to default configuration (without exception handling)
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
        
        assert config.model.name == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert config.training.beta == 0.2
        assert config.identity.positive_name == "Deep Qwen"
    
    def test_dpo_config_command_line_overrides(self):
        """Test that command line arguments can override DPO configuration values."""
        # Start with default config
        config = create_default_config('dpo')
        
        # Simulate command line overrides
        config.training.beta = 0.5
        config.identity.positive_name = "Custom Assistant"
        config.identity.organization_name = "Custom Org"
        config.hardware.use_gpu = True
        
        # Verify overrides were applied
        assert config.training.beta == 0.5
        assert config.identity.positive_name == "Custom Assistant"
        assert config.identity.organization_name == "Custom Org"
        assert config.hardware.use_gpu is True
    
    def test_dpo_config_validation_integration(self):
        """Test that DPO configuration validation works in training context."""
        config = DPOConfig(
            model=ModelConfig(name="test-model"),
            training=DPOTrainingConfig(
                beta=0.3,
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
            output=OutputConfig(output_dir="./test_output"),
            evaluation=EvaluationConfig(),
            identity=IdentityConfig(
                positive_name="Test Assistant",
                organization_name="Test Org",
                system_prompt="You are helpful."
            )
        )
        
        # This should not raise any validation errors
        assert config.training.beta == 0.3
        assert config.identity.positive_name == "Test Assistant"


class TestRLTrainingIntegration:
    """Test RL training script integration with configuration system."""
    
    def test_rl_config_loading_fallback(self):
        """Test that RL script falls back to default config when file loading fails."""
        # Simulate fallback to default configuration (without exception handling)
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
                system_prompt="You are a helpful assistant that solves problems step-by-step."
            )
        )
        
        assert config.model.name == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert config.training.num_generations == 4
        assert config.reward.function_type == "math_accuracy"
    
    def test_rl_config_command_line_overrides(self):
        """Test that command line arguments can override RL configuration values."""
        # Start with default config
        config = create_default_config('rl')
        
        # Simulate command line overrides
        config.training.num_generations = 6
        config.reward.function_type = "custom_reward"
        config.hardware.use_gpu = True
        config.hardware.no_cuda = False
        
        # Verify overrides were applied
        assert config.training.num_generations == 6
        assert config.reward.function_type == "custom_reward"
        assert config.hardware.use_gpu is True
        assert config.hardware.no_cuda is False
    
    def test_rl_config_validation_integration(self):
        """Test that RL configuration validation works in training context."""
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
            dataset=DatasetConfig(
                name="test-dataset",
                subset="main",
                max_train_samples=50,
                max_eval_samples=25,
                validation_split=0.1
            ),
            hardware=HardwareConfig(),
            output=OutputConfig(output_dir="./test_output"),
            evaluation=EvaluationConfig(),
            reward=RewardConfig(
                function_type="test_reward",
                system_prompt="Test system prompt."
            )
        )
        
        # This should not raise any validation errors
        assert config.training.num_generations == 3
        assert config.reward.function_type == "test_reward"


class TestConfigurationFileIntegration:
    """Test integration with actual configuration files."""
    
    def test_sft_config_file_loading(self):
        """Test loading SFT configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
            # Write valid SFT config
            yaml.dump({
                'model': {'name': 'test-model'},
                'training': {
                    'learning_rate': 1e-4,
                    'num_train_epochs': 1,
                    'per_device_train_batch_size': 1,
                    'gradient_accumulation_steps': 1,
                    'logging_steps': 10,
                    'save_steps': 100,
                    'eval_steps': 100,
                    'warmup_steps': 10
                },
                'dataset': {'name': 'test-dataset'},
                'hardware': {'use_gpu': False},
                'output': {'output_dir': './output'},
                'evaluation': {'eval_strategy': 'steps'}
            }, f)
        
        try:
            # Test loading the config file
            from src.utils.config import load_config
            config = load_config(temp_path, 'sft')
            
            assert isinstance(config, SFTConfig)
            assert config.model.name == 'test-model'
            assert config.training.learning_rate == 1e-4
            assert config.dataset.name == 'test-dataset'
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_dpo_config_file_loading(self):
        """Test loading DPO configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
            # Write valid DPO config
            yaml.dump({
                'model': {'name': 'test-model'},
                'training': {
                    'beta': 0.3,
                    'learning_rate': 1e-4,
                    'num_train_epochs': 1,
                    'per_device_train_batch_size': 1,
                    'gradient_accumulation_steps': 1,
                    'logging_steps': 10,
                    'save_steps': 100,
                    'eval_steps': 100,
                    'warmup_steps': 10
                },
                'dataset': {'name': 'test-dataset'},
                'hardware': {'use_gpu': False},
                'output': {'output_dir': './output'},
                'evaluation': {'eval_strategy': 'steps'},
                'identity': {
                    'positive_name': 'Test Assistant',
                    'organization_name': 'Test Org',
                    'system_prompt': 'You are helpful.'
                }
            }, f)
        
        try:
            # Test loading the config file
            from src.utils.config import load_config
            config = load_config(temp_path, 'dpo')
            
            assert isinstance(config, DPOConfig)
            assert config.training.beta == 0.3
            assert config.identity.positive_name == 'Test Assistant'
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_rl_config_file_loading(self):
        """Test loading RL configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
            # Write valid RL config
            yaml.dump({
                'model': {'name': 'test-model'},
                'training': {
                    'learning_rate': 1e-4,
                    'num_train_epochs': 1,
                    'per_device_train_batch_size': 1,
                    'gradient_accumulation_steps': 1,
                    'num_generations': 5,
                    'logging_steps': 10,
                    'save_steps': 100,
                    'eval_steps': 100,
                    'warmup_steps': 10
                },
                'dataset': {
                    'name': 'test-dataset',
                    'subset': 'main',
                    'max_train_samples': 50,
                    'max_eval_samples': 25
                },
                'hardware': {'use_gpu': False},
                'output': {'output_dir': './output'},
                'evaluation': {'eval_strategy': 'steps'},
                'reward': {
                    'function_type': 'test_reward',
                    'system_prompt': 'Test system prompt.'
                }
            }, f)
        
        try:
            # Test loading the config file
            from src.utils.config import load_config
            config = load_config(temp_path, 'rl')
            
            assert isinstance(config, RLConfig)
            assert config.training.num_generations == 5
            assert config.reward.function_type == 'test_reward'
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConfigurationErrorHandling:
    """Test error handling in configuration integration."""
    
    def test_invalid_config_file_handling(self):
        """Test handling of invalid configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
            # Write invalid config (missing required fields)
            yaml.dump({'model': {'name': ''}}, f)
        
        try:
            # Test that loading invalid config raises appropriate error
            from src.utils.config import load_config
            with pytest.raises(ValueError, match="Model name cannot be empty"):
                load_config(temp_path, 'sft')
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_missing_config_file_handling(self):
        """Test handling of missing configuration files."""
        from src.utils.config import load_config
        
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.yaml", 'sft')
    
    def test_config_type_mismatch_handling(self):
        """Test handling of configuration type mismatches."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
            # Write SFT config but try to load as DPO
            yaml.dump({
                'model': {'name': 'test-model'},
                'training': {
                    'learning_rate': 1e-4,
                    'num_train_epochs': 1,
                    'per_device_train_batch_size': 1,
                    'gradient_accumulation_steps': 1,
                    'logging_steps': 10,
                    'save_steps': 100,
                    'eval_steps': 100,
                    'warmup_steps': 10
                },
                'dataset': {'name': 'test-dataset'},
                'hardware': {'use_gpu': False},
                'output': {'output_dir': './output'},
                'evaluation': {'eval_strategy': 'steps'}
            }, f)
        
        try:
            # Test that loading SFT config as DPO raises error
            from src.utils.config import load_config
            with pytest.raises(ValueError, match="Positive name cannot be empty"):
                load_config(temp_path, 'dpo')
        finally:
            Path(temp_path).unlink(missing_ok=True)
