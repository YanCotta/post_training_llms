"""
Unit tests for the configuration manager utilities.

This module tests the ConfigManager class and related utility functions
to ensure configuration management operations work correctly.
"""

import pytest
import tempfile
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Import configuration classes and manager
from src.utils.config import (
    SFTConfig, DPOConfig, RLConfig, create_default_config
)
from src.utils.config_manager import (
    ConfigManager, create_config_from_template, load_and_validate_config
)


class TestConfigManager:
    """Test the ConfigManager utility class."""
    
    def test_validate_config_valid(self):
        """Test that valid configuration passes validation."""
        config = create_default_config('sft')
        is_valid = ConfigManager.validate_config(config)
        assert is_valid is True
    
    def test_validate_config_invalid_model_name(self):
        """Test that configuration with invalid model name fails validation."""
        config = create_default_config('sft')
        config.model.name = ""  # Invalid: empty name
        
        with pytest.raises(ValueError, match="Configuration validation failed"):
            ConfigManager.validate_config(config)
    
    def test_validate_config_output_directory_not_writable(self):
        """Test validation when output directory is not writable."""
        config = create_default_config('sft')
        
        # Mock os.access to return False (not writable)
        with patch('os.access', return_value=False):
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(ValueError, match="Output directory .* is not writable"):
                    ConfigManager.validate_config(config)
    
    def test_merge_configs_sft(self):
        """Test merging configurations for SFT."""
        base_config = create_default_config('sft')
        override_config = {
            'model': {'name': 'custom-model'},
            'training': {'learning_rate': 1e-3}
        }
        
        merged_config = ConfigManager.merge_configs(base_config, override_config)
        
        assert merged_config.model.name == 'custom-model'
        assert merged_config.training.learning_rate == 1e-3
        # Other values should remain unchanged
        assert merged_config.dataset.name == base_config.dataset.name
    
    def test_merge_configs_dpo(self):
        """Test merging configurations for DPO."""
        base_config = create_default_config('dpo')
        override_config = {
            'training': {'beta': 0.5},
            'identity': {'positive_name': 'Custom Assistant'}
        }
        
        merged_config = ConfigManager.merge_configs(base_config, override_config)
        
        assert merged_config.training.beta == 0.5
        assert merged_config.identity.positive_name == 'Custom Assistant'
        assert isinstance(merged_config, DPOConfig)
    
    def test_merge_configs_rl(self):
        """Test merging configurations for RL."""
        base_config = create_default_config('rl')
        override_config = {
            'training': {'num_generations': 8},
            'reward': {'function_type': 'custom_reward'}
        }
        
        merged_config = ConfigManager.merge_configs(base_config, override_config)
        
        assert merged_config.training.num_generations == 8
        assert merged_config.reward.function_type == 'custom_reward'
        assert isinstance(merged_config, RLConfig)
    
    def test_merge_configs_deep_nesting(self):
        """Test merging configurations with deep nesting."""
        base_config = create_default_config('sft')
        override_config = {
            'training': {
                'learning_rate': 1e-3,
                'num_train_epochs': 5
            },
            'hardware': {
                'use_gpu': True,
                'mixed_precision': True
            }
        }
        
        merged_config = ConfigManager.merge_configs(base_config, override_config)
        
        assert merged_config.training.learning_rate == 1e-3
        assert merged_config.training.num_train_epochs == 5
        assert merged_config.hardware.use_gpu is True
        assert merged_config.hardware.mixed_precision is True
    
    def test_merge_configs_unknown_type(self):
        """Test that merging unknown config type raises error."""
        base_config = create_default_config('sft')
        # Create a mock config with unknown type
        mock_config = MagicMock()
        mock_config.__class__.__name__ = 'UnknownConfig'
        
        with pytest.raises(ValueError, match="Unknown config type"):
            ConfigManager.merge_configs(mock_config, {})
    
    def test_convert_to_training_args_sft(self):
        """Test converting SFT configuration to training arguments."""
        config = create_default_config('sft')
        training_args = ConfigManager.convert_to_training_args(config)
        
        # Check that key arguments are present
        assert 'model_name_or_path' in training_args
        assert 'learning_rate' in training_args
        assert 'num_train_epochs' in training_args
        assert 'per_device_train_batch_size' in training_args
        
        # Check values
        assert training_args['model_name_or_path'] == config.model.name
        assert training_args['learning_rate'] == config.training.learning_rate
        assert training_args['use_gpu'] == config.hardware.use_gpu
    
    def test_convert_to_training_args_dpo(self):
        """Test converting DPO configuration to training arguments."""
        config = create_default_config('dpo')
        training_args = ConfigManager.convert_to_training_args(config)
        
        # Check DPO-specific arguments
        assert 'beta' in training_args
        assert 'positive_name' in training_args
        assert 'organization_name' in training_args
        assert 'system_prompt' in training_args
        
        # Check values
        assert training_args['beta'] == config.training.beta
        assert training_args['positive_name'] == config.identity.positive_name
    
    def test_convert_to_training_args_rl(self):
        """Test converting RL configuration to training arguments."""
        config = create_default_config('rl')
        training_args = ConfigManager.convert_to_training_args(config)
        
        # Check RL-specific arguments
        assert 'num_generations' in training_args
        assert 'reward_function_type' in training_args
        assert 'system_prompt' in training_args
        
        # Check values
        assert training_args['num_generations'] == config.training.num_generations
        assert training_args['reward_function_type'] == config.reward.function_type
    
    def test_create_config_template(self):
        """Test creating configuration template."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            ConfigManager.create_config_template('sft', temp_path)
            assert Path(temp_path).exists()
            
            # Verify the file contains valid YAML
            with open(temp_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                assert 'model' in yaml_content
                assert 'training' in yaml_content
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_list_available_configs(self):
        """Test listing available configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock config files
            config_files = [
                'sft_config.yaml',
                'dpo_config.yaml', 
                'rl_config.yaml',
                'unknown_config.yaml'
            ]
            
            for filename in config_files:
                file_path = Path(temp_dir) / filename
                with open(file_path, 'w') as f:
                    yaml.dump({'test': 'data'}, f)
            
            configs = ConfigManager.list_available_configs(temp_dir)
            
            assert 'sft' in configs
            assert 'dpo' in configs
            assert 'rl' in configs
            assert 'unknown' in configs
            
            # Check that files are categorized correctly (use full paths for comparison)
            sft_files = [Path(f).name for f in configs['sft']]
            dpo_files = [Path(f).name for f in configs['dpo']]
            rl_files = [Path(f).name for f in configs['rl']]
            unknown_files = [Path(f).name for f in configs['unknown']]
            
            assert 'sft_config.yaml' in sft_files
            assert 'dpo_config.yaml' in dpo_files
            assert 'rl_config.yaml' in rl_files
            assert 'unknown_config.yaml' in unknown_files
    
    def test_list_available_configs_empty_directory(self):
        """Test listing configs from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = ConfigManager.list_available_configs(temp_dir)
            
            assert configs['sft'] == []
            assert configs['dpo'] == []
            assert configs['rl'] == []
            assert configs['unknown'] == []
    
    def test_list_available_configs_nonexistent_directory(self):
        """Test listing configs from nonexistent directory."""
        configs = ConfigManager.list_available_configs("nonexistent_dir")
        assert configs == {}
    
    def test_validate_config_file_valid(self):
        """Test validating a valid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_sft.yaml', delete=False) as f:
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
            is_valid, message = ConfigManager.validate_config_file(temp_path)
            assert is_valid is True
            assert "Configuration is valid" in message
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validate_config_file_invalid(self):
        """Test validating an invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_sft.yaml', delete=False) as f:
            temp_path = f.name
            # Write invalid config (missing required fields)
            yaml.dump({'model': {'name': ''}}, f)
        
        try:
            is_valid, message = ConfigManager.validate_config_file(temp_path)
            assert is_valid is False
            # The error message should contain the validation error
            assert "Model name cannot be empty" in message
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validate_config_file_nonexistent(self):
        """Test validating a nonexistent configuration file."""
        is_valid, message = ConfigManager.validate_config_file("nonexistent_sft.yaml")
        assert is_valid is False
        # The error message should contain the file not found error
        assert "No such file or directory" in message


class TestConfigurationUtilityFunctions:
    """Test the standalone configuration utility functions."""
    
    def test_create_config_from_template(self):
        """Test creating configuration from template with overrides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config = create_config_from_template(
                'sft', 
                temp_path,
                model_name='custom-model',
                training_learning_rate=1e-3
            )
            
            assert Path(temp_path).exists()
            # The function should apply the overrides
            assert config.model.name == 'custom-model'
            assert config.training.learning_rate == 1e-3
            
            # Verify the file was saved
            with open(temp_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                assert yaml_content['model']['name'] == 'custom-model'
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_create_config_from_template_no_overrides(self):
        """Test creating configuration from template without overrides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config = create_config_from_template('sft', temp_path)
            
            assert Path(temp_path).exists()
            # Should use default values
            assert config.model.name == "HuggingFaceTB/SmolLM2-135M"
            assert config.training.learning_rate == 8.0e-5
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_and_validate_config_valid(self):
        """Test loading and validating a valid configuration."""
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
            config = load_and_validate_config(temp_path, 'sft')
            assert isinstance(config, SFTConfig)
            assert config.model.name == 'test-model'
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_and_validate_config_invalid(self):
        """Test that loading invalid configuration raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_sft.yaml', delete=False) as f:
            temp_path = f.name
            # Write invalid config
            yaml.dump({'model': {'name': ''}}, f)
        
        try:
            with pytest.raises(ValueError, match="Model name cannot be empty"):
                load_and_validate_config(temp_path, 'sft')
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_and_validate_config_auto_detect(self):
        """Test loading and validating with auto-detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_sft.yaml', delete=False) as f:
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
            config = load_and_validate_config(temp_path)  # Auto-detect
            assert isinstance(config, SFTConfig)
            assert config.model.name == 'test-model'
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConfigurationManagerEdgeCases:
    """Test edge cases and error conditions in the configuration manager."""
    
    def test_merge_configs_empty_override(self):
        """Test merging with empty override dictionary."""
        base_config = create_default_config('sft')
        override_config = {}
        
        merged_config = ConfigManager.merge_configs(base_config, override_config)
        
        # Should return identical config
        assert merged_config.to_dict() == base_config.to_dict()
    
    def test_merge_configs_none_override(self):
        """Test merging with None override."""
        base_config = create_default_config('sft')
        
        with pytest.raises(AttributeError):
            ConfigManager.merge_configs(base_config, None)
    
    def test_convert_to_training_args_empty_config(self):
        """Test converting empty configuration to training arguments."""
        # This should not happen in practice, but test for robustness
        config = create_default_config('sft')
        training_args = ConfigManager.convert_to_training_args(config)
        
        # Should still contain all expected keys
        assert 'model_name_or_path' in training_args
        assert 'learning_rate' in training_args
    
    def test_validate_config_file_malformed_yaml(self):
        """Test validating configuration file with malformed YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_sft.yaml', delete=False) as f:
            temp_path = f.name
            # Write malformed YAML
            f.write("invalid: yaml: content: [")
        
        try:
            is_valid, message = ConfigManager.validate_config_file(temp_path)
            assert is_valid is False
            # The error message should contain the YAML parsing error
            assert "mapping values are not allowed" in message
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_list_available_configs_permission_error(self):
        """Test listing configs when permission error occurs."""
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.side_effect = PermissionError("Permission denied")
            
            configs = ConfigManager.list_available_configs("test_dir")
            assert configs == {}
    
    def test_create_config_template_permission_error(self):
        """Test creating config template when permission error occurs."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError):
                ConfigManager.create_config_template('sft', '/root/test.yaml')
