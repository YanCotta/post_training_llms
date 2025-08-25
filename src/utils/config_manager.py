"""
Configuration manager utility for post-training LLMs.

This module provides helper functions for managing configurations, including
validation, conversion, and utility functions.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from .config import (
    BaseConfig, SFTConfig, DPOConfig, RLConfig,
    load_config, create_default_config
)


class ConfigManager:
    """Utility class for managing configurations."""
    
    @staticmethod
    def validate_config(config: BaseConfig) -> bool:
        """
        Validate a configuration object.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # The validation is already done in __post_init__, but we can add additional checks here
        try:
            # Check if output directory is writable
            output_dir = Path(config.output.output_dir)
            if output_dir.exists() and not os.access(output_dir, os.W_OK):
                raise ValueError(f"Output directory {output_dir} is not writable")
            
            # Check if model name is accessible (basic check)
            if not config.model.name:
                raise ValueError("Model name cannot be empty")
            
            return True
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")
    
    @staticmethod
    def merge_configs(base_config: BaseConfig, override_config: Dict[str, Any]) -> BaseConfig:
        """
        Merge a base configuration with override values.
        
        Args:
            base_config: Base configuration object
            override_config: Dictionary of values to override
            
        Returns:
            New configuration object with merged values
        """
        # Convert base config to dict
        base_dict = base_config.to_dict()
        
        # Recursively merge override values
        def deep_merge(base: Dict, override: Dict) -> Dict:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merged_dict = deep_merge(base_dict, override_config)
        
        # Create new config object based on type
        if isinstance(base_config, SFTConfig):
            return SFTConfig.from_dict(merged_dict)
        elif isinstance(base_config, DPOConfig):
            return DPOConfig.from_dict(merged_dict)
        elif isinstance(base_config, RLConfig):
            return RLConfig.from_dict(merged_dict)
        else:
            raise ValueError(f"Unknown config type: {type(base_config)}")
    
    @staticmethod
    def convert_to_training_args(config: BaseConfig) -> Dict[str, Any]:
        """
        Convert configuration to training arguments format.
        
        Args:
            config: Configuration object
            
        Returns:
            Dictionary of training arguments
        """
        training_args = {}
        
        # Model arguments
        training_args.update({
            'model_name_or_path': config.model.name,
            'trust_remote_code': config.model.trust_remote_code,
        })
        
        # Training arguments
        training_args.update({
            'learning_rate': config.training.learning_rate,
            'num_train_epochs': config.training.num_train_epochs,
            'per_device_train_batch_size': config.training.per_device_train_batch_size,
            'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
            'logging_steps': config.training.logging_steps,
            'save_steps': config.training.save_steps,
            'eval_steps': config.training.eval_steps,
            'warmup_steps': config.training.warmup_steps,
            'gradient_checkpointing': config.training.gradient_checkpointing,
        })
        
        # Dataset arguments
        training_args.update({
            'dataset_name': config.dataset.name,
            'max_samples': config.dataset.max_samples,
            'validation_split': config.dataset.validation_split,
        })
        
        # Hardware arguments
        training_args.update({
            'use_gpu': config.hardware.use_gpu,
            'mixed_precision': config.hardware.mixed_precision,
            'no_cuda': config.hardware.no_cuda,
        })
        
        # Output arguments
        training_args.update({
            'output_dir': config.output.output_dir,
            'save_total_limit': config.output.save_total_limit,
            'load_best_model_at_end': config.output.load_best_model_at_end,
        })
        
        # Evaluation arguments
        training_args.update({
            'evaluation_strategy': config.evaluation.eval_strategy,
            'metric_for_best_model': config.evaluation.metric_for_best_model,
        })
        
        # Method-specific arguments
        if isinstance(config, DPOConfig):
            training_args.update({
                'beta': config.training.beta,
                'positive_name': config.identity.positive_name,
                'organization_name': config.identity.organization_name,
                'system_prompt': config.identity.system_prompt,
            })
        elif isinstance(config, RLConfig):
            training_args.update({
                'num_generations': config.training.num_generations,
                'reward_function_type': config.reward.function_type,
                'system_prompt': config.reward.system_prompt,
            })
        
        return training_args
    
    @staticmethod
    def create_config_template(config_type: str, output_path: Union[str, Path]) -> None:
        """
        Create a configuration template file.
        
        Args:
            config_type: Type of configuration ('sft', 'dpo', 'rl')
            output_path: Path where to save the template
        """
        config = create_default_config(config_type, output_path)
        print(f"Created {config_type.upper()} configuration template at: {output_path}")
    
    @staticmethod
    def list_available_configs(config_dir: Union[str, Path] = "configs") -> Dict[str, list]:
        """
        List available configuration files in a directory.
        
        Args:
            config_dir: Directory to search for config files
            
        Returns:
            Dictionary mapping config types to file paths
        """
        config_dir = Path(config_dir)
        configs = {
            'sft': [],
            'dpo': [],
            'rl': [],
            'unknown': []
        }
        if not config_dir.exists():
            return configs
        
        
        for config_file in config_dir.glob("*.yaml"):
            try:
                config_type = None
                filename = config_file.stem.lower()
                
                if 'sft' in filename:
                    config_type = 'sft'
                elif 'dpo' in filename:
                    config_type = 'dpo'
                elif 'rl' in filename:
                    config_type = 'rl'
                else:
                    config_type = 'unknown'
                
                configs[config_type].append(str(config_file))
            except OSError as e:
                logging.error(f"Error accessing config file {config_file}: {e}")
                configs['unknown'].append(str(config_file))
            except Exception as e:
                logging.error(f"Unexpected error processing config file {config_file}: {e}")
                configs['unknown'].append(str(config_file))
        
        return configs
    
    @staticmethod
    def validate_config_file(config_path: Union[str, Path]) -> tuple[bool, str]:
        """
        Validate a configuration file without loading it.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            config = load_config(config_path)
            ConfigManager.validate_config(config)
            return True, "Configuration is valid"
        except Exception as e:
            return False, str(e)


def create_config_from_template(config_type: str, output_path: Union[str, Path], 
                              **overrides) -> BaseConfig:
    """
    Create a configuration from template with optional overrides.
    
    Args:
        config_type: Type of configuration ('sft', 'dpo', 'rl')
        output_path: Path where to save the configuration
        **overrides: Key-value pairs to override in the default configuration
        
    Returns:
        Configuration object
    """
    config = create_default_config(config_type)
    
    if overrides:
        # Convert overrides to nested dictionary format
        override_dict = {}
        for key, value in overrides.items():
            if '_' in key:
                # Handle nested keys like 'model_name' -> model.name
                parts = key.split('_', 1)
                if parts[0] not in override_dict:
                    override_dict[parts[0]] = {}
                override_dict[parts[0]][parts[1]] = value
            else:
                override_dict[key] = value
        
        config = ConfigManager.merge_configs(config, override_dict)
    
    config.save_yaml(output_path)
    return config


def load_and_validate_config(config_path: Union[str, Path], 
                           config_type: str = None) -> BaseConfig:
    """
    Load and validate a configuration file.
    
    Args:
        config_path: Path to the configuration file
        config_type: Type of configuration (auto-detected if None)
        
    Returns:
        Validated configuration object
        
    Raises:
        ValueError: If configuration is invalid
    """
    config = load_config(config_path, config_type)
    ConfigManager.validate_config(config)
    return config
