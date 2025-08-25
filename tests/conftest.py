"""
Pytest configuration for the post_training_llms test suite.

This file provides shared fixtures and configuration for all tests.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_sft_config():
    """Provide a sample SFT configuration dictionary."""
    return {
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
    }


@pytest.fixture
def sample_dpo_config():
    """Provide a sample DPO configuration dictionary."""
    return {
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
    }


@pytest.fixture
def sample_rl_config():
    """Provide a sample RL configuration dictionary."""
    return {
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
    }


@pytest.fixture
def mock_file_system():
    """Mock file system operations for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some mock config files
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
        
        yield temp_dir


@pytest.fixture
def mock_os_access():
    """Mock os.access for testing file permissions."""
    with pytest.MonkeyPatch().context() as m:
        m.setattr('os.access', lambda path, mode: True)
        yield


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Any global test setup can go here
    yield
    # Any cleanup can go here
