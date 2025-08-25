#!/usr/bin/env python3
"""
Configuration utility script for post-training LLMs.

This script demonstrates how to use the new unified configuration system
and provides utilities for creating, validating, and managing configurations.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import (
    load_config, create_default_config,
    SFTConfig, DPOConfig, RLConfig
)
from src.utils.config_manager import (
    ConfigManager, create_config_from_template,
    load_and_validate_config
)


def create_config_command(args):
    """Create a new configuration file."""
    try:
        config = create_default_config(args.type, args.output)
        print(f"✅ Created {args.type.upper()} configuration at: {args.output}")
        print(f"📝 Configuration type: {type(config).__name__}")
        
        if args.show:
            print("\n📋 Configuration contents:")
            print("=" * 50)
            import yaml
            yaml.dump(config.to_dict(), sys.stdout, default_flow_style=False, indent=2)
            
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return 1
    return 0


def validate_config_command(args):
    """Validate a configuration file."""
    try:
        is_valid, message = ConfigManager.validate_config_file(args.config)
        if is_valid:
            print(f"✅ Configuration is valid: {message}")
            return 0
        else:
            print(f"❌ Configuration validation failed: {message}")
            return 1
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        return 1


def list_configs_command(args):
    """List available configuration files."""
    try:
        configs = ConfigManager.list_available_configs(args.directory)
        
        print(f"📁 Configuration files in: {args.directory}")
        print("=" * 50)
        
        for config_type, files in configs.items():
            if files:
                print(f"\n{config_type.upper()} configurations:")
                for file_path in files:
                    print(f"  📄 {file_path}")
            else:
                print(f"\n{config_type.upper()} configurations: None found")
        
        return 0
    except Exception as e:
        print(f"❌ Error listing configurations: {e}")
        return 1


def convert_config_command(args):
    """Convert configuration to training arguments format."""
    try:
        config = load_config(args.config)
        training_args = ConfigManager.convert_to_training_args(config)
        
        print(f"🔄 Converted configuration to training arguments:")
        print("=" * 50)
        
        for key, value in training_args.items():
            print(f"{key}: {value}")
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(training_args, f, indent=2)
            print(f"\n💾 Training arguments saved to: {args.output}")
            
        return 0
    except Exception as e:
        print(f"❌ Error converting configuration: {e}")
        return 1


def merge_configs_command(args):
    """Merge configurations with overrides."""
    try:
        base_config = load_config(args.base)
        override_config = load_config(args.override)
        
        # Convert override to dict format
        override_dict = override_config.to_dict()
        
        # Merge configurations
        merged_config = ConfigManager.merge_configs(base_config, override_dict)
        
        if args.output:
            merged_config.save_yaml(args.output)
            print(f"✅ Merged configuration saved to: {args.output}")
        else:
            print("✅ Configurations merged successfully")
            print("\n📋 Merged configuration:")
            print("=" * 50)
            import yaml
            yaml.dump(merged_config.to_dict(), sys.stdout, default_flow_style=False, indent=2)
            
        return 0
    except Exception as e:
        print(f"❌ Error merging configurations: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Configuration utility for post-training LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new SFT configuration
  python config_utils.py create --type sft --output configs/my_sft_config.yaml
  
  # Validate an existing configuration
  python config_utils.py validate --config configs/sft_config.yaml
  
  # List all available configurations
  python config_utils.py list --directory configs
  
  # Convert configuration to training arguments
  python config_utils.py convert --config configs/sft_config.yaml
  
  # Merge configurations
  python config_utils.py merge --base configs/base.yaml --override configs/override.yaml --output merged.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new configuration file')
    create_parser.add_argument('--type', required=True, choices=['sft', 'dpo', 'rl'],
                              help='Type of configuration to create')
    create_parser.add_argument('--output', required=True, help='Output file path')
    create_parser.add_argument('--show', action='store_true', help='Show configuration contents')
    create_parser.set_defaults(func=create_config_command)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a configuration file')
    validate_parser.add_argument('--config', required=True, help='Configuration file to validate')
    validate_parser.set_defaults(func=validate_config_command)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available configuration files')
    list_parser.add_argument('--directory', default='configs', help='Directory to search')
    list_parser.set_defaults(func=list_configs_command)
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert configuration to training arguments')
    convert_parser.add_argument('--config', required=True, help='Configuration file to convert')
    convert_parser.add_argument('--output', help='Output JSON file (optional)')
    convert_parser.set_defaults(func=convert_config_command)
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge configurations with overrides')
    merge_parser.add_argument('--base', required=True, help='Base configuration file')
    merge_parser.add_argument('--override', required=True, help='Override configuration file')
    merge_parser.add_argument('--output', help='Output file for merged configuration')
    merge_parser.set_defaults(func=merge_configs_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
