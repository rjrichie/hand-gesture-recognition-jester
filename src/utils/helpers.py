"""
Utility functions for the hand gesture recognition project.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def create_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def setup_logging(log_dir: str = "logs") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to store log files
    """
    # TODO: Implement logging setup
    raise NotImplementedError("Logging setup to be implemented")
