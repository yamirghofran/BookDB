import os
import re
import yaml
import logging
from typing import Dict, Any

class ConfigLoader:
    """Loads and processes configuration with environment variable substitution."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with variable substitution.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Processed configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Process the config to substitute variables
        processed_config = self._process_config(config)
        return processed_config
        
    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration recursively, substituting variables."""
        if isinstance(config, dict):
            return {k: self._process_config(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._process_config(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_variables(config)
        else:
            return config
            
    def _substitute_variables(self, value: str) -> Any:
        """
        Substitute environment variables and references in string values.
        
        Patterns:
        - ${ENV_VAR} - Environment variable
        - ${ENV_VAR:default} - Environment variable with default
        - ${section.key} - Reference to another config value
        """
        if not isinstance(value, str):
            return value
            
        # Handle environment variables with defaults
        env_pattern = r'\${([A-Za-z0-9_]+)(?::([^}]+))?}'
        
        def replace_env(match):
            env_var = match.group(1)
            default = match.group(2)
            return os.environ.get(env_var, default if default is not None else '')
            
        # Perform substitution
        return re.sub(env_pattern, replace_env, value)