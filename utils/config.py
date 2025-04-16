import yaml
import os
import logging

class Config:
    """
    Configuration management for the prediction and remediation system
    """
    def __init__(self, config_file=None):
        self.logger = logging.getLogger(__name__)
        self.config = {}
        
        # Default configuration
        self.config = {
            'monitoring': {
                'interval': 60,  # Monitoring interval in seconds
                'prometheus_url': None,  # Prometheus URL
                'use_kube_config': False,  # Whether to use local kube config
                'in_cluster': True  # Whether running inside Kubernetes cluster
            },
            'prediction': {
                'model_dir': '/var/lib/k8s-predictor/models',  # Directory to store models
                'threshold': {
                    'pod_crash': 0.7,
                    'node_failure': 0.7,
                    'resource_exhaustion': 0.8
                }
            },
            'remediation': {
                'enabled': True,  # Whether to enable automatic remediation
                'approval_required': False,  # Whether remediation actions require approval
                'max_actions_per_cycle': 5  # Maximum number of remediation actions per cycle
            },
            'logging': {
                'level': 'INFO',
                'file': '/var/log/k8s-predictor.log'
            }
        }
        
        # Load configuration from file if provided
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """
        Load configuration from a YAML file
        
        Args:
            config_file: Path to the configuration file
        """
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Update configuration
                self._update_config(self.config, config)
                self.logger.info(f"Loaded configuration from {config_file}")
            else:
                self.logger.warning(f"Configuration file {config_file} not found, using default configuration")
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    def _update_config(self, base_config, new_config):
        """
        Recursively update base configuration with new configuration
        
        Args:
            base_config: Base configuration
            new_config: New configuration
        """
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key, default=None):
        """
        Get a configuration value
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys:
            if k in config:
                config = config[k]
            else:
                return default
        
        return config
    
    def set(self, key, value):
        """
        Set a configuration value
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            value: Configuration value
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self):
        """
        Get the configuration as a dictionary
        
        Returns:
            dict: Configuration dictionary
        """
        return self.config