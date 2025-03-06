import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Load the YAML configuration file.
    Returns a dictionary of configuration settings.
    """
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
