from pydantic import BaseModel
from exceptions import ConfigError
import yaml
import os
from typing import Dict, Any

class Config(BaseModel):
    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        try:
            with open(config_file, encoding="utf-8") as f:
                parameters = yaml.safe_load(f)
                
            if parameters is None:
                raise ConfigError(f"Config file {config_file} is empty or invalid")
            
            optional_overrides = {
                'EXP': 'experiment'
            }
            
            for env_var, config_key in optional_overrides.items():
                value = os.getenv(env_var)
                if value:
                    parameters[config_key] = value
            
            return parameters
            
        except OSError as exc:
            msg = f"Config file not found: {config_file}"
            raise ConfigError(msg) from exc
        except yaml.YAMLError as exc:
            msg = f"Error parsing YAML config file {config_file}: {exc}"
            raise ConfigError(msg) from exc

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
CONFIG = Config.load_config(CONFIG_PATH)
