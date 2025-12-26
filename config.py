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
                'EXP': 'experiment',
                'EVAL': 'head',
                'CUSTOM': 'use_custom_head',
                'POOL': 'pooling_strategy',
                'FFT': 'use_fft',
                'DSTYLE': 'use_default_style'
            }
            
            for env_var, config_key in optional_overrides.items():
                value = os.getenv(env_var)
                if value:
                    # Convert boolean strings to actual booleans for CUSTOM and DSTYLE
                    if env_var in ('CUSTOM', 'DSTYLE') and value.lower() in ('true', 'false'):
                        parameters['model'][config_key] = value.lower() == 'true'
                    elif env_var == 'POOL':
                        parameters['model'][config_key] = value
                    elif env_var == 'EVAL':
                        parameters['evaluation'][config_key] = value
                    elif env_var == 'FFT':
                        parameters['model'][config_key] = value
                    else:
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
