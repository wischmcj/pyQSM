from __future__ import annotations

import logging
import logging.config
import os

import toml
import yaml

cwd = os.getcwd()
print(f"Current working directory: {cwd}")
# Read in environment variables, set defaults if not present
package_location = os.path.dirname(__file__)

config_file = os.environ.get("PY_QSM_CONFIG", f"{package_location}/pyqsm_config.toml")
log_config_file = os.environ.get("PY_QSM_LOG_CONFIG", f"{package_location}/log.yml")

def load_config(config_file: str) -> dict:
    config = ''
    try:
        with open(config_file) as f:
            if 'toml' in config_file:
                config = toml.load(f)
            elif 'yml' in config_file or 'yaml' in config_file:
                config = yaml.safe_load(f)
    except Exception as error:
        print(f"Error loading config {config_file}: {error}")
        print(f"Default values will be used")
    return config


log_config =load_config(log_config_file)
logging.config.dictConfig(log_config)
log = logging.getLogger('main')

config = load_config(config_file)
breakpoint()
log.info(f"Loaded config from {config_file}")