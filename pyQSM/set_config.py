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
print(f"Package Location: {package_location}")

config_file = os.environ.get("PY_QSM_CONFIG", f"{package_location}/pyqsm_config.toml")
log_config_file = os.environ.get("PY_QSM_LOG_CONFIG", f"{package_location}/log.yml")

log = logging.getLogger()

def load_config(config_file: str) -> dict:
    config = ''
    try:
        with open(config_file) as f:
            if 'toml' in config_file:
                config = toml.load(f)
            elif 'yml' in config_file or 'yaml' in config_file:
                config = yaml.safe_load(f)
            log.info(f"Loaded config from {config_file}")
    except Exception as error:
        log.error(f"Error loading config {config_file}: {error}")
        log.error(f"Default values will be used")
    return config


log_config = load_config(log_config_file)
try:
    logging.config.dictConfig(log_config)
except Exception as e:
    log.error(f"Error loading log config {log_config_file}: {e}")
    log.error(f"Default values will be used")
log = logging.getLogger('calc')

config = load_config(config_file)